# -*- coding: utf-8 -*-
"""
Implemented strategy Monte Carlo Tree Search
Using previously trained model as heuristik
"""

import math
import os
import random
import time
from train import GoValueNetwork

import Goban
from playerInterface import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TimeOut(Exception):
    pass


class _MCTSNode:
    __slots__ = (
        "parent",
        "move",
        "player",
        "unexpanded",
        "children",
        "visits",
        "value_sum",
    )

    def __init__(self, parent, move, player, moves):
        self.parent = parent
        self.move = move
        self.player = player
        self.unexpanded = list(moves)
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    def mean_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class myPlayer(PlayerInterface):
    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self._opponent = None

        # time control
        self._total_budget = 45.0 * 60.0
        self._used_time = 0.0
        self._safety_margin = 2.0
        self._deadline = 0.0

        # MCTS
        self._c_uct = 1.30
        self._value_model = None
        self._init_value_model()

    def getPlayerName(self):
        return "Kechna/Shrestha"

    def _neighbors(self, flat_coord):
        i = self._board._neighborsEntries[flat_coord]
        while self._board._neighbors[i] != -1:
            yield int(self._board._neighbors[i])
            i += 1

    def _candidate_moves(self):
        moves = self._board.weak_legal_moves()
        non_pass = [m for m in moves if m != -1]
        random.shuffle(non_pass)
        if -1 in moves:
            non_pass.append(-1)
        return non_pass

    def _init_value_model(self):
        model_path = os.path.join(os.path.dirname(__file__), "go_cnn_6_canaux_best.pt")
        model = GoValueNetwork()
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self._value_model = model

    def _board_to_planes(self, to_play):
        # Must match train.py encoding:
        # 0 white, 1 black, 2 turn(1 if black to play), 3 empty,
        # 4 empty-neighbor-of-black, 5 empty-neighbor-of-white.
        board = self._board.get_board()
        planes = np.zeros((6, 8, 8), dtype=np.float32)

        for fc, p in enumerate(board):
            x, y = Goban.Board.unflatten(fc)
            if p == Goban.Board._WHITE:
                planes[0, y, x] = 1.0
            elif p == Goban.Board._BLACK:
                planes[1, y, x] = 1.0

        if to_play == Goban.Board._BLACK:
            planes[2, :, :] = 1.0

        occupied = (planes[0] == 1.0) | (planes[1] == 1.0)
        planes[3] = (~occupied).astype(np.float32)

        for fc, p in enumerate(board):
            if p not in (Goban.Board._BLACK, Goban.Board._WHITE):
                continue
            x, y = Goban.Board.unflatten(fc)
            for n in self._neighbors(fc):
                nx, ny = Goban.Board.unflatten(n)
                if planes[3, ny, nx] != 1.0:
                    continue
                if p == Goban.Board._BLACK:
                    planes[4, ny, nx] = 1.0
                else:
                    planes[5, ny, nx] = 1.0
        return planes

    def _player_value(self, player):
        if self._board.is_game_over():
            winner = self._board.result_number()
            if winner == player:
                return 1.0
            if winner == Goban.Board.flip(player):
                return -1.0
            return 0.0

        planes = self._board_to_planes(player)
        x = torch.from_numpy(planes).unsqueeze(0)
        with torch.no_grad():
            # Model output interpreted as P(win for player-to-move).
            p = float(self._value_model(x).item())
        return 2.0 * p - 1.0

    def _check_time(self):
        if time.perf_counter() >= self._deadline:
            raise _TimeOut()

    def _select_child(self, node):
        log_n = math.log(max(1, node.visits))
        best_move = None
        best_child = None
        best_score = -math.inf

        items = list(node.children.items())
        random.shuffle(items)
        for move, child in items:
            q_child = child.mean_value()
            action_q = -q_child
            u = self._c_uct * math.sqrt(log_n / (1.0 + child.visits))
            score = action_q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _expand_one(self, node):
        """Expand one untried move"""
        while node.unexpanded:
            move = node.unexpanded.pop(0)
            ok = self._board.push(move)
            if not ok:
                self._board.pop()
                continue

            next_player = self._board.next_player()
            next_moves = self._candidate_moves()
            child = _MCTSNode(
                parent=node,
                move=move,
                player=next_player,
                moves=next_moves,
            )
            node.children[move] = child
            return child

        return None

    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _mcts_search(self, default_move):
        root_player = self._board.next_player()
        root_moves = self._candidate_moves()
        root = _MCTSNode(
            parent=None,
            move=None,
            player=root_player,
            moves=root_moves,
        )

        simulations = 0
        while True:
            self._check_time()
            simulations += 1

            node = root
            path = [node]
            pushed = 0

            # Selection
            while (
                not node.unexpanded and node.children and not self._board.is_game_over()
            ):
                move, child = self._select_child(node)
                if child is None:
                    break
                ok = self._board.push(move)
                if not ok:
                    self._board.pop()
                    del node.children[move]
                    continue
                pushed += 1
                node = child
                path.append(node)

            # Expansion
            if not self._board.is_game_over():
                child = self._expand_one(node)
                if child is not None:
                    pushed += 1
                    node = child
                    path.append(node)

            # Evaluation
            leaf_value = self._player_value(node.player)
            self._backpropagate(path, leaf_value)

            for _ in range(pushed):
                self._board.pop()

        # Normally unreachable ?
        return default_move, 0.0, simulations, root

    def _best_move_from_root(self, root, default_move):
        if not root.children:
            return default_move, 0.0, 0

        best_move = default_move
        best_visits = -1
        best_q = -math.inf
        total_visits = 0

        for move, child in root.children.items():
            total_visits += child.visits
            action_q = -child.mean_value()
            if child.visits > best_visits or (
                child.visits == best_visits and action_q > best_q
            ):
                best_visits = child.visits
                best_q = action_q
                best_move = move

        return best_move, best_q, total_visits

    # ------------------------------------------------------------------
    # Time management
    # ------------------------------------------------------------------

    def _allocate_turn_budget(self):
        remaining = self._total_budget - self._used_time - self._safety_margin
        if remaining <= 0.1:
            return 0.05

        empties = len(self._board._empties)
        estimated_own_turns_left = max(6, empties // 2)
        base = remaining / estimated_own_turns_left

        if empties > 42:
            factor = 0.9
        elif empties > 22:
            factor = 1.6
        else:
            factor = 2.2

        budget = base * factor
        return max(0.05, min(14, budget, 0.35 * remaining))

    # ------------------------------------------------------------------
    #
    # ------------------------------------------------------------------

    def getPlayerMove(self):
        t0 = time.perf_counter()
        try:
            if self._board.is_game_over():
                return "PASS"

            legal = self._board.legal_moves()
            if len(legal) == 1:
                only = legal[0]
                self._board.push(only)
                return Goban.Board.flat_to_name(only)

            # Move kept as fallback before MCTS gathers enough statistics.
            non_pass = [m for m in legal if m != -1]
            default_move = random.choice(non_pass) if non_pass else -1

            turn_budget = self._allocate_turn_budget()
            self._deadline = time.perf_counter() + turn_budget

            root_player = self._board.next_player()
            root_moves = self._candidate_moves()
            root = _MCTSNode(
                parent=None,
                move=None,
                player=root_player,
                moves=root_moves,
            )

            simulations = 0
            best_move = default_move
            best_q = 0.0
            root_visits = 0
            try:
                while True:
                    self._check_time()
                    simulations += 1

                    node = root
                    path = [node]
                    pushed = 0

                    while (
                        not node.unexpanded
                        and node.children
                        and not self._board.is_game_over()
                    ):
                        move, child = self._select_child(node)
                        if child is None:
                            break
                        ok = self._board.push(move)
                        if not ok:
                            self._board.pop()
                            del node.children[move]
                            continue
                        pushed += 1
                        node = child
                        path.append(node)

                    if not self._board.is_game_over():
                        child = self._expand_one(node)
                        if child is not None:
                            pushed += 1
                            node = child
                            path.append(node)

                    leaf_value = self._player_value(node.player)
                    self._backpropagate(path, leaf_value)
                    best_move, best_q, root_visits = self._best_move_from_root(
                        root, default_move
                    )

                    for _ in range(pushed):
                        self._board.pop()
            except _TimeOut:
                pass

            # Final read of the best move found so far at timeout.
            best_move, best_q, root_visits = self._best_move_from_root(root, best_move)
            if best_move not in legal:
                best_move = default_move if default_move in legal else legal[0]

            self._board.push(best_move)
            print(
                f"[{self.getPlayerName()}] move={Goban.Board.flat_to_name(best_move)} "
                f"sims={simulations} rootN={root_visits} q={best_q:.3f} "
                f"budget={turn_budget:.2f}s"
            )
            return Goban.Board.flat_to_name(best_move)
        finally:
            self._used_time += time.perf_counter() - t0

    def playOpponentMove(self, move):
        self._board.push(Goban.Board.name_to_flat(move))

    def newGame(self, color):
        self._board = Goban.Board()
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print(f"[{self.getPlayerName()}] I won.")
        else:
            print(f"[{self.getPlayerName()}] I lost.")
