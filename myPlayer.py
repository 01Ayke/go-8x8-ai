# -*- coding: utf-8 -*-
"""
MCTS player using the 6-channel CNN from proj1 as value heuristic.
Model outputs P(black wins); perspective is corrected per player.
"""

import math
import os
import random
import time

import Goban
from playerInterface import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Neural network (same architecture as proj1)
# ---------------------------------------------------------------------------

class GoValueNetwork(nn.Module):
    _BOARD = 8
    _CH = 6

    def __init__(self):
        super().__init__()
        B, C = self._BOARD, self._CH
        self.conv1 = nn.Conv2d(C, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.flat  = nn.Flatten()
        self.fc1   = nn.Linear(128 * B * B, 256)
        self.drop  = nn.Dropout(p=0.5)
        self.fc2   = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return torch.sigmoid(self.fc2(x))


# ---------------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------------

class _MCTSNode:
    __slots__ = ("parent", "move", "player", "unexpanded", "children",
                 "visits", "value_sum")

    def __init__(self, parent, move, player, moves):
        self.parent     = parent
        self.move       = move
        self.player     = player
        self.unexpanded = list(moves)
        self.children   = {}
        self.visits     = 0
        self.value_sum  = 0.0

    def mean_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class _TimeOut(Exception):
    pass


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class myPlayer(PlayerInterface):

    def __init__(self):
        self._board    = Goban.Board()
        self._mycolor  = None
        self._c_uct    = 1.30

        # time budget (seconds)
        self._total_budget  = 45.0 * 60.0
        self._used_time     = 0.0
        self._safety_margin = 2.0
        self._deadline      = 0.0

        self._model = self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        here    = os.path.dirname(os.path.abspath(__file__))
        proj1   = os.path.normpath(os.path.join(here, "../../proj1/go_cnn_6_canaux_best.pt"))
        local   = os.path.join(here, "go_cnn_6_canaux_best.pt")
        path    = proj1 if os.path.isfile(proj1) else local
        model   = GoValueNetwork()
        state   = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Board encoding
    # ------------------------------------------------------------------

    def _board_to_planes(self, to_play):
        """6-channel encoding matching the proj1 training format."""
        board  = self._board.get_board()
        planes = np.zeros((6, 8, 8), dtype=np.float32)

        for fc, p in enumerate(board):
            col, row = Goban.Board.unflatten(fc)
            if p == Goban.Board._WHITE:
                planes[0, row, col] = 1.0
            elif p == Goban.Board._BLACK:
                planes[1, row, col] = 1.0

        if to_play == Goban.Board._BLACK:
            planes[2, :, :] = 1.0          # turn channel: 1 = black to play

        occupied   = (planes[0] == 1.0) | (planes[1] == 1.0)
        planes[3]  = (~occupied).astype(np.float32)

        # empty neighbours of black (ch4) and white (ch5)
        for fc, p in enumerate(board):
            if p not in (Goban.Board._BLACK, Goban.Board._WHITE):
                continue
            for n in self._neighbors(fc):
                ncol, nrow = Goban.Board.unflatten(n)
                if planes[3, nrow, ncol] == 1.0:
                    ch = 4 if p == Goban.Board._BLACK else 5
                    planes[ch, nrow, ncol] = 1.0

        return planes

    def _neighbors(self, fc):
        i = self._board._neighborsEntries[fc]
        while self._board._neighbors[i] != -1:
            yield int(self._board._neighbors[i])
            i += 1

    # ------------------------------------------------------------------
    # Value evaluation
    # ------------------------------------------------------------------

    def _eval(self, player):
        """
        Returns a value in [-1, 1] from `player`'s perspective.
        Positive  → player is winning.
        """
        if self._board.is_game_over():
            winner = self._board.result_number()
            if winner == player:
                return 1.0
            if winner == Goban.Board.flip(player):
                return -1.0
            return 0.0

        planes = self._board_to_planes(player)
        x      = torch.from_numpy(planes).unsqueeze(0)
        with torch.no_grad():
            p_black_wins = float(self._model(x).item())

        # Model → P(black wins).  Convert to current-player advantage.
        v = 2.0 * p_black_wins - 1.0          # positive = black ahead
        return v if player == Goban.Board._BLACK else -v

    # ------------------------------------------------------------------
    # MCTS
    # ------------------------------------------------------------------

    def _candidate_moves(self):
        moves    = self._board.weak_legal_moves()
        non_pass = [m for m in moves if m != -1]
        random.shuffle(non_pass)
        if -1 in moves:
            non_pass.append(-1)
        return non_pass

    def _select_child(self, node):
        log_n      = math.log(max(1, node.visits))
        best_score = -math.inf
        best_move  = None
        best_child = None

        items = list(node.children.items())
        random.shuffle(items)
        for move, child in items:
            # UCT: value from parent's perspective = -child's value
            q     = -child.mean_value()
            u     = self._c_uct * math.sqrt(log_n / (1.0 + child.visits))
            score = q + u
            if score > best_score:
                best_score = score
                best_move  = move
                best_child = child

        return best_move, best_child

    def _expand_one(self, node):
        while node.unexpanded:
            move = node.unexpanded.pop(0)
            ok   = self._board.push(move)
            if not ok:
                self._board.pop()
                continue
            child = _MCTSNode(
                parent=node,
                move=move,
                player=self._board.next_player(),
                moves=self._candidate_moves(),
            )
            node.children[move] = child
            return child
        return None

    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.visits     += 1
            node.value_sum  += value
            value            = -value

    def _best_move(self, root, fallback):
        if not root.children:
            return fallback, 0.0, 0
        best_move   = fallback
        best_visits = -1
        best_q      = -math.inf
        total       = 0
        for move, child in root.children.items():
            total += child.visits
            q      = -child.mean_value()
            if child.visits > best_visits or (
                child.visits == best_visits and q > best_q
            ):
                best_visits = child.visits
                best_q      = q
                best_move   = move
        return best_move, best_q, total

    def _check_time(self):
        if time.perf_counter() >= self._deadline:
            raise _TimeOut()

    # ------------------------------------------------------------------
    # Time management
    # ------------------------------------------------------------------

    def _turn_budget(self):
        remaining = self._total_budget - self._used_time - self._safety_margin
        if remaining <= 0.1:
            return 0.05
        empties    = len(self._board._empties)
        own_turns  = max(6, empties // 2)
        base       = remaining / own_turns
        factor     = 0.9 if empties > 42 else (1.6 if empties > 22 else 2.2)
        return max(0.05, min(14.0, base * factor, 0.35 * remaining))

    # ------------------------------------------------------------------
    # PlayerInterface
    # ------------------------------------------------------------------

    def getPlayerName(self):
        return "Kechna/Shrestha"

    def getPlayerMove(self):
        t0 = time.perf_counter()
        try:
            if self._board.is_game_over():
                return "PASS"

            legal = self._board.legal_moves()
            if len(legal) == 1:
                self._board.push(legal[0])
                return Goban.Board.flat_to_name(legal[0])

            non_pass = [m for m in legal if m != -1]
            fallback = random.choice(non_pass) if non_pass else -1

            budget         = self._turn_budget()
            self._deadline = time.perf_counter() + budget

            root = _MCTSNode(
                parent=None, move=None,
                player=self._board.next_player(),
                moves=self._candidate_moves(),
            )

            simulations = 0
            best_move   = fallback
            try:
                while True:
                    self._check_time()
                    simulations += 1

                    node   = root
                    path   = [node]
                    pushed = 0

                    # Selection
                    while (not node.unexpanded and node.children
                           and not self._board.is_game_over()):
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

                    # Evaluation + backprop
                    self._backpropagate(path, self._eval(node.player))

                    # Update best move periodically
                    best_move, best_q, root_n = self._best_move(root, fallback)

                    for _ in range(pushed):
                        self._board.pop()

            except _TimeOut:
                pass

            best_move, best_q, root_n = self._best_move(root, best_move)
            if best_move not in legal:
                best_move = fallback if fallback in legal else legal[0]

            self._board.push(best_move)
            print(
                f"[{self.getPlayerName()}] "
                f"move={Goban.Board.flat_to_name(best_move)} "
                f"sims={simulations} rootN={root_n} q={best_q:.3f} "
                f"budget={budget:.2f}s"
            )
            return Goban.Board.flat_to_name(best_move)

        finally:
            self._used_time += time.perf_counter() - t0

    def playOpponentMove(self, move):
        self._board.push(Goban.Board.name_to_flat(move))

    def newGame(self, color):
        self._board    = Goban.Board()
        self._mycolor  = color

    def endGame(self, winner):
        if self._mycolor == winner:
            print(f"[{self.getPlayerName()}] I won.")
        else:
            print(f"[{self.getPlayerName()}] I lost.")
