# -*- coding: utf-8 -*-
"""Tournament player for 8x8 Go (Tromp-Taylor rules).

Main ideas:
- iterative deepening + alpha-beta negamax
- weak legal moves in search (faster), with explicit super-ko rejection via push()/pop()
- lightweight heuristic combining score proxy, liberties proxy, influence and center pressure
- move ordering + candidate reduction to keep search stable under time constraints
"""

import math
import time

import Goban
from playerInterface import *


class _TimeOut(Exception):
    pass


class myPlayer(PlayerInterface):
    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self._opponent = None

        self._ttable = {}
        self._total_budget = 45.0 * 60.0
        self._used_time = 0.0
        self._safety_margin = 2.0

        self._max_depth = 5
        self._max_candidates = 18
        self._root_move = -1
        self._deadline = 0.0

    def getPlayerName(self):
        return "Ing2 AlphaBeta 8x8"

    def _neighbors(self, flat_coord):
        i = self._board._neighborsEntries[flat_coord]
        while self._board._neighbors[i] != -1:
            yield int(self._board._neighbors[i])
            i += 1

    def _has_neighbor_stone(self, move):
        b = self._board.get_board()
        for n in self._neighbors(move):
            if b[n] != Goban.Board._EMPTY:
                return True
        return False

    def _quick_order_score(self, move, color):
        if move == -1:
            return -1000.0

        b = self._board.get_board()
        opponent = Goban.Board.flip(color)
        score = 0.0

        for n in self._neighbors(move):
            stone = b[n]
            if stone == color:
                score += 2.0
            elif stone == opponent:
                score += 3.0
            else:
                score += 0.4

        x, y = Goban.Board.unflatten(move)
        score -= 0.4 * (abs(x - 3.5) + abs(y - 3.5))
        return score

    def _ordered_moves(self, color):
        moves = self._board.weak_legal_moves()
        non_pass = [m for m in moves if m != -1]

        if self._board._nbBLACK + self._board._nbWHITE == 0:
            opening = [
                Goban.Board.flatten((3, 3)),
                Goban.Board.flatten((4, 4)),
                Goban.Board.flatten((3, 4)),
                Goban.Board.flatten((4, 3)),
                Goban.Board.flatten((2, 2)),
                Goban.Board.flatten((5, 5)),
                Goban.Board.flatten((2, 5)),
                Goban.Board.flatten((5, 2)),
            ]
            legal = set(non_pass)
            ordered = [m for m in opening if m in legal]
            ordered.extend(m for m in non_pass if m not in legal)
            ordered.append(-1)
            return ordered

        adjacent = [m for m in non_pass if self._has_neighbor_stone(m)]
        frontier = adjacent if adjacent else non_pass
        frontier.sort(key=lambda m: self._quick_order_score(m, color), reverse=True)

        if len(frontier) > self._max_candidates:
            frontier = frontier[: self._max_candidates]

        frontier.append(-1)
        return frontier

    def _evaluate(self):
        if self._board.is_game_over():
            winner = self._board.result_number()
            if winner == self._mycolor:
                return 10_000.0
            if winner == self._opponent:
                return -10_000.0
            return 0.0

        score_b, score_w = self._board.compute_score()
        if self._mycolor == Goban.Board._BLACK:
            score_term = float(score_b - score_w)
            stones_term = float(self._board._nbBLACK - self._board._nbWHITE)
            captures_term = float(self._board._capturedBLACK - self._board._capturedWHITE)
        else:
            score_term = float(score_w - score_b)
            stones_term = float(self._board._nbWHITE - self._board._nbBLACK)
            captures_term = float(self._board._capturedWHITE - self._board._capturedBLACK)

        board = self._board.get_board()
        lib_my = 0.0
        lib_opp = 0.0
        infl_my = 0.0
        infl_opp = 0.0
        center_my = 0.0
        center_opp = 0.0

        for fc in range(len(board)):
            p = board[fc]
            if p == Goban.Board._EMPTY:
                continue

            libs = 0
            influence = 0
            for n in self._neighbors(fc):
                if board[n] == Goban.Board._EMPTY:
                    libs += 1
                    influence += 1
                elif board[n] == p:
                    influence += 0.4

            x, y = Goban.Board.unflatten(fc)
            center_bonus = 3.5 - (abs(x - 3.5) + abs(y - 3.5)) * 0.5
            if p == self._mycolor:
                lib_my += libs
                infl_my += influence
                center_my += center_bonus
            else:
                lib_opp += libs
                infl_opp += influence
                center_opp += center_bonus

        liberties_term = lib_my - lib_opp
        influence_term = infl_my - infl_opp
        center_term = center_my - center_opp

        return (
            1.8 * score_term
            + 0.7 * stones_term
            + 1.0 * captures_term
            + 0.35 * liberties_term
            + 0.2 * influence_term
            + 0.2 * center_term
        )

    def _check_time(self):
        if time.perf_counter() >= self._deadline:
            raise _TimeOut()

    def _negamax(self, depth, alpha, beta):
        self._check_time()

        player_to_move = self._board.next_player()
        sign = 1.0 if player_to_move == self._mycolor else -1.0

        if depth == 0 or self._board.is_game_over():
            return sign * self._evaluate()

        key = (int(self._board._currentHash), depth, player_to_move)
        cached = self._ttable.get(key)
        if cached is not None:
            return cached

        best = -math.inf
        has_valid = False

        for move in self._ordered_moves(player_to_move):
            ok = self._board.push(move)
            if not ok:
                self._board.pop()
                continue
            has_valid = True
            try:
                value = -self._negamax(depth - 1, -beta, -alpha)
            finally:
                self._board.pop()

            if value > best:
                best = value
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break

        if not has_valid:
            best = sign * self._evaluate()

        self._ttable[key] = best
        return best

    def _search_root(self, depth):
        color = self._board.next_player()
        alpha = -math.inf
        beta = math.inf
        best_val = -math.inf
        best_move = self._root_move

        for move in self._ordered_moves(color):
            self._check_time()
            ok = self._board.push(move)
            if not ok:
                self._board.pop()
                continue

            try:
                value = -self._negamax(depth - 1, -beta, -alpha)
            finally:
                self._board.pop()

            if value > best_val:
                best_val = value
                best_move = move
            if value > alpha:
                alpha = value

        return best_move, best_val

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
            factor = 1.3
        else:
            factor = 1.8

        budget = base * factor
        return max(0.05, min(10.0, budget, 0.25 * remaining))

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

            # Good fallback if deep search gets interrupted quickly.
            self._root_move = next((m for m in legal if m != -1), legal[0])
            self._ttable.clear()

            turn_budget = self._allocate_turn_budget()
            self._deadline = time.perf_counter() + turn_budget

            best_move = self._root_move
            best_val = -math.inf
            reached_depth = 0

            for depth in range(1, self._max_depth + 1):
                try:
                    candidate, value = self._search_root(depth)
                except _TimeOut:
                    break
                if candidate is not None:
                    best_move = candidate
                    best_val = value
                    reached_depth = depth

            if best_move not in legal:
                best_move = self._root_move

            self._board.push(best_move)
            print(
                f"[{self.getPlayerName()}] move={Goban.Board.flat_to_name(best_move)} "
                f"depth={reached_depth} eval={best_val:.2f} budget={turn_budget:.2f}s"
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
        self._ttable.clear()

    def endGame(self, winner):
        if self._mycolor == winner:
            print(f"[{self.getPlayerName()}] I won.")
        else:
            print(f"[{self.getPlayerName()}] I lost.")

