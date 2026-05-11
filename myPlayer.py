# -*- coding: utf-8 -*-
"""Tournament player for 8x8 Go (Tromp-Taylor rules).

Search:
- Iterative deepening + PVS (Principal Variation Search / NegaScout)
- Transposition table with EXACT / LOWER / UPPER bounds
- Aspiration windows (narrows alpha-beta window using previous iteration result)
- Killer moves (2 per depth) + history heuristic for move ordering
- Adaptive max depth: deeper search when few empty cells remain

Evaluation:
- Group-level liberty analysis via union-find (_stringLiberties / _stringSizes)
- Atari detection: heavy penalty/bonus for groups with 1 or 2 liberties
- Eye counting: structural safety bonus
- Territory score (compute_score), captures, influence, centre
"""

import math
import time

import Goban
from playerInterface import *

_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2


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

        self._max_depth = 6
        self._max_candidates = 18
        self._root_move = -1
        self._deadline = 0.0

        # killer_moves[depth] = [best_killer, second_killer]
        self._killer_moves = {}
        # history[move] = accumulated cutoff score
        self._history = {}

    def getPlayerName(self):
        return "Ing2 PVS 8x8"

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def _quick_order_score(self, move, color):
        """Heuristic score for a candidate move (higher = explore first)."""
        if move == -1:
            return -1000.0

        b = self._board.get_board()
        opponent = Goban.Board.flip(color)
        score = 0.0

        for n in self._neighbors(move):
            stone = b[n]
            if stone == Goban.Board._EMPTY:
                score += 0.4
            else:
                root = self._board._getStringOfStone(n)
                libs = self._board._stringLiberties[root]
                if stone == opponent:
                    if libs == 1:
                        score += 12.0   # capture opponent group
                    elif libs == 2:
                        score += 6.0    # put opponent in atari
                    else:
                        score += 3.0
                else:  # same color
                    if libs == 1:
                        score += 8.0    # save own group from capture
                    elif libs == 2:
                        score += 2.5    # shore up nearly-threatened group
                    else:
                        score += 2.0

        x, y = Goban.Board.unflatten(move)
        score -= 0.4 * (abs(x - 3.5) + abs(y - 3.5))
        return score

    def _ordered_moves(self, color, depth=0):
        moves = self._board.weak_legal_moves()
        non_pass = [m for m in moves if m != -1]

        # Opening: prefer strong corner / near-centre points
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

        # Killer moves bump to front
        killers = self._killer_moves.get(depth, [None, None])
        non_pass_set = set(non_pass)
        killer_set = {k for k in killers if k is not None and k in non_pass_set}

        adjacent = [m for m in non_pass if self._has_neighbor_stone(m)]
        frontier = adjacent if adjacent else non_pass

        hist = self._history
        frontier.sort(
            key=lambda m: (
                (100.0 if m in killer_set else 0.0)
                + self._quick_order_score(m, color)
                + 0.01 * hist.get(m, 0)
            ),
            reverse=True,
        )

        if len(frontier) > self._max_candidates:
            frontier = frontier[: self._max_candidates]

        frontier.append(-1)
        return frontier

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self):
        """
        Sophisticated position evaluation.

        Combines:
        - Territory score (Chinese area scoring via compute_score)
        - Captures
        - Group-level liberty analysis (uses _stringLiberties / _stringSizes)
        - Atari: heavy weight on groups with 1 liberty (in danger)
        - Near-atari: moderate weight on groups with 2 liberties
        - Eye count (structural safety)
        - Influence and centre control
        """
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
            captures_term = float(self._board._capturedBLACK - self._board._capturedWHITE)
        else:
            score_term = float(score_w - score_b)
            captures_term = float(self._board._capturedWHITE - self._board._capturedBLACK)

        board = self._board.get_board()

        # --- Group-level analysis (one pass per group via union-find) ---
        analyzed_groups = set()
        lib_my = 0.0
        lib_opp = 0.0
        atari_my = 0.0    # weighted size of MY groups in atari/near-atari
        atari_opp = 0.0   # weighted size of OPPONENT groups in atari/near-atari

        # --- Stone-level analysis ---
        infl_my = 0.0
        infl_opp = 0.0
        center_my = 0.0
        center_opp = 0.0

        for fc in range(len(board)):
            p = board[fc]
            if p == Goban.Board._EMPTY:
                continue

            # Stone-level: influence and centre
            influence = 0.0
            for n in self._neighbors(fc):
                if board[n] == Goban.Board._EMPTY:
                    influence += 1.0
                elif board[n] == p:
                    influence += 0.4

            x, y = Goban.Board.unflatten(fc)
            center_bonus = 3.5 - (abs(x - 3.5) + abs(y - 3.5)) * 0.5

            if p == self._mycolor:
                infl_my += influence
                center_my += center_bonus
            else:
                infl_opp += influence
                center_opp += center_bonus

            # Group-level: only process each group root once
            root = self._board._getStringOfStone(fc)
            if root in analyzed_groups:
                continue
            analyzed_groups.add(root)

            libs = self._board._stringLiberties[root]
            size = self._board._stringSizes[root]

            if p == self._mycolor:
                lib_my += libs
                if libs == 1:
                    atari_my += size          # in atari: full penalty
                elif libs == 2:
                    atari_my += size * 0.35   # near-atari: soft penalty
            else:
                lib_opp += libs
                if libs == 1:
                    atari_opp += size         # opponent in atari: full reward
                elif libs == 2:
                    atari_opp += size * 0.35  # opponent near-atari: soft reward

        # --- Eye counting ---
        eyes_my = 0.0
        eyes_opp = 0.0
        for fc in range(len(board)):
            if board[fc] == Goban.Board._EMPTY:
                if self._board._is_an_eye(fc, self._mycolor):
                    eyes_my += 1.0
                elif self._board._is_an_eye(fc, self._opponent):
                    eyes_opp += 1.0

        liberties_term = lib_my - lib_opp
        atari_term = atari_opp - atari_my   # positive = good for us
        influence_term = infl_my - infl_opp
        center_term = center_my - center_opp
        eye_term = eyes_my - eyes_opp

        return (
            2.0  * score_term
            + 1.2  * captures_term
            + 0.35 * liberties_term
            + 3.5  * atari_term
            + 0.2  * influence_term
            + 0.15 * center_term
            + 1.5  * eye_term
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _check_time(self):
        if time.perf_counter() >= self._deadline:
            raise _TimeOut()

    def _negamax(self, depth, alpha, beta):
        """PVS (Principal Variation Search) with transposition table."""
        self._check_time()

        player_to_move = self._board.next_player()
        sign = 1.0 if player_to_move == self._mycolor else -1.0

        if depth == 0 or self._board.is_game_over():
            return sign * self._evaluate()

        # --- Transposition table lookup ---
        key = (int(self._board._currentHash), depth, player_to_move)
        cached = self._ttable.get(key)
        if cached is not None:
            tt_val, tt_flag = cached
            if tt_flag == _TT_EXACT:
                return tt_val
            elif tt_flag == _TT_LOWER:
                alpha = max(alpha, tt_val)
            else:  # _TT_UPPER
                beta = min(beta, tt_val)
            if alpha >= beta:
                return tt_val

        orig_alpha = alpha
        best = -math.inf
        has_valid = False
        first_move = True

        for move in self._ordered_moves(player_to_move, depth):
            ok = self._board.push(move)
            if not ok:
                self._board.pop()
                continue
            has_valid = True

            try:
                if first_move:
                    # Full window on first (expected best) move
                    value = -self._negamax(depth - 1, -beta, -alpha)
                    first_move = False
                else:
                    # PVS: null-window search, re-search only if it might be better
                    value = -self._negamax(depth - 1, -alpha - 1, -alpha)
                    if alpha < value < beta:
                        value = -self._negamax(depth - 1, -beta, -alpha)
            finally:
                self._board.pop()

            if value > best:
                best = value
            if value > alpha:
                alpha = value
            if alpha >= beta:
                # Beta cutoff — update killers and history
                if move != -1:
                    killers = self._killer_moves.setdefault(depth, [None, None])
                    if killers[0] != move:
                        killers[1] = killers[0]
                        killers[0] = move
                    self._history[move] = self._history.get(move, 0) + depth * depth
                break

        if not has_valid:
            best = sign * self._evaluate()

        # --- Store in transposition table ---
        if best <= orig_alpha:
            tt_flag = _TT_UPPER
        elif best >= beta:
            tt_flag = _TT_LOWER
        else:
            tt_flag = _TT_EXACT
        self._ttable[key] = (best, tt_flag)

        return best

    def _search_root(self, depth, alpha=-math.inf, beta=math.inf):
        """Root search with PVS. Returns (best_move, best_value)."""
        color = self._board.next_player()
        best_val = -math.inf
        best_move = self._root_move
        first_move = True

        for move in self._ordered_moves(color, depth):
            self._check_time()
            ok = self._board.push(move)
            if not ok:
                self._board.pop()
                continue

            try:
                if first_move:
                    value = -self._negamax(depth - 1, -beta, -alpha)
                    first_move = False
                else:
                    value = -self._negamax(depth - 1, -alpha - 1, -alpha)
                    if alpha < value < beta:
                        value = -self._negamax(depth - 1, -beta, -alpha)
            finally:
                self._board.pop()

            if value > best_val:
                best_val = value
                best_move = move
            if value > alpha:
                alpha = value

        return best_move, best_val

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
            factor = 1.3
        else:
            factor = 1.8

        budget = base * factor
        return max(0.05, min(10.0, budget, 0.25 * remaining))

    # ------------------------------------------------------------------
    # PlayerInterface implementation
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

            # Default fallback in case search is interrupted immediately
            self._root_move = next((m for m in legal if m != -1), legal[0])

            self._ttable.clear()
            self._killer_moves.clear()
            self._history.clear()

            turn_budget = self._allocate_turn_budget()
            self._deadline = time.perf_counter() + turn_budget

            # Deeper endgame search when branching factor is small
            empties = len(self._board._empties)
            if empties <= 8:
                max_d = 12
            elif empties <= 15:
                max_d = 9
            else:
                max_d = self._max_depth

            best_move = self._root_move
            best_val = -math.inf
            reached_depth = 0

            for depth in range(1, max_d + 1):
                try:
                    if reached_depth >= 2 and abs(best_val) < 9_000.0:
                        # Aspiration window: narrow search around previous result
                        window = 1.5
                        lo, hi = best_val - window, best_val + window
                        candidate, value = self._search_root(depth, lo, hi)
                        if value <= lo or value >= hi:
                            # Window failed — full re-search at this depth
                            candidate, value = self._search_root(depth)
                    else:
                        candidate, value = self._search_root(depth)
                except _TimeOut:
                    break

                if candidate is not None:
                    best_move = candidate
                    best_val = value
                    reached_depth = depth

                # Stop early if we found a forced win/loss
                if abs(best_val) >= 9_000.0:
                    break

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
        self._killer_moves.clear()
        self._history.clear()

    def endGame(self, winner):
        if self._mycolor == winner:
            print(f"[{self.getPlayerName()}] I won.")
        else:
            print(f"[{self.getPlayerName()}] I lost.")
