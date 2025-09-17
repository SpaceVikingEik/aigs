# games.py
#   aigs games
# by: Noah Syrkis
import sys

from optax.tree import where

# imports
from aigs.types import State, Env
import numpy as np


# connect four
class ConnectFour(Env):
    def init(self) -> State:
        board = np.zeros((6, 7), dtype=int)
        legal = board[0] == 0
        sys.setrecursionlimit(10000)
        state = State(board=board, legal=legal)
        return state

    def checkBoardRecursive(self, i, j, board, direction, counter) -> int:

        if counter == 4:
            return 4

        if i < 0 or j < 0 or i > 5 or j > 6:
            return 0

        if board[i, j] == 1:
            counter += 1
            return ConnectFour.checkBoardRecursive(self, i + direction[0], j + direction[1], board, direction, counter)
        return 0

    def assertWin(self, pos, board) -> bool:

        counterRL = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (0, -1), counter=0)
        counterRR = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (0, 1), counter=0)
        counterCD = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (1, 0), counter=0)
        counterCU = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (-1, 0), counter=0)
        counterDLU = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (-1, -1), counter=0)
        counterDLD = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (1, -1), counter=0)
        counterDRU = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (-1, 1), counter=0)
        counterDRD = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (1, 1), counter=0)

        if (counterRL == 4 or
                counterRR == 4 or
                counterCD == 4 or
                counterCU == 4 or
                counterDLU == 4 or
                counterDLD == 4 or
                counterDRU == 4 or
                counterDRU == 4 or
                counterDRD == 4):
            return True
        return False

    def assertValue(self, pos, board) -> int:

        counterRL = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (0, -1), counter=0)
        counterRR = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (0, 1), counter=0)
        counterCD = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (1, 0), counter=0)
        counterCU = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (-1, 0), counter=0)
        counterDLU = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (-1, -1), counter=0)
        counterDLD = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (1, -1), counter=0)
        counterDRU = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (-1, 1), counter=0)
        counterDRD = ConnectFour.checkBoardRecursive(self, pos[0], pos[1], board, (1, 1), counter=0)

        return counterRL*counterRL + counterRR*counterRR + counterCD*counterCD + counterCU*counterCU + counterDLU*counterDLU + counterDLD*counterDLD + counterDRU*counterDRU + counterDRD*counterDRD

    def step(self, state, action) -> State:
        # hint: use x.diagonal(i)
        board = state.board.copy()
        action # always between 0,6
        assert board[0,action] == 0, f"Invalid action {action}"

        pos = (0,0)
        #Perform action on board
        for i in range (5, -1, -1):
            if board[i, action] == 0:
                board[i, action] = 1
                pos = (i, action)
                break

        winner = ConnectFour.assertWin(self, pos, board)
        point = ConnectFour.assertValue(self, pos, board)
        legal = np.array([])
        for i in range (0, 7):
            if board[0, i] == 0:
                legal = np.append(legal, i)
        return State(
            board=-board,
            legal=legal,
            ended=not legal.any() or winner,
            point=point if not winner else winner,
            maxim = not state.maxim,
        )



# tic tac toe
class TicTacToe(Env):
    def init(self) -> State:
        board = np.zeros((3, 3), dtype=int)
        legal = board.flatten() == 0
        state = State(board=board, legal=legal)
        return state

    def step(self, state, action) -> State:
        # make your move
        board = state.board.copy()
        assert board[action // 3, action % 3] == 0, f"Invalid move: {action}"
        board[action // 3, action % 3] = 1 if state.maxim else -1

        # was it a winning move?
        mask = board == (1 if state.maxim else -1)
        winner: bool = (
            mask.all(axis=1).any()  # |
            or mask.all(axis=0).any()  # —
            or mask.trace() == 3  # \
            or np.fliplr(mask).trace() == 3  # /
        )

        # return the next state
        return State(
            board=board,
            legal=board.flatten() == 0,  # empty board positions
            ended=(board != 0).all() | winner,
            point=(1 if state.maxim else -1) if winner else 0,
            maxim=not state.maxim,
        )
