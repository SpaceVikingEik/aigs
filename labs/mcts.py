# imports
from __future__ import annotations

import math

import numpy as np
import aigs
from aigs import State, Env
from dataclasses import dataclass, field


# %% Setup
env: Env


# %%
def minimax(state: State, maxim: bool) -> int:
    if state.ended:
        return state.point
    else:
        temp: int = -10 if maxim else 10
        for action in np.where(state.legal)[0]:  # for all legal actions
            value = minimax(env.step(state, action), not maxim)
            temp = max(temp, value) if maxim else min(temp, value)
        return temp


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int, depth: int) -> int:
    depth += 1;
    if state.ended or depth == 10:
        return -state.point if maxim else state.point
    else:
        if maxim:
            value = -math.inf
            for action in state.legal:  # for all legal actions
                if depth == 10:
                    value = alpha
                    break
                value = max(value, alpha_beta(env.step(state, int(action)), not maxim, alpha, beta, depth))
                if value >= beta:
                    break
                alpha = max(alpha, value)
            return value
        elif not maxim:
            value = math.inf
            for action in state.legal:  # for all legal actions
                value = min(value, alpha_beta(env.step(state, int(action)), not maxim, alpha, beta, depth))
                if value <= alpha:
                    break
                beta = min(beta, value)
        return value


@dataclass
class Node:
    state: State  # Add more fields


# Intuitive but difficult in terms of code
def monte_carlo(state: State, cfg) -> int:
    raise NotImplementedError  # you do this


def tree_policy(node: Node, cfg) -> Node:
    raise NotImplementedError  # you do this


def expand(v: Node) -> Node:
    raise NotImplementedError  # you do this


def best_child(root: Node, c) -> Node:
    raise NotImplementedError  # you do this


def default_policy(state: State) -> int:
    raise NotImplementedError  # you do this


def backup(node, delta) -> None:
    raise NotImplementedError  # you do this


# Main function
def main(cfg) -> None:
    global env
    env = aigs.make(cfg.game)
    state = env.init()

    while not state.ended:
        actions = state.legal  # the actions to choose from

        match getattr(cfg, state.player):
            case "random":
                a = np.random.choice(actions).item()

            case "human":
                print(state, end="\n\n")
                a = int(input(f"Place your piece ({'x' if state.minim else 'o'}): "))

            case "minimax":
                values = [minimax(env.step(state, a), not state.maxim) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case "alpha_beta":
                values = [alpha_beta(env.step(state, int(a)), not state.maxim, -math.inf, math.inf, 0) for a in actions]
                a = int(actions[np.argmax(values) if state.maxim else np.argmin(values)])

            case "monte_carlo":
                raise NotImplementedError

            case _:
                raise ValueError(f"Unknown player {state.player}")

        state = env.step(state, a)

    print(f"{['nobody', 'o', 'x'][1 if state.maxim else 2]} won", state, sep="\n")
