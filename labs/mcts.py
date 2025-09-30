# imports
from __future__ import annotations

import math
import random

import numpy as np
import aigs
from aigs import State, Env
from dataclasses import dataclass, field


# %% Setup
env: Env


def heuristic_value(board: np.ndarray) -> int:
    return np.sum(board)


# %%
def minimax(state: State, maxim: bool, depth: int) -> int:
    if state.ended or depth > 10:
        return state.point if state.ended else heuristic_value(state.board)
    else:
        temp: int = -10 if maxim else 10
        for action in np.where(state.legal)[0]:  # for all legal actions
            value = minimax(env.step(state, action), not maxim, depth + 1)
            temp = max(temp, value) if maxim else min(temp, value)
        return temp


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int, depth: int) -> int:
    depth += 1;
    if state.ended or depth == 5:
        return -state.point if maxim else state.point
    else:
        if maxim:
            value = -math.inf
            for action in state.legal:  # for all legal actions
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
    state: State
    actions: [] # preamble to create children
    parent: Node
    children: [] # nodes
    value: int
    visitCount: int = 0
    actionTaken: int = 0
    # Add more fields


# Intuitive but difficult in terms of code
def monte_carlo(state: State) -> int:
    #Nodes do not get stored, the entire structure is recreated on every action
    v0 = Node(state, state.legal, None, [], 0, 0, 0)
    monte_carlo_why(v0, 0)
    return int(best_child(v0).actionTaken) # you do this

def monte_carlo_why(node, depth):
    v0 = node
    if depth < 4 and v0 is not None:
        #print(depth)
        v1 = tree_policy(node, depth + 1)
        delta = default_policy(v1.state)
        backup(v1, delta)
        monte_carlo_why(best_child(v0), depth + 1)


def tree_policy(node: Node, depth: int) -> Node:

    while not node.state.ended:
        if len(node.actions) > 0:
            test = expand(node)
            #The below line does not follow the standard algorithm, but in this case it should ensure that we simulate the MCTS tree down to a depth of 4
            monte_carlo_why(test, depth)
        else:
            node = best_child(node)
    return node



def expand(v: Node) -> Node:
   action, v.actions = int(v.actions[-1]), v.actions[:-1]
   newState = env.step(v.state, action)
   child = Node(newState, newState.legal, v, [], newState.point, 0, action)
   v.children.append(child)
   return child
def best_child(root: Node) -> Node:
    bestChild = None
    for child in root.children:
        if (bestChild is None) or (bestChild.value < child.value):
            bestChild = child
    return bestChild

def default_policy(state: State) -> int:
    while not state.ended:
        random.seed()
        rand = random.randint(0, len(state.legal)-1)
        state = env.step(state, int(state.legal[rand]))
    return state.point

def backup(node, delta) -> None:
    while node is not None:
        node.visitCount += 1
        node.value = node.value + delta
        delta = -delta
        node = node.parent

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
                a = monte_carlo(state)
            case _:
                raise ValueError(f"Unknown player {state.player}")

        state = env.step(state, a)

    print(f"{['nobody', 'o', 'x'][1 if state.maxim else 2]} won", state, sep="\n")
