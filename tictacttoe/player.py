from initializer import *
from numpy import random

counter = []


def random_player(board, o=False):
    # plays tictactoe randomly, no policy.
    actions = board.actions(o)
    try:
        act = actions[random.randint(len(actions))]
    except ValueError:
        act = actions[0]
    return act


def rand_enhance(board, o=False):
    actions = board.actions(o)
    for act in actions:
        for sym in symmetries:
            acsy = multi(sym, act.string)
            if Board(acsy).win_lose() == 'lose':
                return act
    act = actions[random.randint(len(actions))]
    return act


def reinforce(board, e=0.1):
    # if learn=True then update layouts
    global layouts
    actions = board.actions(True)
    crit = random.rand()
    counter.append(crit)
    if crit < e:
        # take non greedy action => don't update the previous preference
        action = actions[random.randint(len(actions))]
        for i in symmetries:
            if multi(i, action.string) in layouts:
                action_alt = multi(i, action.string)
    else:
        prefs = []
        alts = []
        for i in actions:
            for j in symmetries:
                if multi(j, i.string) in layouts:
                    prefs.append(layouts[multi(j, i.string)])
                    alts.append(multi(j, i.string))
                    break
        # if len(prefs) != len(actions):
        #     print('problem, len=', len(prefs), len(actions))

        maxes2 = [k for k in range(len(prefs)) if prefs[k] == max(prefs)]

        # print(maxes2, np.random.randint(len(maxes2)))
        try:
            option = maxes2[random.randint(len(maxes2))]
        except ValueError:
            option = 0

        action = actions[option]
        action_alt = alts[option]

    return action, action_alt, crit < e


def contained(board):
    # checks what permutation of a board is contained in layouts
    for i in range(len(symmetries)):
        if symmetries[i] * board in layouts:
            return i
