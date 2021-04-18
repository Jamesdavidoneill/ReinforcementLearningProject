from initializer import *
from numpy import random


def random_player(board, o=False, **_):
    # plays tictactoe randomly, no policy.
    actions = board.actions(o)
    try:
        act = actions[random.randint(len(actions))]
    except ValueError:
        act = actions[0]
    return act


def rand_enhance(board, o=False, **_):
    actions = board.actions(o)
    prefs = [0.5] * len(actions)
    for i in range(len(actions)):
        # for sym in symmetries:
        #     acsy = multi(sym, act.mutable)
        if actions[i].win_lose() == 'lose' and o:
            prefs[i] = 1
        elif actions[i].win_lose() == 'win' and not o:
            prefs[i] = 0
        elif actions[i].win_lose() == 'draw':
            prefs[i] = 0
    maxes2 = [k for k in range(len(prefs)) if prefs[k] == max(prefs)]
    try:
        option = maxes2[random.randint(len(maxes2))]
    except ValueError:
        option = 0
    return actions[option]


def reinforce(board, e=0.1, o=True):
    #  always plays o's
    global layouts
    # look for available actions
    actions = board.actions(o)
    crit = random.rand()
    if crit < e:
        # take non greedy action => don't update the previous preference
        action = random.choice(actions)

    else:
        prefs = []
        alts = []
        # Try all the actions and all their symmetries to find their layouts score
        for i in actions:
            _, actual = check_in(i)
            prefs.append(layouts[actual])
            alts.append(actual)

        maxes2 = [k for k in range(len(prefs)) if prefs[k] == max(prefs)]

        # print(maxes2, np.random.randint(len(maxes2)))
        option = random.choice(maxes2)

        action = actions[option]

    return action

