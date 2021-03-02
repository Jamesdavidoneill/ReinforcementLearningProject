from initializer import *
from numpy import random


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


def reinforce(board, e=0.1):
    #  always plays o's
    global layouts
    # look for available actions
    actions = board.actions(True)
    crit = random.rand()
    if crit < e:
        # take non greedy action => don't update the previous preference
        action = actions[random.randint(len(actions))]
        in_bool, action_alt = check_in(action)
        if not in_bool:
            print("PROBLEM 1")

    else:
        prefs = []
        alts = []
        # Try all the actions and all their symmetries to find their layouts score
        for i in actions:
            in_bool, actual = check_in(i)
            if not in_bool:
                print("PROBLEM 2")
            prefs.append(layouts[actual])
            alts.append(actual)

        maxes2 = [k for k in range(len(prefs)) if prefs[k] == max(prefs)]

        # print(maxes2, np.random.randint(len(maxes2)))
        try:
            option = maxes2[random.randint(len(maxes2))]
        except ValueError:
            option = 0

        action = actions[option]
        action_alt = alts[option]

    return action, action_alt, crit < e


def asker(current):
    # Where can I put my x's
    actions = current.actions(False)
    for i in range(len(actions)):
        print(i, str(layouts[check_in(actions[i])[1]]), '\n' + str(actions[i]))
    board = actions[int(input("Enter your choice: "))]

    return board
