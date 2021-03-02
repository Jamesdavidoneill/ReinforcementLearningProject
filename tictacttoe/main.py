from player import *
from matplotlib import pyplot as plt
import csv
import curses
import time


def to_train(label, steps=600, a=0.2, e=0.2, opponent=random_player, save=False):
    # If rand_policy is True we just train the algo vs the random player
    # randomplayer plays with o's if o=True
    global rewards1
    global rewards2
    initializer()

    # current = Board(0)
    rewards1 = []
    rewards2 = []
    for i in range(steps):
        play1(a, e, opponent)
        play2(a, e, opponent)

    # plt.plot(range(steps), [sum(rewards2[max(0, i - 100):i]) / 100 for i in range(steps)])

    # for the one starting
    plt.plot(range(5, steps), [sum(rewards1[:i]) / (i + 1) for i in range(5, steps)], "r", label="agent starts")
    # for the not starting one
    plt.plot(range(5, steps), [sum(rewards2[:i]) / (i + 1) for i in range(5, steps)], "b", label="opponent starts")
    plt.ylim(0, 1)
    plt.ylabel("Average Reward")
    plt.xlabel("# Games")
    plt.legend(loc='lower right')
    plt.title(label)
    if save:
        plt.savefig(label + ".png")
    plt.show()


def random_play1(label, steps=600, save=False, agent=random_player, opponent=random_player):
    global rewards3
    global rewards4
    rewards3 = []
    rewards4 = []
    for i in range(steps):
        rewards3.append(rando([agent, opponent], True))
        rewards4.append(rando([agent, opponent], False))

    # print(rewards3)
    # print(rewards4)
    plt.plot(range(5, steps), [sum(rewards3[:i]) / (i + 1) for i in range(5, steps)], "r", label="agent starts")
    plt.plot(range(5, steps), [sum(rewards4[:i]) / (i + 1) for i in range(5, steps)], "b", label="opponent starts")
    plt.title(label)
    plt.ylim(0, 1)
    plt.ylabel("Average Reward")
    plt.xlabel("# Games")
    plt.legend(loc='lower right')
    plt.title(label)
    if save:
        plt.savefig(label+'.png')
    plt.show()


def rando(ops, first):
    current = Board(0)
    while True:
        current = ops[int(not first)](current, first)
        # print(current)
        state1 = current.win_lose()
        # print(current, state1)
        if state1 in ['win', 'lose', 'draw']:
            # print("APPEND 1", int(state1 == 'win'))
            return int(state1 == 'win')
        current = ops[int(first)](current, not first)
        # print(current)
        state1 = current.win_lose()
        # print(current, state1)
        if state1 in ['win', 'lose', 'draw']:
            # print("APPEND 2", int(state1 == 'win'))
            return int(state1 == 'win')


def play1(a, e, opponent):
    global layouts
    global rewards1
    old_cle = True
    old_alt = None
    current = Board(0)

    while True:
        current, current_alt, cle = reinforce(current, e=e)
        # print(current, '\n')
        state_1 = current.win_lose()
        if state_1 in ['win', 'lose', 'draw']:
            if not old_cle:
                layouts[old_alt] = layouts[old_alt] + a * (int(state_1 == 'win') - layouts[old_alt])
            rewards1.append(int(state_1 == 'win'))
            return state_1

        if state_1 == 'even' and not old_cle:
            layouts[old_alt] = layouts[old_alt] + a * (layouts[current_alt] - layouts[old_alt])
        current = opponent(current)
        state_1 = current.win_lose()

        if state_1 in ['win', 'lose', 'draw']:
            if not old_cle:
                layouts[old_alt] = layouts[old_alt] + a * (int(state_1 == 'win') - layouts[old_alt])
            rewards1.append(int(state_1 == 'win'))
            return state_1

        old_alt = current_alt
        old_cle = cle


def play2(a, e, opponent):
    global layouts
    global rewards2
    old_cle = True
    old_alt = None
    current = Board(0)

    while True:
        current = opponent(current)
        if current == Board(449):
            print("PROBLEM2\n", current)
        # print(current, '\n')
        state_1 = current.win_lose()
        if state_1 in ['win', 'lose', 'draw']:
            if not old_cle:
                layouts[old_alt] = layouts[old_alt] + a * (int(state_1 == 'win') - layouts[old_alt])
            rewards2.append(int(state_1 == 'win'))
            return state_1

        current, current_alt, cle = reinforce(current, e=e)
        # cles.append(int(cle))
        state_1 = current.win_lose()

        if state_1 in ['win', 'lose', 'draw']:
            if not old_cle:
                layouts[old_alt] = layouts[old_alt] + a * (int(state_1 == 'win') - layouts[old_alt])
            rewards2.append(int(state_1 == 'win'))
            return state_1

        if state_1 == 'even' and not old_cle:
            layouts[old_alt] = layouts[old_alt] + a * (layouts[current_alt] - layouts[old_alt])
        old_alt = current_alt
        old_cle = cle


def chart_maker(steps=600):
    random_play1("Enhanced vs Enhanced", steps=steps, save=True, agent=rand_enhance, opponent=rand_enhance)
    to_train("Reinforced vs Random  Epsilon=0.2", steps=steps, save=True, e=0.2)
    to_train("Reinforced vs Random  Epsilon=0.02", steps=steps, save=True, e=0.02)
    to_train("Reinforced vs Enhanced Random  Epsilon=0.2", steps=steps, save=True, e=0.2, opponent=rand_enhance)
    to_train("Reinforced vs Enhanced Random  Epsilon=0.02", steps=steps, save=True, e=0.02, opponent=rand_enhance)

    with open("layouts.csv", "w", newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['boards', 'value'])
        for entry in layouts:
            if layouts[entry] not in [0, 1]:
                writer.writerow([entry, layouts[entry]])
    random_play1("Random v Random", steps=steps, save=True)
    random_play1("Random v Enhanced Random", opponent=rand_enhance, steps=steps, save=True)


def demo(stdscr):
    global layouts

    curses.curs_set(0)
    current = Board(0)
    dicto = {'win': 'LOSE', 'lose': 'WIN', 'draw': 'DRAW'}
    while True:
        current, current_alt, cle = reinforce(current, e=0)
        state_1 = current.win_lose()
        if state_1 in ['win', 'lose', 'draw']:
            stdscr.clear()
            stdscr.addstr(str(current) + '\n' + dicto[state_1])
            stdscr.refresh()
            time.sleep(3)
            break

        actions = current.actions(False)
        position = 0
        # clear screen, pause for 1 second and show given board
        stdscr.clear()
        stdscr.addstr(str(current))
        stdscr.refresh()
        time.sleep(3)

        # clear screen and show first member of actions
        stdscr.clear()
        act = actions[position]
        stdscr.addstr(str(act))
        dict2 = {'0': '0', '1': '2', '2': '1'}
        opp_act_str = ''.join([dict2[k] for k in act.string])
        opp_act = Board(rebase(opp_act_str, False))
        stdscr.addstr(str("\nValue is " + str(layouts[check_in(opp_act)[1]])))
        stdscr.refresh()
        while True:
            key = stdscr.getch()

            if position < len(actions)-1 and key == curses.KEY_RIGHT:
                position += 1
            elif position > 0 and key == curses.KEY_LEFT:
                position -= 1

            act=actions[position]
            opp_act_str = ''.join([dict2[k] for k in act.string])
            opp_act = Board(rebase(opp_act_str, False))
            if key != curses.KEY_ENTER and key not in [10, 13]:
                stdscr.clear()
                stdscr.addstr(str(act))
                stdscr.addstr(str("\nValue is " + str(layouts[check_in(opp_act)[1]])))
                stdscr.refresh()
            elif key == curses.KEY_ENTER or key in [10, 13]:
                current = actions[position]
                break

        state_1 = current.win_lose()
        if state_1 in ['win', 'lose', 'draw']:
            state_2 = dicto[state_1]
            stdscr.clear()
            stdscr.addstr(str(current) + '\n' + dicto[state_1])
            stdscr.refresh()
            time.sleep(3)
            break
