from player import *
from matplotlib import pyplot as plt
import csv
import curses
import time


def to_train(label, steps=600, a=0.02, e=0.2, opponent=random_player, save=False):
    """ This function finds the action-values through playing against a chosen opponent with Sarsa learning"""
    global rewards1
    global rewards2
    initializer()

    rewards1 = []
    rewards2 = []
    for i in range(steps):
        play1(a, e, reinforce, opponent, 0, True)
        play1(a, e, opponent, reinforce, 1, False)
    print(rewards1, rewards2)
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
    """For playing one agent against another without training it."""
    global rewards3
    global rewards4
    rewards3 = []
    rewards4 = []
    for i in range(steps):
        rewards3.append(rando([agent, opponent], True))
        rewards4.append(rando([agent, opponent], False))

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
        if decider(current, old_alt=0, ):
            current = ops[int(first)](current, not first)
        state1 = current.win_lose()
        if state1 in ['win', 'lose', 'draw']:
            return int(state1 == 'win')


def play1(a, e, first, second, learn, o):
    """This function manages playing but with the RL agent going first"""
    # learn=0 if first is the agent, 1 if second is the agent, 2 if neither are, 3 if both are
    # Initializing the action-value dictionary, and the empty board.
    old_alt = None
    current = Board(0)

    while True:
        # the agent takes an action, the board status is checked
        current = first(current, e=e, o=o)
        state, alto = decider(current, old_alt, a, learn=learn)
        if learn == 0:
            current_alt = alto
        if state != 3:
            return state

        # Opponent takes an action
        current = second(current, e=e, o=not o)
        state, alto = decider(current, old_alt, a, learn=learn)
        if learn == 1:
            current_alt = alto
        if state != 3:
            return state
        old_alt = current_alt


def chart_maker(steps=600):
    random_play1("Enhanced vs Enhanced", steps=steps, save=True, agent=rand_enhance, opponent=rand_enhance)
    to_train("Reinforced vs Random  Epsilon=0.2", steps=steps, save=True, e=0.2)
    to_train("Reinforced vs Random  Epsilon=0.02", steps=steps, save=True, e=0.02)
    to_train("Reinforced vs Enhanced Random  Epsilon=0.2", steps=steps, save=True, e=0.2, opponent=rand_enhance)
    to_train("Reinforced vs Enhanced Random s[old_alt] Epsilon=0.02", steps=steps, save=True, e=0.02, opponent=rand_enhance)

    with open("layouts.csv", "w", newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['boards', 'value'])
        for entry in layouts:
            if layouts[entry] not in [0, 1]:
                writer.writerow([entry, layouts[entry]])
    random_play1("Random v Random", steps=steps, save=True)
    random_play1("Random v Enhanced Random", opponent=rand_enhance, steps=steps, save=True)


def demo(stdscr):
    """ Demo mode for tictactoe. Train the agent first with to_train("label", opponent=rand_enhance, steps=3000)"""
    global layouts

    curses.curs_set(0)
    current = Board(0)
    dicto = {'win': 'LOSE', 'lose': 'WIN', 'draw': 'DRAW'}
    while True:
        current = reinforce(current, e=0)
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

            act = actions[position]
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


def decider(current, old_alt, a=0, learn=0):
    """Returns 0 for win, 1 for loss, 2 for draw, 3 for even"""
    # state=0 ==> 'win', state=1 ==> 'lose', state==2 ==>'draw', state=3 ==> 'even' = in play
    index_dict = {'win': 0, 'lose': 1, 'draw': 2, 'even': 3}
    state = index_dict[current.win_lose()]
    _, current_alt = check_in(current)
    if learn <= 1:
        # if the game has ended
        if state in [0, 1, 2]:
            # layouts[old_alt] = layouts[old_alt] + a * (int(state == 1) - layouts[old_alt])
            if learn == 0:
                layouts[old_alt] = layouts[old_alt] + a * (int(state == 0) - layouts[old_alt])
                rewards1.append(int(state == 0))
            elif learn == 1:
                layouts[old_alt] = layouts[old_alt] + a * (int(state == 0) - layouts[old_alt])
                rewards2.append(int(state == 0))
        elif state == 3 and old_alt is not None:
            layouts[old_alt] = layouts[old_alt] + a * (layouts[current_alt] - layouts[old_alt])
    return state, current_alt
