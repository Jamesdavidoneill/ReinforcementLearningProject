from player import *
from matplotlib import pyplot as plt


def to_train(label, steps=600, a=0.01, e=0.1, opponent=random_player, save=False):
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


def random_play1(label, steps=600, agent=random_player, opponent=random_player):
    global rewards3
    global rewards4
    rewards3 = []
    rewards4 = []
    for i in range(steps):
        rewards3.append(rando([agent, opponent], True))
        rewards3.append(rando([agent, opponent], False))

    plt.plot(range(5, steps), [sum(rewards3[:i]) / (i + 1) for i in range(5, steps)], "r", label="agent starts")
    plt.plot(range(5, steps), [sum(rewards4[:i]) / (i + 1) for i in range(5, steps)], "b", label="opponent starts")
    plt.title(label)
    plt.legend('lower right')
    plt.ylabel("Average Reward")
    plt.xlabel("# Games")
    plt.ylim(0, 1)
    plt.savefig(label+'.png')
    plt.show()


def rando(ops, first):
    current = Board(0)
    while True:
        current = ops[int(not first)](current, first)
        state1 = current.win_lose()
        if state1 in ['win', 'lose', 'draw']:
            return int(state1 == 'win')
        current = ops[int(first)](current, first)
        state1 = current.win_lose()
        if state1 in ['win', 'lose', 'draw']:
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


to_train("Reinforced vs Random")
