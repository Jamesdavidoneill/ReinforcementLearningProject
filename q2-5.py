"""
This is a simple non-stationary 10-armed bandit example.
That is we have 10 actions each of which results in a reward being given from a normal distribution with
variance 1 and pre-assigned means. The program will basically try to figure out how to get the highest
reward only by estimating the mean from samples.
At each time-step it will either choose either to exploit or to explore (probability e).
The true means will undergo a random walk with variance sigma
"""

import numpy as np
from matplotlib import pyplot as plt


def to_exec(steps, e, sigma=0.0):
    # making variables global so they can be edited by select function
    global means
    global est_reward
    global selections

    means = np.random.normal(0, 1, 10)      # true means
    est_reward = np.array([0.0]*10)         # estimated means
    selections = np.array([0]*10)           # how many times an index has been selected
    rewards_avg = [0]                       # avg reward to be plotted
    for i in range(steps):
        means += np.random.normal(0, sigma, 10)         # random walk
        avg = rewards_avg[-1]
        reward = select(e)
        rewards_avg.append(avg + 1/(i+1)*(reward-avg))

    return rewards_avg[1:]


def select(e):
    # exploit with prob=1-e or explore with p=e
    p = np.random.uniform(0, 1)
    if p <= e:
        choice = np.random.randint(10)

    else:
        m = max(est_reward)
        greedy = [i for i in range(len(est_reward)) if est_reward[i] == m]
        choice = greedy[np.random.randint(len(greedy))]

    reward = np.random.normal(means[choice], 1)     # reward for the chosen action
    # updating the estimated mean reward without re-adding all guesses
    est_reward[choice] = est_reward[choice] + 1 / (selections[choice]+1) * (reward - est_reward[choice])
    selections[choice] += 1
    return reward


# plotting the average reward when operating with different probs of exploring e.
plt.plot(range(2000), to_exec(2000, 0, 0.01), 'g-', label='e=0')
plt.plot(range(2000), to_exec(2000, 0.01, 0.01), 'r-', label='e=0.01')
plt.plot(range(2000), to_exec(2000, 0.1, 0.01), 'b-', label='e=0.1')
plt.legend(loc="lower right")
plt.show()
