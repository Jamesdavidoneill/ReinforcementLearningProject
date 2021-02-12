"""
So gonna do a stationary k-armed bandit first then a non-stationary
"""

import numpy as np
from matplotlib import pyplot as plt


def to_exec(steps, e, sigma=0):
    global means
    global est_reward
    global selections

    means = np.random.normal(0, 1, 10)
    est_reward = np.array([0.0]*10)
    selections = np.array([0]*10)
    rewards_avg = [0]
    for i in range(steps):
        means += np.random.normal(0, sigma, 10)
        avg = rewards_avg[-1]
        reward = select(e)
        rewards_avg.append(avg + 1/(i+1)*(reward-avg))

    return rewards_avg[1:]


def select(e):
    # take greedy option with p=1-e or non-greedy with p=e
    p = np.random.uniform(0, 1)
    if p <= e:
        choice = np.random.randint(10)

    else:
        m = max(est_reward)
        greedy = [i for i in range(len(est_reward)) if est_reward[i] == m]
        choice = greedy[np.random.randint(len(greedy))]

    reward = np.random.normal(means[choice], 1)
    est_reward[choice] = est_reward[choice] + 1 / (selections[choice]+1) * (reward - est_reward[choice])
    selections[choice] += 1
    return reward


plt.plot(range(2000), to_exec(2000, 0, 0.01), 'g-', label='e=0')
plt.plot(range(2000), to_exec(2000, 0.01, 0.01), 'r-', label='e=0.01')
plt.plot(range(2000), to_exec(2000, 0.1, 0.01), 'b-', label='e=0.1')
plt.legend(loc="lower right")
plt.show()