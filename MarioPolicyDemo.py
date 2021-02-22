from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#env = gym_super_mario_bros.make('SuperMarioBros-v0')

#env = JoypadSpace(env, SIMPLE_MOVEMENT)

#done = True
#for step in range(100):
#    if done:
#        state = env.reset()
#    state, reward, done, info = env.step(env.action_space.sample())
#    env.render()

#env.close()


###########
#Set up display
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

#Deep Q Network
class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

#experience class
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

e = Experience(2, 3, 1, 4)

#Replay memory
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

#epsilon greedy strategy
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1 * current_step * self.decay)

#reinforcement learning agent
class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return  torch.tensor([action]).to(self.device) #explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device) #exploit


#environment manager
class CartPoleEnvManager():
    def __init__(self, device):
        self.device = device
        tempenv = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(tempenv, SIMPLE_MOVEMENT)
        self.env.reset()
        self.current_screen = None
        self.done = False
        self.info = None
        #print(self.env)

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        #print("num actions is ", self.env.action_space.n)
        return self.env.action_space.n

    def take_action(self, action):
        _, reward, self.done, self.info=self.env.step(action.item())
        if int(self.info['life']) < 2:
            #Only give mario 1 life
            self.done = True
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1)) #pytorch
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        #strip off top and bottom
        top = int(screen_height * 0.25)
        bottom = int(screen_height * 1.0)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        #convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32)/255
        screen = torch.from_numpy(screen)

        #Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            ,T.Resize((40, 90))
            ,T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device) #add a batch dimension

#Examples of non-processed screen
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#em = CartPoleEnvManager(device)
#em.reset()
#screen = em.render('rgb_array')

#plt.figure()
#plt.imshow(screen)
#plt.title('Non-processed screen example')
#plt.show()

#Example of processed screen
#screen = em.get_processed_screen()

#plt.figure()
#plt.imshow(screen.squeeze(0).permute(1, 2, 0), interpolation='none')
#plt.title('Processed screen example')
#plt.show()

#Example of non starting state

#for i in range(10):
#    em.take_action(torch.tensor([1]))
#screen = em.get_state()

#plt.figure()
#plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
#plt.title('Processed screen example')
#plt.show()


#Utility functions
def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training ...')
    plt.xlabel('Episode')
    plt.ylabel('duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

#Tensor processing
def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4)

#Q-Value Calculator
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        #print("index is ", actions.unsqueeze(-1))
        #changed dim 1 to dim 0
        return policy_net(states).gather(dim=0, index=actions.unsqueeze(-1))
    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations==False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

#Main program

#parameters
batch_size = 256
gamma = 0.999
eps_start =0
eps_end = 0.00
eps_decay = 0.000
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)


#load policy network
PATH = "mario_policy_net.pt"

policy_net = DQN(em.get_screen_height(), em.get_screen_width())
policy_net.load_state_dict(torch.load(PATH))
policy_net.eval()

#save test
#torch.save(policy_net.state_dict(), PATH)

for episode in range(3):
    em.reset()
    state = em.get_state()
    print("this is a pass", episode)
    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        state = next_state

        if em.done:
           print("episode complete")
           break
        em.render('human')
em.close()