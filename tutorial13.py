import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


if gym.__version__ < '0.26':
    env = gym.make('CartPole-v0').unwrapped
else:
    env = gym.make('CartPole-v0', render_mode='rgb_array').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#replay memory
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """ save a transition """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#Q network

class DQN(nn.Module):
    
    def __init__(self, h, w, output):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride= 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32,kernel_size= 3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # number of linear input conections depends on output of conv2d layers
        # and therefore the input image size, so compute it

        def conv2d_size_out(size, kernel_size= 5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)


    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.bn3(self.conv3(x))))
        return self.head(x.view(x.size(0), -1))



# input extraction

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

print(resize)


# get cart location

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)



