import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout == nn.Dropout(p=0.6)
        self.affine2 == nn.Linear(128, 2)

    
    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim= 1)
        

import argparse
import gymnasium as gym
import torch.distributed.rpc as rpc

parser = argparse.ArgumentParser(
    description="Rpc reinforcement learning example",
    formatter_class = argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--world_size', default = 2,type=int, metavar="W", help="number of workers")

parser.add_argument("--log_interval", type = int, default = 10, metavar ="N", help='number of workers')

parser.add_argument("--gamma", type=float, default= 0.9, metavar="G", help="how much to value future networks")

arg = parser.parse_args()

class Observer:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = gym.make("CartPole-v1")
        self.env_seed(args.seed)


    def run_episode(self, agent_rref):
        state, ep_reward = self.env.reset(), 0
        for _ in range(10000):
            action = agent_rref.rpc_sync().select_action(self.id, state)

            state, reward, done, _ = self.env.step(action)

            agent_rref.rpc_sync().report_reward(self.id, reward)

            if done:
                break

            
import gymnasium as gym
import numpy as np

import torch
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote
from torch.distributions import Categorical

class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = []
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr = 1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = gym.make("CartPole-v1").spec_reward_threshold
        for ob_rank in range(1, word_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []


    def select_action(self, ob_id, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_probs(action))
        return action.item()



    def report_reward(self, ob_id, reward):
        self.rewards[ob_id].append(reward)


    def run_episode(self):
        futs = []
        for ob_rref in self.ob_rrefs:
            futs.append(rpc_async, ob_rref.owner(), ob_rref.rpc_sync().run_episode, args=(self.agent_rref,))

        for fut in futs:
            fut.wait()


        