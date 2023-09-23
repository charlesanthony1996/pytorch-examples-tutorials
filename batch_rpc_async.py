import threading
import torchvision
import torch
import torch.distributed.rpc as rpc
from torch import optim
# import gymansium as gym


num_classes, batch_update_size = 30, 5

class BatchUpdateParameterServer(object):
    def __init__(self, batch_update_size= batch_update_size):
        self.model = torchvision.models.resnet50(num_classes = num_classes)
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr= 0.001, momentum = 0.9 )
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def get_model(self):
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        self = ps_rref.local_value()
        with self.lock:
            self.curr_update_size += 1

            for p, g in zip(self.model.parameters(), grads):
                p.grad += g


            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                # update the model
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size

                self.curr_update_size
                self.optimizer.step()
                self.optimizer.zero_grad()

                fut.set_result(self.model)
                self.future_model = torch.futures.Future()


        return fut



batch_size, image_w, image_h = 20, 64, 64


class Trainer(object):
    def __init__(self, ps_rref):
        self.ps_rref, self.loss_fn = ps_rref, torch.nn.MSELoss()
        self.one_hot_indices = torch.LongTensor(batch_size).random_(0,num_classes).view(batch_size_, 1)
    

    def get_next_batch(self):
        for _ in range(6):
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes).scatter_(1, self.one_hot_indices, 1)

            yield inputs.cuda(), labels.cuda()


    def train(self):
        name = rpc.get_worker_info().name
        # get initial model parameters
        m = self.ps_rref.rpc_sync().get_model().cuda()
        # start training
        for inputs, labels in self.get_next_batch():
            m = rpc.rpc_sync(self.ps_rref.owner(), 
            BatchUpdateParameterServer.update_and_fetch_model,
            args=(self.ps_rref,[p.grad for p in m.cpu().parameters()])).cuda()




import argparse
import os
import torch
import  torch.nn as nn
import torch.nn.functional as f

parser = argparse.ArgumentParser(description ="Pytorch rpc batch rl example")
parser.add_argument("--gamma", type=float, default=1.0, metavar= "G", help="discount factor (default 1.0)")
parser.add_argument("--seed", type=int, default= 543, metavar="E", help="number of epsisodes (default: 10)")
parser.add_argument("--num-episode", type=int, default= 10, metavar="E", help="number of episodes (default: 10)")

args = parser.parse_args()


torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self, batch=True):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p = 0.6)
        self.affine2 = nn.Linear(128, 2)
        self.dim = 2 if batch else 1


    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim = self.dim)


import gymnasium as gym
import torch.distributed.rpc as rpc

class Observer:
    def __init__(self, batch=True):
        self.id = rpc.get_worker_info().id - 1
        self.env = gym.make("CartPole-v1")
        self.env.seed(args.seed)
        self.select_action = Agent.select_action_batch if batch else Agent.select_action


    def run_episode(self, agent_rref, n_steps):
        state, ep_ward = self.env.reset(), NUM_STEPS
        rewards = torch.zeros(n_steps)
        start_step = 0
        for step in range(n_steps):
            state = torch.from_numpy(state).float().unsqueeze(0)
            # send the state to the agent to get an action
            action = rpc.rpc_async(
                agent_rref.owner(),
                self.select_action,
                args=(agent_rref, self.id, state)
            )

            # apply the action to the environment , and get the reward
            state, reward , done , _ = self.env.step(action)
            rewards[step] = reward

            if done or step + 1 >= n_steps:
                curr_rewards = rewards[start_step:(step + 1)]
                R = 0
                for i in range(curr_rewards.numel() - 1, -1, -1):
                    R = curr_rewards[i] + args.gamma * R
                    curr_rewards[i] = R
                state = self.env.reset()
                if start_step == 0:
                    ep_reward = min(ep_reward, step - start_step + 1)
                start_step = step + 1

            
        return [rewards, reward]

import threading
from torch.distributed.rpc import RRef

class Agent:
    def __init__(self, world_size, batch=True):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.policy = Policy(batch).cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.running_reward = 0

        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format())
            self.ob_rrefs.append(rpc.remote(ob_info, Observer, args=(batch,)))
            self.rewards[ob_info.id] = []


        self.states = torch.zeros(len(self.ob_rrefs), 1, 4)
        self.batch = batch
        self.saved.log_probs = [] if batch else {k:[] for k in range(len(self.ob_rrefs))}
        self.future_actions = torch.futures.Future()
        self.pending_states = len(self.ob_rrefs)

    
    def select_action(agent_rref, ob_id, state):
        self = agent_rref.local_value()
        probs = self.policy(state.cuda())
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.time()



    @staticmethod
    @rpc.functions.async_execution
    def select_action_batch(agent_rref, ob_id, state):
        self = agent_rref.local_value()
        self.states[ob_id].copy_(state)
        future_action = self.future_actions.then(
            lambda future_actions: future_actions.wait()[ob_id].item()
        )

        with self.lock:
            self.pending_states -= 1
            if self.pending_states == 0:
                self.pending_states = len(self.ob_rrefs)
                probs = self.policy(self.states.cuda())
                m = Categorical(probs)
                actions = m.sample()
                self.saved_log_probs.append(m.log_prob(actions).t()[0])
                future_actions = self.future_actions
                self.future_actions = torch.future.Future()
                future_actions.set_result(actions.cpu())

        return future_action



        
