import threading
import torchvision
import torch
import torch.distributed.rpc as rpc
from torch import optim


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
        pass