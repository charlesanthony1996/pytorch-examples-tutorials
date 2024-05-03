import torch
from torch import nn

import math
import matplotlib.pyplot as plt

torch.manual_seed(111)

train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
print(train_data)