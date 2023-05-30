import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

# lets see a single head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias =False)
query = nn.Linear(C, head_size, bias = False)
value = nn.Linear(C, head_size, bias = False)
k = key(x)
q = query(x)

wei = q @ k.transpose(-2, -1)
print(wei)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim = -1)

v = value(x)
out = wei @ v
print(out.shape)

