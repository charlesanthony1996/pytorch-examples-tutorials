import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float32)
x = F.avg_pool1d(input, kernel_size=3, stride= 1)

print(x)
