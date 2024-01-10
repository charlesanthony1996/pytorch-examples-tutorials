from torch.nn.functional import *
import torch

print(torch.nn.functional.one_hot(torch.arange(0, 5) % 2))

import torch.nn.functional as F

x = F.one_hot(torch.arange(0, 2) * 2, num_classes = 5)

print(x)


y = F.one_hot(torch.arange(0, 6).view(3, 2) % 3)

print(y)

# working with ctc loss

log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
print(log_probs)
targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
print(targets)
input_lengths = torch.full((16,), 50, dtype=torch.long)
print(input_lengths)

target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
print(target_lengths)

# calculating loss here
loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)

print(loss)