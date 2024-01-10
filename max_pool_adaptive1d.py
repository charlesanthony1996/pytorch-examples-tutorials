import torch
import torch.nn.functional as F
import torch.nn as nn

x = nn.AdaptiveMaxPool1d(5)
# print(x)
input = torch.randn(1, 1, 5)
# print(input)
output = x(input)

input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
F.avg_pool1d(input, kernel_size = 3, stride = 2)



print(output)
print(input)


mse_loss = nn.MSELoss()
loss = mse_loss(output, input)

print("mse loss: ", loss)

mae_loss = nn.L1Loss()
loss = mae_loss(output, input)
print("mae loss : ",loss)

