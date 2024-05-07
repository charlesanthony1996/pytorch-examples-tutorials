import torch
import torch.nn as nn
import torch.optim as optim

# define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# input and target
input = torch.randn(16, 10)
target = torch.randint(0, 2, (16, ))


# using torch.cuda.amp for mixed precision
scaler = torch.amp.GradScaler()

model.train()
optimizer.zero_grad()

# with torch.amp:
#     output = model(input)
#     loss = criterion(output, target)

output = model(input)
loss = criterion(output, target)


loss.backward()
optimizer.step()

print("finished training step on cpu")
