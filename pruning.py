import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

# create the lenet module

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84 , 10)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = LeNet().to(device = device)


print(model)

print()

module = model.conv1
print(list(module.named_parameters()))

print()



prune.random_unstructured(module, name="weight", amount=0.3)

print()

print(list(module.named_parameters()))

print()

print(list(module.named_buffers()))

print()

print(module.weight)

print()

print(module._forward_pre_hooks)

print()

prune.l1_unstructured(module, name="bias", amount = 3)

print()

print(list(module.named_parameters()))

print()

print(list(module.named_buffers()))

print()

print(module.bias)

print()

print(module._forward_pre_hooks)

prune.ln_structured(module, name="weight", amount=0.5, n = 2, dim = 0)

print()

print(module.weight)


prune.remove(module, "weight")
print(list(module.named_parameters()))

print()

print(list(module.named_buffers()))

print()

# pruning multiple parameters in a model

new_model = LeNet()
for name , module in new_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name="weight", amount = 0.2)
    #prune 40% of connections in linear layers
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.4)


print(dict(new_model.named_buffers()).keys())

#global pruning

model = LeNet()

parameters_to_prune = (
    (model.conv1, "weight"),
    (model.conv2, "weight"),
    (model.fc1, "weight"),
    (model.fc2 , "weight"),
    (model.fc3, "weight")
)


prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2
)


print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    )
)
print(
    "Sparsity in conv2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv2.weight == 0))
        / float(model.conv2.weight.nelement())
    )
)
print(
    "Sparsity in fc1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc1.weight == 0))
        / float(model.fc1.weight.nelement())
    )
)
print(
    "Sparsity in fc2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc2.weight == 0))
        / float(model.fc2.weight.nelement())
    )
)
print(
    "Sparsity in fc3.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc3.weight == 0))
        / float(model.fc3.weight.nelement())
    )
)
print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
            + torch.sum(model.fc2.weight == 0)
            + torch.sum(model.fc3.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
            + model.fc2.weight.nelement()
            + model.fc3.weight.nelement()
        )
    )
)










