import torch
import numpy
import pandas
import torchvision.models as models
import time
import torch.nn as nn

model = models.resnet50(pretrained=False)
print(model)
model.eval()
print(model.eval())

batch_size = 32

data = torch.rand(batch_size, 3 , 224, 224)

for _ in range(100):
    model(data)


with torch.autograd.profiler.emit_itt():
    start = time.time()
    for i in range(100):
        torch.profiler.itt.range_push("step{}".format(i))
        model(data)
        torch.profiler.itt.range_pop()
    end = time.time()


print("Inference took {:.2f} ms in average".format((end - start) / 100 * 1000))


import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(16, 33, 3, stride = 2)
        self.relu = torch.nn.ReLU()

    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

    
model = Model()
model.eval()

data = torch.rand(20, 16, 50, 100)

import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)
print()
print(model)
print()
with torch.no_grad():
    model = torch.jit.trace(model, data)
    model = torch.jit.freeze(model)



