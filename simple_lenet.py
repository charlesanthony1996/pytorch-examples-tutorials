import numpy as np
from PIL import Image
from tinygrad import nn
from tinygrad.nn.datasets import mnist
from training import train, evaluate

class LeNet:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.l1 = nn.Linear(400, 120)
        self.l2 = nn.Linear(120, 84)
        self.l3 = nn.Linear(84, 10)

    def __call__(self, x):
        x = self.bn1(self.conv1(x)).relu().max_pool2d(stride=2)
        x = self.bn2(self.conv2(x)).relu().max_pool2d(stride=2)
        x = x.reshape(x.shape[0], -1)
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x)
        return x
    


def transform(x):
    x = [Image.fromarray(xx).resize((32, 32)) for xx in x]
    x = np.stack([np.asarray(xx) for xx in x], axis= 0)
    x = x.reshape(-1, 1, 32, 32)
    return x



x_train, y_train, x_test, y_test = mnist()
x_train = x_train.reshape(-1, 28, 28).numpy().astype(np.float32)
y_train = y_train.numpy().astype(np.int8)
x_test = x_test.reshape(-1, 28, 28).numpy().astype(np.float32)
y_test = y_test.numpy().astype(np.int8)


model = LeNet()
optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=0.02)
train(model, x_train, y_train, optimizer, 1000, BS=256, transform=transform)
evaluate(model, x_test, y_test, transform=transform)
