import numpy as np
from tinygrad import nn, Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.datasets import mnist
from training import train, evaluate

class ConvNet:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear = Tensor.scaled_uniform(256, 10)

    def __call__(self, x):
        x = self.conv1(x).relu().max_pool2d(kernel_size=(2, 2))
        x = self.conv2(x).relu().max_pool2d(kernel_size=(2, 2))
        x = x.reshape(x.shape[0], -1).dot(self.linear)
        return x
    
x_train, y_train, x_test, y_test = mnist()
x_train = x_train.numpy().astype(np.float32)
y_train = y_train.numpy().astype(np.int8)
x_test = x_test.numpy().astype(np.float32)
y_test = y_test.numpy().astype(np.int8)

# x_train = Tensor(x_train.numpy().astype(np.float32))
# y_train = Tensor(y_train.numpy().astype(np.int8))
# x_test = Tensor(x_test.numpy().astype(np.float32))
# y_test = Tensor(y_test.numpy().astype(np.int8))

model = ConvNet()
optimizer = nn.optim.Adam(get_parameters(model), lr =0.05)
train(model, x_train, y_train, optimizer, 500, BS=256)
evaluate(model, x_test, y_test, num_classes=10)