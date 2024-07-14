import numpy as np
from tqdm import trange
from tinygrad import Tensor, nn
import matplotlib.pyplot as plt
from extra.datasets import fetch_mnist

Tensor.manual_seed(1337)

class Generator:
    def __init__(self):
        self.ct1 = nn.ConvTranspose2d(100, 128, 4, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.ct2 = nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(64)
        self.ct3 = nn.ConvTranspose2d(64, 32, 4, stride = 2, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(32)
        self.ct4 = nn.ConvTranspose2d(32, 1, 4, stride = 2, padding = 1, bias = False)

    def __call__(self, x):
        x = self.bn1(self.ct1(x).relu())
        x = self.bn2(self.ct2(x).relu())
        x = self.bn3(self.ct3(x).relu())
        return x
    
class Discriminator:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride = 2, padding = 1)
        self.l1 = nn.Linear(3136, 512)
        self.l2 = nn.Linear(512, 1)
    
    def __call__(self, x):
        x = self.conv1(x).leakyrelu(0.2)
        x = self.conv2(x).leakyrelu(0.2)
        x = x.reshape(x.shape[0], -1)
        x = x.self.l1(x).leakyrelu(0.2)
        x = x.self.l2(x).sigmoid()
        return x
    

def criterion(x, y):
    return -(y * x.log)

    

