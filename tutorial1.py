#!usr/bin/python3
import torch
import math
from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
from IPython.core.debugger import set_trace
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np


DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents = True, exist_ok = True)


URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME ).open("wb").write(content)


with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# print(x_train.shape)


# training and testing

x_train , y_train , x_valid , y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

n , c = x_train.shape
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())



weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad= True )


# print(weights)

# print(bias)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def forward():
    pass


# print(log_softmax(torch.rand(1)))

def model(xb):
    return log_softmax(xb @ weights + bias)
    
# print(model(log_softmax(torch.rand(10, 10))))

#defining batch sizes here
bs = 64

xb = x_train[0:bs]
preds = model(xb)
preds[0] , preds.shape
# print(preds[0], preds.shape)


def nll(input , target):
    return -input[range(target.shape[0]), target].mean()


loss_func = nll

#checking our loss function here

yb = y_train[0:bs]
# print(loss_func(preds,yb))

#accuracy model
def accuracy(out, yb):
    preds = torch.argmax(out, dim =1)
    return (preds == yb).float().mean()


# print(accuracy(preds, yb))

#tracing and debugging

lr = 0.5
epochs = 2

for epoch in range(epochs):
    for i in range((n -1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i: end_i]
        yb = y_train[start_i: end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()


# print(loss_func(model(xb), yb) , accuracy(model(xb) , yb))


loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias
    
#confirming loss and accuracy was the same before

# print(loss_func(model(xb), yb), accuracy(model(xb) , yb))


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias


#instantiating the model

model = Mnist_Logistic()

# print(loss_func(model(xb), yb))

# with torch.no_grad():
#     weights -= weights.grad * lr
#     bias -= bias.grad * lr
#     weights.grad.zero_()
#     bias.grad.zero_()



# with torch.no_grad():
#     for p in model.parameters(): p -= p.grad * lr
#         model.zero_grad()



def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i : end_i]
            yb = y_train[start_i: end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)
            
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()


# fit()



# print(loss_func(model(xb), yb))

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


    
model = Mnist_Logistic()
# print(loss_func(model(xb), yb))


fit()

# print(loss_func(model(xb) , yb))

with torch.no_grad():
    for p in model.parameters():
        p -= p.grad * lr
        model.zero_grad()


# opt.step()
# opt.zero_grad()

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr = lr)

model , opt = get_model()
# print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i: end_i]
        yb = y_train[start_i: end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)


        loss.backward()
        opt.step()
        opt.zero_grad()


# print(loss_func(model(xb), yb))

train_ds = TensorDataset(x_train, y_train)

xb = x_train[start_i:end_i]
yb = y_train[start_i:end_i]


xb, yb = train_ds[i*bs:i * bs + bs]


for epoch in range(epochs):
    for i in range((n -1) // bs+ 1):
        xb, yb = train_ds[i *bs: i *bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()


# print(loss_func(model(xb), yb))

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size = bs)

for i in range((n -1) //bs + 1):
    xb, yb = train_ds[i *bs : i* bs + bs]
    pred = model(xb)


for xb, yb in train_dl:
    pred = model(xb)

model , opt = get_model()


for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()


# print(loss_func(model(xb), yb))
# print("not trained yet")

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size = bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size = bs * 2)

# print("training and validating done")


model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()


    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
    print(epoch, valid_loss / len(valid_dl))

#create fit() and get_data()

def loss_batch(model, loss_func, xb, yb, opt = None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item() ,len(xb)


def fit(epochs, model, loss_func , opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb,opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            print(epoch, val_loss)


def get_data(train_ds, valid_ds, bs):
    return (DataLoader(train_ds, batch_size = bs, shuffle =False), (DataLoader(train_ds, batch_size =bs * 2, shuffle =False)))



# train_dl , valid_dl = get_data(train_ds, valid_ds, bs)
# model, opt = get_model()
# fit(epochs, model, loss_func, opt, train_dl, valid_dl)


#switching to CNN here


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= nn.Conv2d(1, 16, kernel_size = 3, stride =2, padding = 1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size= 3, stride = 2, padding =1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride= 2, padding =1)


    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1 , xb.size(1))
        


lr = 0.1



model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr = lr , momentum = 0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)




#nn.Sequential() -> custom layer


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)


model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride = 2, padding = 1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size = 3, stride=2 , padding = 1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size =3 , stride =2, padding= 1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)



#wrapping a dataloader around it

def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


train_dl , valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)



#using adaptive pool2d

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size = 3, stride = 2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3 , stride= 2, padding =1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr= lr, momentum = 0.9)


fit(epochs, model, loss_func, opt, train_dl, valid_dl)

