# this is a complete version of karpathy's version of gpt
# will be explaining exactl what happens here
# what is gt?
# gt -> general transformer

import torch
import keras
# import tensorflow
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16
block_size = 32
max_iters = 5000

eval_interval = 1000
learning_rate = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

# randomness setting
# find a fix here to get a different set of batch size
# check get_batch(split) function
torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# here are all the unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from chars to integers
stoi = {ch: i for i, ch in enumerate(chars)}
print(stoi)

itos = { i: ch for i,ch in enumerate(chars)}

# encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]
# print(encode)

# decode: take a list of integers, output a string
decode = lambda l: "".join([itos[i]] for i in l)
# print(decode)


# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
whole_value_of_data = int(1.0*len(data))
n = int(0.9*len(data))

print(whole_value_of_data)
print(n)

train_data = data[:n]
test_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    # generates a batch size of a random number 
    # why is it the same set of numbers?

    ix = torch.randint(len(data) - block_size, (batch_size,))
    # whats x?
    # repesents the batch size and the block size
    # 16 and 32 here
    # i is from the randomly generated batch size
    x = torch.stack([data[i: i + block_size] for i in ix])

    y = torch.stack([data[i+1: i + block_size+ 1] for i in ix])

    x, y = x.to(device), y.to(device)
    print(data)



@torch.no_grad()
def estimate_loss():
    pass



# for debugging purposes onl
# remove this shit after your done nigga!
if __name__ == "__main__":
    get_batch("train")




