import torch


with open("input.txt", "r", encoding= "utf-8") as f:
    text = f.read()


# print("length of the dataset in the charcters: ", len(text))

# print(text[:1000])


# here all the unique charachters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("".join(chars))
# print(vocab_size)



# create a mapping from chars to integers
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: [ ''.join([itos[i] for i in l])]


# print(encode("hii there"))
# print(decode(encode("hii there")))

print()


import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# the 1000 chars we looked at earlier
print(data[:1000])



# splitting the data to train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


block_size = 8
train_data[:block_size + 1]

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"When input is {context} the target: {target}")



torch.manual_seed(1337)
# how many independent sequences will we process in parallel
batch_size = 4
# whats the maximum context length of the predictions
block_size = 8

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else data
    ix = torch.randint(len(data) - block_size ,(batch_size,))
    x = torch.stack([data [i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

    return x, y


xb, yb = get_batch("train")
print("inputs: ")
print(xb.shape)
print(xb)

print("targets: ")
print(yb.shape)
print(yb)

print("-----------")


for b in range(batch_size):
    # batch dimension
    for t in range(block_size):
        context = xb[b, :t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")


# our input to the transformer
print(xb)

# get this cleared out
# print(yb)

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets = None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1,:]
            # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1)
            
        return idx



m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)



# decoding
print(decode(m.generate(idx = torch.zeros((1, 1), dtype = torch.long), max_new_tokens=100)[0].tolist()))
print()



# create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)


print(optimizer)


batch_size = 32

# increase the numbers of steps for good results
for steps in range(100):
    # sample a batch size
    xb, yb = get_batch("train")
    # evaluate the loss
    # this area is not complete







# the math behind self attention
torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b

print("a=")
print(a)
print("--")
print("b=")
print(b)
print("--")
print("c=")
print(c)



# consider the following toy example
# torch.manual_seed(1337)

# batch time and the channels
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
x.shape
# print it
print(x.shape)


# we want to x[b, t] = mean_{i<=t} x[b, i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        #[T, C]
        xprev = x[b, :t+ 1]
        xbow[b, t] = torch.mean(xprev, 0)


    
# version 2: using matrix multiply for a weighted aggression
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x
torch.allclose(xbow, xbow2)


# version 3 softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim = 1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)


# version 4 self attention






