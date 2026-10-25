import time
import mlx.core as mx

d = 256
chunk = 256
steps = 1024

mx.random.seed(0)
wq, wk, wv = (mx.random.normal((d, d)) * d **-0.5 for _ in range(3))

print(wq)
print(wk)
print(wv)

print(wq.shape)
print(wk.shape)
print(wv.shape)

tokens = mx.random.normal((1, steps, d))

def attend(q, k, v):

    scores = (q @ k.transpose(0, 2, 1)) * d**-0.5

    return mx.softmax(scores, axis=1) @ v


class kvcache:
    def __init__(self):

        self.k = mx.zeros((1, chunk, d))
        self.v = mx.zeros((1, chunk, d))
        self.offset = 0

    def update(self, k, v):

        if self.offset == self.k.shape[1]:
            pad = mx.zeros((1, chunk, d))
            self.k = mx.concatenate([self.k, pad], axis = 1)
            self.v = mx.concatenate([self.v, pad], axis = 1)

        self.k[:, self.offset : self.offset + 1, :] = k
        self.v[:, self.offset: self.offset + 1, :] = v

        self.offset += 1
        return self.k[:, :self.offset], self.v[:, :self.offset]


def step_uncached(history):

    q = history[:, -1:] @ wq
    return attend(q, history @ wk,history @ wv)

def step_cached(x, cache):

    k_all, v_all = cache.update(x @ wk, x * wv)

    return attend(x @ wq, k_all, v_all)

t0 = time.perf_counter()
# print(t0)
out_uncached = []
for t in range(steps):
    y = step_uncached(tokens[:, : t + 1])
    mx.eval(y)
    out_uncached.append(y)

t_uncached = time.perf_counter() - t0

t0 = time.perf_counter()
cache = kvcache()
out_cached = []
for t in range(steps):

    y = step_cached(tokens[:, t : t+ 1], cache)
    mx.eval(y, cache.k, cache.v)
    out_cached.append(y)

t_cached = time.perf_counter() - t0

a = mx.concatenate(out_uncached, axis = 1)
b = mx.concatenate(out_cached, axis = 1)


