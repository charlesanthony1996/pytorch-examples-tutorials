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


def step_uncached():
    pass

def step_cached(x, cache):

    k_all, v_all = cache.update(x @ wk, x * wv)

    return attend(x @ wq, k_all, v_all)

t0 = time.perf_counter()
# print(t0)