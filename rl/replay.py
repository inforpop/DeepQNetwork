"""Memória de replay (Experience Replay) circular."""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim, seed=None):
        self.capacity = int(capacity)
        self.s = np.zeros((self.capacity, state_dim), dtype=np.float64)
        self.a = np.zeros(self.capacity, dtype=np.int64)
        self.r = np.zeros(self.capacity, dtype=np.float64)
        self.s2 = np.zeros((self.capacity, state_dim), dtype=np.float64)
        self.d = np.zeros(self.capacity, dtype=np.float64)
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(done)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ix = self.rng.integers(0, self.size, size=batch_size)
        return self.s[ix], self.a[ix], self.r[ix], self.s2[ix], self.d[ix]

    def __len__(self):
        return self.size
