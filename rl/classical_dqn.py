"""
Deep Q-Network clássica (Mnih et al., 2015) em NumPy puro.

Componentes:
- MLP (entrada -> oculta -> oculta -> Q(s, a)) com ReLU
- Otimizador Adam implementado manualmente
- Perda de Huber (smooth L1) sobre o erro TD
- Rede-alvo (target network) sincronizada periodicamente
- Double DQN opcional (van Hasselt et al., 2016)
"""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=None):
        rng = np.random.default_rng(seed)
        self.params = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            W = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_out))  # He init
            b = np.zeros(n_out)
            self.params += [W, b]

    @property
    def n_params(self):
        return int(sum(p.size for p in self.params))

    def forward(self, x, cache=False):
        acts = [x]
        h = x
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            z = h @ W + b
            h = np.maximum(z, 0) if i < n_layers - 1 else z
            acts.append(h)
        return (h, acts) if cache else h

    def backward(self, acts, grad_out):
        grads = [None] * len(self.params)
        n_layers = len(self.params) // 2
        g = grad_out
        for i in reversed(range(n_layers)):
            W = self.params[2 * i]
            h_in = acts[i]
            grads[2 * i] = h_in.T @ g
            grads[2 * i + 1] = g.sum(axis=0)
            if i > 0:
                g = (g @ W.T) * (acts[i] > 0)
        return grads

    def copy_from(self, other):
        self.params = [p.copy() for p in other.params]


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=10.0):
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        # clipping pela norma global
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if norm > self.clip:
            grads = [g * (self.clip / norm) for g in grads]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def huber(delta, k=1.0):
    a = np.abs(delta)
    loss = np.where(a <= k, 0.5 * delta ** 2, k * (a - 0.5 * k))
    grad = np.where(a <= k, delta, k * np.sign(delta))
    return loss.mean(), grad


class ClassicalDQN:
    kind = "classical"

    def __init__(self, state_dim, n_actions, hidden=64, lr=1e-3, gamma=0.99,
                 double=True, seed=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.double = double
        sizes = [state_dim, hidden, hidden, n_actions]
        self.q = MLP(sizes, seed)
        self.target = MLP(sizes, seed)
        self.target.copy_from(self.q)
        self.opt = Adam(self.q.params, lr=lr)
        self.architecture = sizes

    @property
    def n_params(self):
        return self.q.n_params

    def q_values(self, states):
        return self.q.forward(np.atleast_2d(states))

    def update(self, s, a, r, s2, d):
        B = len(a)
        q_all, acts = self.q.forward(s, cache=True)
        q_sa = q_all[np.arange(B), a]

        q_next_target = self.target.forward(s2)
        if self.double:
            a_star = self.q.forward(s2).argmax(axis=1)
            q_next = q_next_target[np.arange(B), a_star]
        else:
            q_next = q_next_target.max(axis=1)
        y = r + self.gamma * (1.0 - d) * q_next

        loss, g = huber(q_sa - y)
        grad_out = np.zeros_like(q_all)
        grad_out[np.arange(B), a] = g / B
        grads = self.q.backward(acts, grad_out)
        self.opt.step(self.q.params, grads)
        return float(loss)

    def sync_target(self):
        self.target.copy_from(self.q)
