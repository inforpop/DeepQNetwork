"""
Deep Q-Network Quântica: a rede neural é substituída por um
Circuito Quântico Variacional (VQC) simulado.

Q(s, a) = w_a * <Z_a>_θ(s) + b_a
- θ  : ângulos das portas de rotação (gradiente pela regra de deslocamento)
- w,b: escala/viés clássicos de saída (gradiente analítico)
"""
import numpy as np

from .classical_dqn import Adam, huber
from .quantum import VQC


class QuantumDQN:
    kind = "quantum"

    def __init__(self, n_actions, n_qubits=4, n_layers=3, encoding="basis",
                 lr=0.01, lr_out=0.05, gamma=0.99, out_scale=1.0, reupload=True, seed=None):
        assert n_actions <= n_qubits
        self.n_actions = n_actions
        self.gamma = gamma
        self.vqc = VQC(n_qubits, n_layers, encoding, reupload, seed)
        self.target_theta = self.vqc.theta.copy()
        self.w = np.full(n_actions, float(out_scale))
        self.b = np.zeros(n_actions)
        self.target_w, self.target_b = self.w.copy(), self.b.copy()
        self.opt_theta = Adam([self.vqc.theta], lr=lr)
        self.opt_out = Adam([self.w, self.b], lr=lr_out)

    @property
    def n_params(self):
        return int(self.vqc.n_params + self.w.size + self.b.size)

    def _prep(self, states):
        X = np.asarray(states, dtype=np.float64)
        if self.vqc.encoding == "basis":
            return X.reshape(-1, 1)
        return np.atleast_2d(X)

    def q_values(self, states, target=False):
        X = self._prep(states)
        th = self.target_theta if target else None
        E = self.vqc.expectations(X, th)[:, : self.n_actions]
        w, b = (self.target_w, self.target_b) if target else (self.w, self.b)
        return E * w + b

    def update(self, s, a, r, s2, d):
        B = len(a)
        X = self._prep(s)
        E0, dE = self.vqc.expectations_and_grad(X)
        E0 = E0[:, : self.n_actions]
        dE = dE[:, :, : self.n_actions]
        q_all = E0 * self.w + self.b
        q_sa = q_all[np.arange(B), a]

        # Double DQN: ação escolhida pela rede online, avaliada pela alvo
        a_star = self.q_values(s2).argmax(axis=1)
        q_next = self.q_values(s2, target=True)[np.arange(B), a_star]
        y = r + self.gamma * (1.0 - d) * q_next

        loss, g = huber(q_sa - y)
        gQ = np.zeros_like(q_all)
        gQ[np.arange(B), a] = g / B                       # dL/dQ (B, A)

        g_theta = np.einsum("ba,pba->p", gQ * self.w, dE)  # cadeia via <Z>
        g_w = (gQ * E0).sum(axis=0)
        g_b = gQ.sum(axis=0)

        self.opt_theta.step([self.vqc.theta], [g_theta])
        self.opt_out.step([self.w, self.b], [g_w, g_b])
        return float(loss)

    def sync_target(self):
        self.target_theta = self.vqc.theta.copy()
        self.target_w, self.target_b = self.w.copy(), self.b.copy()
