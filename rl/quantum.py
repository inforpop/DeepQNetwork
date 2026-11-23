"""
Simulador de vetor de estado (statevector) em NumPy e
Circuito Quântico Variacional (VQC) usado como aproximador da função Q.

Referências:
- Chen et al. (2020) "Variational Quantum Circuits for Deep Reinforcement Learning"
- Skolik, Jerbi & Dunjko (2022) "Quantum agents in the Gym"
- Regra de deslocamento de parâmetros (parameter-shift rule) para gradientes exatos.

O simulador é vetorizado: N circuitos (amostras x deslocamentos de parâmetros)
são simulados ao mesmo tempo, cada um com seu próprio vetor de ângulos.
"""
import numpy as np

SHIFT = np.pi / 2


# ---------------------------------------------------------------- portas
def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    m = np.empty((theta.size, 2, 2), dtype=np.complex128)
    m[:, 0, 0] = c; m[:, 0, 1] = -1j * s
    m[:, 1, 0] = -1j * s; m[:, 1, 1] = c
    return m


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    m = np.empty((theta.size, 2, 2), dtype=np.complex128)
    m[:, 0, 0] = c; m[:, 0, 1] = -s
    m[:, 1, 0] = s; m[:, 1, 1] = c
    return m


def _rz(theta):
    m = np.zeros((theta.size, 2, 2), dtype=np.complex128)
    m[:, 0, 0] = np.exp(-0.5j * theta)
    m[:, 1, 1] = np.exp(0.5j * theta)
    return m


GATES = {"RX": _rx, "RY": _ry, "RZ": _rz}


class StateVectorSimulator:
    """Simulador em lote. Qubit 0 = bit mais significativo."""

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        idx = np.arange(self.dim)
        self.bits = np.array([(idx >> (self.n - 1 - q)) & 1 for q in range(self.n)])
        self.zsign = 1.0 - 2.0 * self.bits  # (n, dim): autovalores de Z
        self._cnot_perm = {}

    def apply_1q(self, state, q, mats):
        """state (N, dim); mats (N, 2, 2) -> um gate por circuito."""
        N = state.shape[0]
        psi = state.reshape(N, 2 ** q, 2, 2 ** (self.n - q - 1))
        psi = np.einsum("nij,najb->naib", mats, psi)
        return psi.reshape(N, self.dim)

    def apply_cnot(self, state, c, t):
        key = (c, t)
        if key not in self._cnot_perm:
            idx = np.arange(self.dim)
            flip = 1 << (self.n - 1 - t)
            perm = np.where(self.bits[c] == 1, idx ^ flip, idx)
            self._cnot_perm[key] = perm
        return state[:, self._cnot_perm[key]]

    def expect_z(self, state):
        probs = np.abs(state) ** 2
        return probs @ self.zsign.T  # (N, n)


# ---------------------------------------------------------------- VQC
class VQC:
    """
    Ansatz hardware-efficient:
        [codificação] -> L x ( RX RY RZ em cada qubit  ->  anel de CNOTs )
    Codificações:
        'basis' : estado discreto s -> |s>  (GridWorld, 16 estados = 4 qubits)
        'angle' : variáveis contínuas -> RX(atan(x)) RY(atan(x)) com
                  data re-uploading a cada camada (CartPole)
    Leitura: <Z_a> no qubit a  ->  Q(s,a) = w_a * <Z_a> + b_a
    """

    ROTS = ("RX", "RY", "RZ")

    def __init__(self, n_qubits=4, n_layers=3, encoding="basis", reupload=True, seed=None):
        self.reupload = reupload
        self.n = n_qubits
        self.L = n_layers
        self.encoding = encoding
        self.sim = StateVectorSimulator(n_qubits)
        rng = np.random.default_rng(seed)
        self.n_params = n_layers * n_qubits * len(self.ROTS)
        self.theta = rng.uniform(-np.pi, np.pi, self.n_params) * 0.5

    # ----- codificação dos dados
    def _encode(self, X, N_rep):
        """X: (B, ...). Retorna estado inicial (N_rep*B, dim)."""
        B = X.shape[0]
        if self.encoding == "basis":
            st = np.zeros((B, self.sim.dim), dtype=np.complex128)
            st[np.arange(B), X.astype(int).ravel()] = 1.0
        else:
            st = np.zeros((B, self.sim.dim), dtype=np.complex128)
            st[:, 0] = 1.0
        return np.tile(st, (N_rep, 1))

    def _basis_layer(self, state, Xrep):
        """Reaplica X nos qubits cujo bit do estado é 1 (RX(pi) ~ X)."""
        bits = ((Xrep.astype(int).ravel()[:, None] >> (self.n - 1 - np.arange(self.n))) & 1)
        for q in range(self.n):
            state = self.sim.apply_1q(state, q, _rx(np.pi * bits[:, q]))
        return state

    def _angle_layer(self, state, Xrep):
        ang = np.arctan(Xrep)  # (N, n)
        for q in range(self.n):
            state = self.sim.apply_1q(state, q, _rx(ang[:, q]))
            state = self.sim.apply_1q(state, q, _ry(ang[:, q]))
        return state

    # ----- execução
    def run(self, X, theta_rows):
        """
        X: (B, feat) ; theta_rows: (K, P) -> expectativas (K, B, n)
        Cada combinação (k, b) é um circuito independente.
        """
        K = theta_rows.shape[0]
        B = X.shape[0]
        state = self._encode(X, K)
        th = np.repeat(theta_rows, B, axis=0)  # (K*B, P)
        Xrep = np.tile(X, (K, 1))

        p = 0
        for layer in range(self.L):
            if self.encoding == "angle":
                state = self._angle_layer(state, Xrep)  # data re-uploading
            elif self.encoding == "basis" and self.reupload and layer > 0:
                state = self._basis_layer(state, Xrep)  # re-uploading dos bits
            for q in range(self.n):
                for g in self.ROTS:
                    state = self.sim.apply_1q(state, q, GATES[g](th[:, p]))
                    p += 1
            for q in range(self.n):
                state = self.sim.apply_cnot(state, q, (q + 1) % self.n)
        return self.sim.expect_z(state).reshape(K, B, self.n), state

    def expectations(self, X, theta=None):
        th = self.theta if theta is None else theta
        E, _ = self.run(X, th[None, :])
        return E[0]

    def expectations_and_grad(self, X):
        """Valores e derivadas exatas via regra do deslocamento de parâmetros."""
        P = self.n_params
        rows = np.tile(self.theta, (2 * P + 1, 1))
        for i in range(P):
            rows[1 + 2 * i, i] += SHIFT
            rows[2 + 2 * i, i] -= SHIFT
        E, _ = self.run(X, rows)
        E0 = E[0]                                   # (B, n)
        dE = 0.5 * (E[1::2] - E[2::2])              # (P, B, n)
        return E0, dE

    def statevector(self, x):
        X = np.atleast_2d(x)
        _, st = self.run(X, self.theta[None, :])
        return st[0]

    def describe(self):
        """Estrutura do circuito para desenho no dashboard."""
        ops = []
        if self.encoding == "basis":
            ops.append({"type": "encode", "label": "|s⟩ (base)"})
        p = 0
        for layer in range(self.L):
            if self.encoding == "angle":
                ops.append({"type": "encode", "label": "RX·RY(atan x)", "layer": layer})
            elif self.reupload and layer > 0:
                ops.append({"type": "encode", "label": "RX(π·bit)", "layer": layer})
            for q in range(self.n):
                for g in self.ROTS:
                    ops.append({"type": "rot", "gate": g, "qubit": q, "param": p, "layer": layer})
                    p += 1
            for q in range(self.n):
                ops.append({"type": "cnot", "control": q, "target": (q + 1) % self.n, "layer": layer})
        return {"n_qubits": self.n, "n_layers": self.L, "encoding": self.encoding, "ops": ops}
