"""
Ambientes de Aprendizado por Reforço implementados em NumPy puro
(sem dependência do Gym, para manter o deploy no Heroku leve).

- CartPole: equilibrar um pêndulo invertido sobre um carrinho (estado contínuo).
- GridWorld (FrozenLake 4x4 determinístico): chegar ao objetivo evitando buracos
  (estado discreto, ideal para codificação em 4 qubits).
"""
import math
import numpy as np


class CartPole:
    """Física clássica do CartPole-v1 (Barto, Sutton & Anderson, 1983)."""

    name = "cartpole"
    n_actions = 2
    state_dim = 4
    max_steps = 500

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.state = None
        self.steps = 0

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = math.cos(theta), math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        terminated = bool(
            x < -self.x_threshold or x > self.x_threshold
            or theta < -self.theta_threshold or theta > self.theta_threshold
        )
        truncated = self.steps >= self.max_steps
        reward = 1.0
        return self.state.copy(), reward, terminated, truncated

    @staticmethod
    def normalize(s):
        """Escala aproximada para a rede neural."""
        return np.asarray(s) / np.array([2.4, 3.0, 0.21, 3.5])


class GridWorld:
    """
    FrozenLake 4x4 determinístico.
        S . . .
        . H . H
        . . . H
        H . . G
    Ações: 0=esquerda, 1=baixo, 2=direita, 3=cima.
    """

    name = "gridworld"
    n_actions = 4
    n_states = 16
    state_dim = 16
    max_steps = 40
    size = 4
    holes = (5, 7, 11, 12)
    goal = 15
    layout = ["SFFF", "FHFH", "FFFH", "HFFG"]

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.state = 0
        self.steps = 0

    def reset(self):
        self.state = 0
        self.steps = 0
        return self.state

    def step(self, action):
        r, c = divmod(self.state, self.size)
        if action == 0:
            c = max(c - 1, 0)
        elif action == 1:
            r = min(r + 1, self.size - 1)
        elif action == 2:
            c = min(c + 1, self.size - 1)
        elif action == 3:
            r = max(r - 1, 0)
        self.state = r * self.size + c
        self.steps += 1

        if self.state == self.goal:
            return self.state, 1.0, True, False
        if self.state in self.holes:
            return self.state, -1.0, True, False
        truncated = self.steps >= self.max_steps
        return self.state, -0.01, False, truncated

    @staticmethod
    def one_hot(s):
        v = np.zeros(16)
        v[int(s)] = 1.0
        return v


def make_env(name, seed=None):
    if name == "cartpole":
        return CartPole(seed)
    if name == "gridworld":
        return GridWorld(seed)
    raise ValueError(f"Ambiente desconhecido: {name}")
