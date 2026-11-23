"""
Gerenciador de treinos em segundo plano.

O Heroku encerra requisições HTTP após 30 s, por isso o treino roda em uma
thread e o dashboard consulta o progresso periodicamente (polling).
"""
import copy
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np

from .classical_dqn import ClassicalDQN
from .envs import CartPole, GridWorld, make_env
from .quantum_dqn import QuantumDQN
from .replay import ReplayBuffer

MAX_JOBS = 20
MAX_RUNNING = 3

DEFAULTS = {
    "classical": {
        "cartpole": dict(episodes=300, lr=5e-4, gamma=0.99, hidden=64, batch=64,
                         eps_start=1.0, eps_end=0.02, eps_decay=0.98,
                         target_sync=500, warmup=500, buffer=20000, double=True),
        "gridworld": dict(episodes=200, lr=3e-3, gamma=0.95, hidden=32, batch=32,
                          eps_start=1.0, eps_end=0.05, eps_decay=0.97,
                          target_sync=50, warmup=64, buffer=5000, double=True),
    },
    "quantum": {
        "gridworld": dict(episodes=150, lr=0.02, gamma=0.95, layers=3, batch=16,
                          eps_start=1.0, eps_end=0.05, eps_decay=0.96,
                          target_sync=100, warmup=100, buffer=2000, out_scale=1.0),
        "cartpole": dict(episodes=150, lr=0.01, gamma=0.99, layers=4, batch=16,
                         eps_start=1.0, eps_end=0.05, eps_decay=0.97,
                         target_sync=300, warmup=300, buffer=10000, out_scale=20.0),
    },
}

# limites de segurança para o servidor
LIMITS = {"episodes": (10, 1000), "hidden": (8, 256), "layers": (1, 8),
          "batch": (8, 128), "buffer": (500, 50000)}


def _clean_config(mode, env, cfg):
    base = dict(DEFAULTS[mode][env])
    for k, v in (cfg or {}).items():
        if k not in base:
            continue
        if isinstance(base[k], bool):
            base[k] = bool(v)
        elif isinstance(base[k], int):
            base[k] = int(float(v))
        else:
            base[k] = float(v)
    for k, (lo, hi) in LIMITS.items():
        if k in base:
            base[k] = int(min(max(base[k], lo), hi))
    base["gamma"] = float(min(max(base["gamma"], 0.5), 0.999))
    base["eps_decay"] = float(min(max(base["eps_decay"], 0.8), 0.9999))
    return base


class Job:
    def __init__(self, mode, env_name, cfg, seed):
        self.id = uuid.uuid4().hex[:10]
        self.mode = mode
        self.env_name = env_name
        self.cfg = cfg
        self.seed = seed
        self.status = "na fila"
        self.error = None
        self.history = {"reward": [], "avg": [], "loss": [], "epsilon": [], "steps": []}
        self.total_steps = 0
        self.started = time.time()
        self.finished = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.agent = None
        self.best_agent = None   # cópia do agente no melhor momento (média móvel)
        self.best_avg = None
        self.best_eval = None    # melhor desempenho da política gulosa

    # ------------------------------------------------------------ helpers
    def encode(self, s):
        if self.env_name == "cartpole":
            return CartPole.normalize(s)
        if self.mode == "quantum":
            return np.array([float(s)])
        return GridWorld.one_hot(s)

    def build_agent(self):
        c = self.cfg
        if self.mode == "classical":
            dim = 4 if self.env_name == "cartpole" else 16
            n_act = 2 if self.env_name == "cartpole" else 4
            return ClassicalDQN(dim, n_act, hidden=c["hidden"], lr=c["lr"],
                                gamma=c["gamma"], double=c["double"], seed=self.seed)
        n_act = 2 if self.env_name == "cartpole" else 4
        enc = "angle" if self.env_name == "cartpole" else "basis"
        return QuantumDQN(n_act, n_qubits=4, n_layers=c["layers"], encoding=enc,
                          lr=c["lr"], gamma=c["gamma"], out_scale=c["out_scale"],
                          seed=self.seed)

    # ------------------------------------------------------------ treino
    def run(self):
        try:
            self.status = "treinando"
            c = self.cfg
            env = make_env(self.env_name, self.seed)
            self.agent = agent = self.build_agent()
            dim = len(self.encode(env.reset()))
            buf = ReplayBuffer(c["buffer"], dim, self.seed)
            rng = np.random.default_rng(self.seed)
            eps = c["eps_start"]

            for ep in range(c["episodes"]):
                if self.stop_flag.is_set():
                    self.status = "interrompido"
                    break
                s = self.encode(env.reset())
                ep_r, losses, steps = 0.0, [], 0
                while True:
                    if rng.random() < eps:
                        a = int(rng.integers(agent.n_actions))
                    else:
                        a = int(agent.q_values(s[None, :] if s.ndim else s).argmax())
                    s2_raw, r, term, trunc = env.step(a)
                    s2 = self.encode(s2_raw)
                    buf.push(s, a, r, s2, term)
                    s = s2
                    ep_r += r
                    steps += 1
                    self.total_steps += 1

                    if len(buf) >= max(c["warmup"], c["batch"]):
                        bs, ba, br, bs2, bd = buf.sample(c["batch"])
                        if self.mode == "quantum" and self.env_name == "gridworld":
                            bs, bs2 = bs.ravel(), bs2.ravel()
                        losses.append(agent.update(bs, ba, br, bs2, bd))
                    if self.total_steps % c["target_sync"] == 0:
                        agent.sync_target()
                    if term or trunc or self.stop_flag.is_set():
                        break

                eps = max(c["eps_end"], eps * c["eps_decay"])
                with self.lock:
                    h = self.history
                    h["reward"].append(round(ep_r, 3))
                    window = h["reward"][-20:]
                    avg = float(np.mean(window))
                    h["avg"].append(round(avg, 3))
                    h["loss"].append(round(float(np.mean(losses)), 5) if losses else None)
                    h["epsilon"].append(round(eps, 4))
                    h["steps"].append(steps)
                    if self.best_avg is None or avg > self.best_avg:
                        self.best_avg = avg

                # avalia a política gulosa e guarda o melhor modelo real
                every = 1 if self.env_name == "gridworld" else 5
                if len(buf) >= c["warmup"] and ep % every == 0:
                    score = self.greedy_score(agent)
                    if self.best_eval is None or score >= self.best_eval:
                        self.best_eval = score
                        self.best_agent = copy.deepcopy(agent)
            else:
                self.status = "concluído"
            if self.status == "treinando":
                self.status = "concluído"
        except Exception as exc:  # pragma: no cover
            self.status = "erro"
            self.error = repr(exc)
        finally:
            self.finished = time.time()

    # ------------------------------------------------------------ relatórios
    def snapshot(self, since=0):
        with self.lock:
            hist = {k: v[since:] for k, v in self.history.items()}
            n = len(self.history["reward"])
        elapsed = (self.finished or time.time()) - self.started
        return {
            "id": self.id, "mode": self.mode, "env": self.env_name,
            "status": self.status, "error": self.error, "config": self.cfg,
            "episode": n, "episodes": self.cfg["episodes"], "since": since,
            "history": hist, "total_steps": self.total_steps,
            "elapsed": round(elapsed, 1),
            "best_avg": None if self.best_avg is None else round(self.best_avg, 3),
            "best_eval": self.best_eval,
            "n_params": self.agent.n_params if self.agent else None,
            "done": self.status in ("concluído", "interrompido", "erro"),
        }

    def greedy_score(self, agent):
        """Retorno de um episódio sem exploração (GridWorld) ou passos (CartPole)."""
        if self.env_name == "gridworld":
            X = np.arange(16, dtype=float) if self.mode == "quantum" else np.eye(16)
            greedy = agent.q_values(X).argmax(1)
            env, ret, done = GridWorld(), 0.0, False
            st = env.reset()
            while not done:
                st, r, term, trunc = env.step(int(greedy[st]))
                ret += r
                done = term or trunc
            return round(ret, 4)
        env = CartPole(seed=self.seed + 1)
        st, n, done = env.reset(), 0, False
        while not done:
            a = int(agent.q_values(CartPole.normalize(st)[None, :]).argmax())
            st, _, term, trunc = env.step(a)
            n += 1
            done = term or trunc
        return n

    def _eval_agent(self, which="best"):
        if which == "best" and self.best_agent is not None:
            return self.best_agent
        return self.agent

    def policy(self, which="best"):
        """Avalia a política gulosa (sem exploração)."""
        if self.agent is None:
            return {"error": "agente ainda não criado"}
        agent = self._eval_agent(which)
        if self.env_name == "gridworld":
            X = np.arange(16, dtype=float) if self.mode == "quantum" else np.eye(16)
            Q = agent.q_values(X)
            env = GridWorld()
            s, path, done, ret = env.reset(), [0], False, 0.0
            while not done:
                s, r, term, trunc = env.step(int(Q[s].argmax()))
                path.append(int(s)); ret += r
                done = term or trunc
            return {"env": "gridworld", "layout": GridWorld.layout,
                    "q": np.round(Q, 4).tolist(), "greedy": Q.argmax(1).tolist(),
                    "path": path, "success": path[-1] == GridWorld.goal,
                    "return": round(ret, 3), "model": which}
        env = CartPole(seed=int(time.time()) % 10000)
        s = env.reset()
        frames, done = [], False
        while not done:
            frames.append([round(float(s[0]), 4), round(float(s[2]), 4)])
            a = int(agent.q_values(CartPole.normalize(s)[None, :]).argmax())
            s, r, term, trunc = env.step(a)
            done = term or trunc
        frames.append([round(float(s[0]), 4), round(float(s[2]), 4)])
        return {"env": "cartpole", "frames": frames, "steps": len(frames) - 1, "model": which}

    def statevector(self, s):
        if self.mode != "quantum" or self.agent is None:
            return {"error": "disponível apenas para o modo quântico"}
        vqc = self._eval_agent().vqc
        if self.env_name == "gridworld":
            x = np.array([float(int(s) % 16)])
        else:
            x = CartPole.normalize(np.array([0.0, 0.0, 0.05, 0.0]))
        psi = vqc.statevector(x)
        probs = np.abs(psi) ** 2
        return {"basis": [format(i, f"0{vqc.n}b") for i in range(vqc.sim.dim)],
                "probs": np.round(probs, 5).tolist(),
                "phases": np.round(np.angle(psi), 4).tolist(),
                "expect_z": np.round(vqc.sim.expect_z(psi[None, :])[0], 4).tolist()}


class JobManager:
    def __init__(self):
        self.jobs = OrderedDict()
        self.lock = threading.Lock()

    def running(self):
        return sum(1 for j in self.jobs.values() if not j.finished)

    def start(self, mode, env_name, cfg, seed=None):
        if mode not in DEFAULTS or env_name not in DEFAULTS[mode]:
            raise ValueError("modo/ambiente inválido")
        with self.lock:
            if self.running() >= MAX_RUNNING:
                raise RuntimeError("Servidor ocupado: aguarde um treino terminar.")
            while len(self.jobs) >= MAX_JOBS:
                old = next((k for k, j in self.jobs.items() if j.finished), None)
                if old is None:
                    break
                self.jobs.pop(old)
            seed = int(seed) if seed not in (None, "") else int(time.time()) % 100000
            job = Job(mode, env_name, _clean_config(mode, env_name, cfg), seed)
            self.jobs[job.id] = job
        threading.Thread(target=job.run, daemon=True).start()
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)
