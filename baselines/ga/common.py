"""
Module: baselines.ga.common
Description: Shared pieces of the genetic-algorithm baselines: the two reward tasks built with
             src.env.make_env (the PPO TRAINING reward, used as fitness, and the evaluation task), a
             per-frame sticky-actions patch, the episode runner with the per-episode metrics, a persistent
             worker pool (1 emulator per process).

Nothing in src/ is modified. Both tasks are exactly make_env(); they differ only in the reward:
  - "train": the reward PPO is trained on (src/train.py: d, m=DAMAGE_PENALTY_MULT, WIN_BONUS). It is
    the fitness, so the GAs and PPO optimise the same function.
  - "eval": the evaluation task (d=0.05, m=1, no bonus).
The game dynamics do not depend on the reward, so the same individual plays the same trajectory in both.
Sticky actions are injected below the ActionSkip/Frameskip wrappers by patching the retro env's step
(per-frame stickiness, Machado et al. 2018), so the wrapper stack stays identical to the one PPO is
evaluated in.
"""

import multiprocessing as mp
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Bosses: save state and episode length in emulator frames (the TimeLimit is frames // 4 agent steps).
BOSSES = {b: {"state": f"{b.capitalize()}-boss", "frames": 7200}
          for b in ("gutsman", "bombman", "cutman", "elecman", "iceman", "fireman")}
D = 0.05          # reward scale of the evaluation task
FRAMESKIP = 4
ACTION_DIMS = (2, 5, 2)   # MultiDiscrete: [B, D-pad (none/UP/DOWN/LEFT/RIGHT), A]
N_GENES = 20              # 2 * 5 * 2 joint actions


# ----------------------------------------------------------------------------------------------
# Evaluation environment
# ----------------------------------------------------------------------------------------------
class FrameCounter:
    """
    Patches the retro env's step/reset to (a) count emulator frames and (b) optionally apply
    per-frame sticky actions: with probability p the frame repeats the previous frame's executed
    action (Machado et al. 2018; the first frame repeats a no-op). It sits
    BELOW ActionSkipWrapper/FrameskipWrapper, so the B toggle and the blink-based frame extension are
    unchanged and stickiness acts on each emulator frame.
    """
    def __init__(self, env, p=0.0, rng=None):
        self.raw = env.unwrapped
        self.p = p
        self.rng = rng
        self.frames = 0
        self.prev = np.zeros(3, dtype=np.int64)
        self._step = self.raw.step
        self._reset = self.raw.reset
        self.raw.step = self.step
        self.raw.reset = self.reset

    def reset(self, *args, **kwargs):
        self.frames = 0
        self.prev = np.zeros(3, dtype=np.int64)
        return self._reset(*args, **kwargs)

    def step(self, action):
        action = np.array(action)
        if self.p > 0.0 and self.rng.random() < self.p:
            action = self.prev.copy()
        self.prev = action.copy()
        self.frames += 1
        return self._step(action)


TASKS = ("train", "eval")


def task_reward_kwargs(task):
    """
    Reward parameters of a task. "train" reads src/train.py's constants, so the fitness is always the
    PPO training reward (undiscounted and unnormalised: gamma and VecNormalize belong to PPO, not to
    the task). "eval" is the evaluation task (m=1, no bonus).
    """
    if task == "train":
        from src.train import D as TRAIN_D, DAMAGE_PENALTY_MULT, WIN_BONUS
        return {"d": TRAIN_D, "damage_penalty_mult": DAMAGE_PENALTY_MULT, "win_bonus": WIN_BONUS}
    if task == "eval":
        return {"d": D, "damage_penalty_mult": 1.0, "win_bonus": 0.0}
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")


def task_description(task):
    """Human-readable reward of a task, written to config.json."""
    k = task_reward_kwargs(task)
    name = "PPO training reward" if task == "train" else "evaluation task"
    return f"{name} (d={k['d']}, m={k['damage_penalty_mult']}, win bonus={k['win_bonus']})"


def make_task_env(boss, task):
    """make_env of the boss with the reward of `task` ("train" or "eval"), frameskip 4."""
    from src.env import make_env
    cfg = BOSSES[boss]
    env = make_env(state=cfg["state"], frameskip=FRAMESKIP,
                   max_episode_frames=cfg["frames"], **task_reward_kwargs(task))
    counter = FrameCounter(env)
    return env, counter


def run_episode(env, counter, act_fn, sticky_p=0.0, rng=None, noop_max=0, on_obs=None):
    """
    Run one episode in `env` and return the per-episode metrics; "reward" is the undiscounted sum of the
    env's reward (the task the env was built with).

    act_fn(step_index, obs) -> np.ndarray of shape (3,); called once per agent step (step_index counts
    the agent's own steps, after the optional no-op starts). If noop_max > 0, a random number of
    no-op agent steps in [0, noop_max] is executed first (no-op starts). Frames are counted from the
    reset, no-op frames included. `obs` is the WarpFrame output (84, 84, 1) of the current step.
    on_obs(obs, is_reset), if given, sees every observation (reset and each step, no-ops included)
    so that a frame stack stays identical to VecFrameStack's.
    """
    from src.eval import is_win
    counter.p = sticky_p
    counter.rng = rng
    obs, info = env.reset()
    if on_obs is not None:
        on_obs(obs, True)
    prev_hp = int(info["hp"])
    hp0 = prev_hp
    prev_boss = int(info["boss_hp"])
    reward = 0.0
    dmg_events = 0
    hits = 0
    steps = 0                 # agent steps, no-op starts included (what the TimeLimit counts)
    terminated = truncated = False
    n_noop = int(rng.integers(0, noop_max + 1)) if noop_max > 0 else 0
    agent_i = 0
    boss_hp = prev_boss
    while not (terminated or truncated):
        if steps < n_noop:
            a = np.zeros(3, dtype=np.int64)
        else:
            a = act_fn(agent_i, obs)
            agent_i += 1
        obs, r, terminated, truncated, info = env.step(a)
        if on_obs is not None:
            on_obs(obs, False)
        steps += 1
        reward += float(r)
        hp = int(info["hp"])
        boss_hp = int(info["boss_hp"])
        dmg_events += int(hp < prev_hp)
        hits += int(boss_hp < prev_boss)
        prev_hp, prev_boss = hp, boss_hp
    win = is_win(boss_hp, prev_hp)        # boss dead with Mega Man alive, as in src/eval.py
    return {
        "reward": round(reward, 6),
        "win": bool(win),
        "flawless": bool(win and dmg_events == 0),
        "damage_events": dmg_events,
        "hp_lost": hp0 - prev_hp,
        "hits": hits,
        "died": bool(prev_hp <= 0),
        "steps": steps,
        "frames": counter.frames,
        "frames_to_kill": counter.frames if win else None,
        "noop_steps": n_noop,
    }


def fitness_key(m):
    """Sort key (larger is better): episode reward of the search task, then win, then fewer frames."""
    return (m["reward"], int(m["win"]), -(m["frames_to_kill"] or 0))


def decode_gene(g):
    """Gene in [0, 20) -> MultiDiscrete([2,5,2]) action [B, D-pad, A]."""
    g = int(g)
    return np.array([g // 10, (g // 2) % 5, g % 2], dtype=np.int64)


# ----------------------------------------------------------------------------------------------
# Worker pool: one persistent emulator per process, at most 12 processes
# ----------------------------------------------------------------------------------------------
MAX_PROCS = 12
_W = {}   # per-process state


def _worker_init(boss, task):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)          # 1 thread per process (12 procs must not oversubscribe the CPU)
    import cv2
    cv2.setNumThreads(1)
    _W["env"], _W["counter"] = make_task_env(boss, task)


def _worker_task(task):
    """task = (index, kind, payload). Returns (index, metrics)."""
    idx, kind, payload = task
    env, counter = _W["env"], _W["counter"]
    if kind == "ol":
        genome, k, sticky_p, seed, noop_max = payload
        genome = np.asarray(genome)
        rng = np.random.default_rng(seed)

        def act(i, obs):
            return decode_gene(genome[min(i // k, len(genome) - 1)])
        m = run_episode(env, counter, act, sticky_p, rng, noop_max)
    elif kind == "deep":
        from baselines.ga import deep_ga
        chain, sigma, sticky_p, seed, noop_max = payload
        m = deep_ga.run_chain_episode(_W, chain, sigma, sticky_p, np.random.default_rng(seed), noop_max)
    else:
        raise ValueError(kind)
    return idx, m


class EvalPool:
    """
    Persistent pool of `procs` workers (spawn), each owning one emulator of the given boss with the
    reward of `task`. A process can hold only one retro emulator.
    """
    def __init__(self, boss, procs=MAX_PROCS, task="eval"):
        if procs > MAX_PROCS:
            raise ValueError(f"at most {MAX_PROCS} emulator processes are allowed")
        task_reward_kwargs(task)      # fail fast on an unknown task
        self.procs = procs
        self.task = task
        self.pool = mp.get_context("spawn").Pool(procs, initializer=_worker_init, initargs=(boss, task))

    def map(self, kind, payloads):
        """Evaluate payloads in parallel (unordered scheduling, ordered result list)."""
        out = [None] * len(payloads)
        tasks = [(i, kind, p) for i, p in enumerate(payloads)]
        for i, m in self.pool.imap_unordered(_worker_task, tasks, chunksize=1):
            out[i] = m
        return out

    def close(self):
        self.pool.close()
        self.pool.join()
