"""
Module: src.eval
Description: Test-time evaluation of a saved model (.zip) against a boss. Runs N episodes on the
             real task (no reward shaping: 1x damage penalty, no win bonus) under an
             optional perturbation protocol and reports, per episode and aggregated:
               - win (boss_hp == 0 with Mega Man alive) and no-damage win (win and HP lost == 0);
               - HP lost and number of damage events;
               - emulator frames until the kill (real frames, not agent steps x 4);
               - the episode reward.
             Proportions come with a 95% Wilson interval, everything else as mean +/- std.

Protocol:
  --sticky P      sticky actions PER FRAME (Machado et al. 2018): at every emulator frame the
                  previous frame's action is repeated with probability P. It sits BELOW the
                  ActionSkipWrapper, so the B-toggle and the frameskip see the perturbed actions.
  --noop-max N    no-op starts: 0..N agent steps of "no button" after the reset (N=30 in the campaign).
  --deterministic / --stochastic   greedy or sampled policy (default: deterministic).
Every episode is seeded independently from --seed (sticky RNG, no-op RNG and torch), so a run
is reproducible and episode i does not depend on the episodes before it.

Metric definitions:
  - "win" needs boss_hp == 0 AND hp > 0: a simultaneous death is reported as a death (and counted in
    `simultaneous_deaths`).
  - "no damage" = win with final HP == initial HP (initial HP is read from the RAM at the reset).
  - damage events are counted per emulator frame from the RAM (a hit is one drop of `health`),
    not per agent step.
  - frames_to_kill counts real emulator frames from the first policy action (the no-op start is
    excluded), and only exists for wins. It differs from steps x 4 because the frameskip
    stretches steps while Mega Man blinks after a hit.
  - the reward is an event indicator (+d per hit on the boss, -d per hit taken): it ignores the
    HP magnitude (4 to 10 HP per hit, depending on the enemy), so it is not comparable across
    bosses. Use hp_lost / damage_events / frames_to_kill for comparisons.

Known limitation: the episode ends at the FIRST boss_hp == 0 (BossWrapper), and that
is what is counted as a win. That is exact for the six Robot Master fights, but a
multi-phase boss (e.g. the Wily fights), whose HP bar reaches 0 between phases, would produce a
false win. Guarding it needs a "zero for K stable frames" check inside BossWrapper.step
(src/wrappers.py); it is not implemented here, so do not use this tool on multi-phase bosses
without adding that guard.

Usage:
  python src/eval.py --model models/<tag>_best/best_model.zip --state Gutsman-boss
  python src/eval.py --model models/gutsman_best/best_model.zip --state Gutsman-boss \
      --episodes 30 --deterministic --sticky 0.25 --noop-max 30 --seed 123 --json out.json
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym
import stable_retro
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

try:
    from src import env as env_module
    from src.wrappers import ActionSkipWrapper, FrameskipWrapper, BossWrapper, StickyFrames, WarpFrame
except ImportError:
    import env as env_module
    from wrappers import ActionSkipWrapper, FrameskipWrapper, BossWrapper, StickyFrames, WarpFrame

# Must match the training recipe (src/train.py).
D = 0.05
FRAMESKIP = 4
INITIAL_HP = 28   # Mega Man's HP at the start of every save state
Z95 = 1.959963984540054
NOOP = np.zeros((1, 3), dtype=np.int64)   # MultiDiscrete([2,5,2]): no B, no D-pad, no A


# ----------------------------------------------------------------------------------------------
# Outcome classification and statistics (pure functions; no emulator needed)
# ----------------------------------------------------------------------------------------------
def is_win(boss_hp, hp) -> bool:
    """Boss dead with Mega Man alive (a simultaneous death is NOT a win)."""
    return int(boss_hp) == 0 and int(hp) > 0


def is_no_damage(boss_hp, hp, initial_hp: int = INITIAL_HP) -> bool:
    """Win that ends with the initial HP intact (no damage taken in the whole episode)."""
    return is_win(boss_hp, hp) and int(hp) == int(initial_hp)


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval (default 95%) for a proportion k/n. Returns (nan, nan) if n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # exact endpoints: the interval touches 0 at k == 0 and 1 at k == n (avoids float residue)
    return (0.0 if k == 0 else max(0.0, centre - half), 1.0 if k == n else min(1.0, centre + half))


def _proportion(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan"), "ci95": [lo, hi]}


def _mean_std(values) -> dict:
    """Mean and sample standard deviation (ddof=1; 0 for a single value; nan when empty)."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}


def summarize(episodes: list[dict]) -> dict:
    """Aggregate per-episode results: Wilson intervals for win / no-damage, mean +/- std otherwise."""
    n = len(episodes)
    return {
        "n_episodes": n,
        "win": _proportion(sum(e["win"] for e in episodes), n),
        "no_damage": _proportion(sum(e["no_damage"] for e in episodes), n),
        "hp_lost": _mean_std(e["hp_lost"] for e in episodes),
        "damage_events": _mean_std(e["damage_events"] for e in episodes),
        "frames_to_kill": _mean_std(e["frames_to_kill"] for e in episodes if e["frames_to_kill"] is not None),
        "reward": _mean_std(e["reward"] for e in episodes),
        "deaths": sum(e["outcome"] == "death" for e in episodes),
        "timeouts": sum(e["outcome"] == "timeout" for e in episodes),
        "simultaneous_deaths": sum(e["simultaneous_death"] for e in episodes),
    }


# ----------------------------------------------------------------------------------------------
# Environment: the same wrapper stack as src.env.make_env, plus sticky actions and frame counting
# (StickyFrames lives in src/wrappers.py, shared with the sticky training env of src/env.py)
# ----------------------------------------------------------------------------------------------
class FrameCounter(gym.Wrapper):
    """
    Counts the real emulator frames of the episode and the damage events (drops of the `health`
    RAM value, frame by frame). Sits on the raw stable-retro env, so every frame is seen exactly
    once no matter how the wrappers above stretch the agent step. On reset it stashes the totals
    of the episode that just ended in `last_*` (a VecEnv resets by itself when an episode ends).
    """

    def __init__(self, env):
        super().__init__(env)
        self.frames = 0
        self.damage_events = 0
        self.last_frames = 0
        self.last_damage_events = 0
        self._prev_health = INITIAL_HP

    def reset(self, **kwargs):
        self.last_frames, self.last_damage_events = self.frames, self.damage_events
        self.frames = 0
        self.damage_events = 0
        out = self.env.reset(**kwargs)
        self._prev_health = int(self.unwrapped.data.lookup_value("health"))
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames += 1
        health = info.get("health", self._prev_health)
        if health < self._prev_health:
            self.damage_events += 1
        self._prev_health = health
        return obs, reward, terminated, truncated, info

    def ram(self, name: str) -> int:
        return int(self.unwrapped.data.lookup_value(name))


@dataclass
class EvalEnv:
    """A single-env stack for evaluation plus the handles the episode loop needs."""
    venv: Any
    env: Any   # the wrapped gym env inside `venv` (same stack as make_env's output)
    counter: FrameCounter
    sticky: StickyFrames


def make_eval_env(state: str, sticky: float = 0.0, episode_frames: int = 7200,
                  game: str = "MegaMan-v1-Nes") -> EvalEnv:
    """
    Builds the evaluation stack. It mirrors src.env.make_env for the real task (d=0.05, 1x
    damage penalty, no win bonus) with the same classes and order, adding only
    FrameCounter and StickyFrames right above the emulator, i.e. BELOW ActionSkipWrapper. With
    sticky == 0 the observations and rewards are identical to make_env (checked in tests/test_eval.py).
    """
    env_module._register_custom_integrations()
    env_module._verify_rom(game)
    raw = stable_retro.make(
        game=game, state=state, inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=stable_retro.Actions.MULTI_DISCRETE,
        obs_type=stable_retro.Observations.IMAGE, render_mode=None,
    )
    sticky_env = StickyFrames(raw, sticky, np.random.default_rng(0))
    counter = FrameCounter(sticky_env)
    env = ActionSkipWrapper(counter)
    env = FrameskipWrapper(env, skip=FRAMESKIP)
    env = TimeLimit(env, max_episode_steps=episode_frames // FRAMESKIP)
    env = BossWrapper(env, d=D)
    env = WarpFrame(env)
    venv = VecTransposeImage(VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4, channels_order="last"))
    return EvalEnv(venv=venv, env=env, counter=counter, sticky=sticky_env)


# ----------------------------------------------------------------------------------------------
# Episode loop
# ----------------------------------------------------------------------------------------------
def run_episode(model, ee: EvalEnv, deterministic: bool, noop_max: int,
                noop_rng: np.random.Generator, sticky_rng: np.random.Generator,
                torch_seed: int | None = None) -> dict:
    """Run one episode and return its metrics (see the module docstring for the definitions)."""
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    ee.sticky.rng = sticky_rng
    venv, counter = ee.venv, ee.counter
    obs = venv.reset()
    if hasattr(model, "reset_episode"):
        model.reset_episode()  # non-SB3 policies with per-episode state (the open-loop GA baseline)
    initial_hp = counter.ram("health")
    noop_steps = int(noop_rng.integers(0, noop_max + 1)) if noop_max > 0 else 0

    total_reward, steps, noop_frames, done = 0.0, 0, 0, False
    infos = [{}]
    while not done:
        if steps < noop_steps:
            action = NOOP
        else:
            if steps == noop_steps:
                noop_frames = counter.frames  # frames spent before the first policy action
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, infos = venv.step(action)
        total_reward += float(reward[0])
        steps += 1
        done = bool(dones[0])  # TimeLimit guarantees termination

    info = infos[0]
    boss_hp, hp = int(info["boss_hp"]), int(info["hp"])
    frames, damage_events = counter.last_frames, counter.last_damage_events
    win = is_win(boss_hp, hp)
    outcome = "win" if win else ("death" if hp <= 0 or boss_hp == 0 else "timeout")
    return {
        "outcome": outcome,
        "win": win,
        "no_damage": is_no_damage(boss_hp, hp, initial_hp),
        "boss_hp_final": boss_hp,
        "hp_initial": initial_hp,
        "hp_final": hp,
        "hp_lost": initial_hp - hp,
        "damage_events": damage_events,
        "frames": frames,
        "frames_to_kill": (frames - noop_frames) if win else None,
        "noop_steps": noop_steps,
        "steps": steps,
        "reward": total_reward,
        "simultaneous_death": boss_hp == 0 and hp <= 0,
    }


def evaluate(model, state: str, episodes: int = 30, deterministic: bool = True, sticky: float = 0.0,
             noop_max: int = 0, episode_frames: int = 7200,
             seed: int = 0, verbose: bool = True) -> dict:
    """Evaluate `model` for `episodes` episodes. Returns {"config": ..., "aggregate": ..., "episodes": [...]}."""
    ee = make_eval_env(state, sticky=sticky, episode_frames=episode_frames)
    results = []
    try:
        # One independent stream per episode: episode i is reproducible without the previous ones.
        children = np.random.SeedSequence(seed).spawn(episodes)
        for i, child in enumerate(children):
            noop_rng, sticky_rng = (np.random.default_rng(s) for s in child.spawn(2))
            r = run_episode(model, ee, deterministic, noop_max, noop_rng, sticky_rng,
                            torch_seed=int(child.generate_state(1)[0]))
            r["episode"] = i
            results.append(r)
            if verbose:
                ftk = "-" if r["frames_to_kill"] is None else str(r["frames_to_kill"])
                print(f"  ep {i + 1:>3}: {r['outcome']:<7} hp_lost={r['hp_lost']:>2} "
                      f"dmg_events={r['damage_events']} frames_to_kill={ftk:>5} "
                      f"frames={r['frames']:>5} reward={r['reward']:.3f}")
    finally:
        ee.venv.close()
    return {
        "config": {"state": state, "episodes": episodes, "deterministic": deterministic,
                   "sticky": sticky, "noop_max": noop_max, "episode_frames": episode_frames,
                   "seed": seed, "d": D, "frameskip": FRAMESKIP},
        "aggregate": summarize(results),
        "episodes": results,
    }


def json_safe(obj):
    """Recursively replace NaN/inf by None so the output is strict JSON."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def format_summary(agg: dict) -> str:
    """Human-readable aggregate."""
    def prop(name, p):
        lo, hi = p["ci95"]
        return f"{name:<15}{p['k']}/{p['n']} = {100 * p['rate']:.0f}%  (95% Wilson [{100 * lo:.0f}, {100 * hi:.0f}])"

    def ms(name, m, fmt="{:.2f}"):
        if m["n"] == 0:
            return f"{name:<15}n/a  (n=0)"
        return f"{name:<15}{fmt.format(m['mean'])} +/- {fmt.format(m['std'])}  (n={m['n']})"

    return "\n".join([
        prop("Win", agg["win"]),
        prop("No damage", agg["no_damage"]),
        ms("HP lost", agg["hp_lost"]),
        ms("Damage events", agg["damage_events"]),
        ms("Frames to kill", agg["frames_to_kill"], "{:.0f}"),
        ms("Reward", agg["reward"], "{:.3f}"),
        f"{'Deaths':<15}{agg['deaths']}   timeouts: {agg['timeouts']}   simultaneous deaths: {agg['simultaneous_deaths']}",
    ])


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model against a boss (win, no-damage, HP lost, "
                                                 "frames to kill; Wilson 95% intervals).")
    parser.add_argument('--model', type=str, required=True, help='Model path (.zip)')
    parser.add_argument('--state', type=str, required=True, help='Fight save state (e.g. Bombman-boss)')
    parser.add_argument('--episodes', type=int, default=30, help='Number of episodes')
    parser.add_argument('--episode-frames', type=int, default=7200,
                        help='Frames per episode (must match training)')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the sticky/no-op RNGs and the policy sampling')
    parser.add_argument('--sticky', type=float, default=0.0,
                        help='Per-frame sticky-action probability (Machado et al. 2018; 0.25 in the campaign)')
    parser.add_argument('--noop-max', type=int, default=0, help='Random no-op start of 0..N agent steps (30 in the campaign)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--deterministic', dest='deterministic', action='store_true', help='Greedy policy (default)')
    mode.add_argument('--stochastic', dest='deterministic', action='store_false', help='Sample actions from the policy')
    parser.set_defaults(deterministic=True)
    parser.add_argument('--json', type=str, default=None, help='Write config, aggregate and per-episode metrics to this JSON file')
    parser.add_argument('--device', type=str, default='cpu', help='Torch device for the policy (cpu keeps the GPU free for training)')
    args = parser.parse_args()

    path = args.model
    if not os.path.exists(path) and os.path.exists(path + '.zip'):
        path = path + '.zip'
    if not os.path.exists(path):
        raise SystemExit(f"Model not found: {path}")
    if not 0.0 <= args.sticky <= 1.0:
        raise SystemExit("--sticky must be in [0, 1]")

    from stable_baselines3 import PPO
    if args.device == 'cpu':
        # Batch-1 inference on a small CNN gains nothing from intra-op threads, and they slow
        # everything down when several evaluations (or a training run) share the machine.
        torch.set_num_threads(1)
    # Inference only: skip unpickling the training LR schedule (it may reference src.train).
    model = PPO.load(path, device=args.device, custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})

    mode_name = "deterministic" if args.deterministic else "stochastic"
    print(f"Evaluating {os.path.basename(path)} on {args.state}: {args.episodes} episodes ({mode_name}, "
          f"sticky={args.sticky}, noop_max={args.noop_max}, seed={args.seed})")
    result = evaluate(model, args.state, episodes=args.episodes, deterministic=args.deterministic,
                      sticky=args.sticky, noop_max=args.noop_max, episode_frames=args.episode_frames,
                      seed=args.seed)
    result["config"]["model"] = path
    print("-" * 72)
    print(format_summary(result["aggregate"]))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(json_safe(result), f, indent=2, allow_nan=False)
        print(f"JSON written to {args.json}")


if __name__ == '__main__':
    main()
