"""
Tests for src/eval.py and the no-damage stop of src/train.py.

Split in two groups: pure logic and a synthetic env (no ROM needed) and emulator tests (skipped
when custom_integrations/MegaMan-v1-Nes/rom.nes is missing).
"""

import json
import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src import eval as ev
from src.train import ValidationCallback



# ----------------------------------------------------------------------------------------------
# Wilson interval, aggregation, outcome classification
# ----------------------------------------------------------------------------------------------
def test_wilson_reference_values():
    lo, hi = ev.wilson_interval(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-3)
    lo, hi = ev.wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = ev.wilson_interval(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)
    # 89/90 stochastic wins -> [94, 100]; 65/90 -> [62, 80]
    lo, hi = ev.wilson_interval(89, 90)
    assert (round(100 * lo), round(100 * hi)) == (94, 100)
    lo, hi = ev.wilson_interval(65, 90)
    assert (round(100 * lo), round(100 * hi)) == (62, 80)
    assert all(math.isnan(x) for x in ev.wilson_interval(0, 0))


def test_wilson_is_symmetric_and_contains_the_estimate():
    for n in (1, 5, 30, 90):
        for k in range(n + 1):
            lo, hi = ev.wilson_interval(k, n)
            lo2, hi2 = ev.wilson_interval(n - k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0
            assert (lo, hi) == pytest.approx((1 - hi2, 1 - lo2))


def episode(outcome, hp_lost=0, damage_events=None, frames_to_kill=None, reward=0.0, simultaneous=False):
    win = outcome == "win"
    return {"outcome": outcome, "win": win, "no_damage": win and hp_lost == 0, "hp_lost": hp_lost,
            "damage_events": hp_lost // 4 if damage_events is None else damage_events,
            "frames_to_kill": frames_to_kill, "reward": reward, "simultaneous_death": simultaneous}


def test_summarize_aggregates_synthetic_episodes():
    eps = [
        episode("win", 0, frames_to_kill=1000, reward=0.70),
        episode("win", 0, frames_to_kill=1200, reward=0.70),
        episode("win", 8, frames_to_kill=1400, reward=0.60),
        episode("death", 28, reward=0.10),
        episode("timeout", 4, reward=0.05),
        episode("death", 28, reward=0.20, simultaneous=True),
    ]
    agg = ev.summarize(eps)
    assert agg["n_episodes"] == 6
    assert agg["win"]["k"] == 3 and agg["win"]["rate"] == 0.5
    assert agg["no_damage"]["k"] == 2 and agg["no_damage"]["n"] == 6
    assert agg["no_damage"]["ci95"] == pytest.approx(list(ev.wilson_interval(2, 6)))
    assert agg["hp_lost"]["mean"] == pytest.approx(np.mean([0, 0, 8, 28, 4, 28]))
    assert agg["hp_lost"]["std"] == pytest.approx(np.std([0, 0, 8, 28, 4, 28], ddof=1))
    # frames_to_kill only exists for wins
    assert agg["frames_to_kill"] == {"mean": 1200.0, "std": pytest.approx(200.0), "n": 3}
    assert agg["reward"]["mean"] == pytest.approx(np.mean([0.70, 0.70, 0.60, 0.10, 0.05, 0.20]))
    assert (agg["deaths"], agg["timeouts"], agg["simultaneous_deaths"]) == (2, 1, 1)


def test_summarize_edge_cases():
    agg = ev.summarize([episode("timeout", 0, reward=0.0)])
    assert agg["win"]["k"] == 0 and agg["frames_to_kill"]["n"] == 0
    assert math.isnan(agg["frames_to_kill"]["mean"])
    assert agg["reward"]["std"] == 0.0  # a single episode has no spread
    assert ev.summarize([])["win"]["n"] == 0


def test_outcome_classification():
    assert ev.is_win(0, 28) and ev.is_no_damage(0, 28)
    assert ev.is_win(0, 20) and not ev.is_no_damage(0, 20)
    assert not ev.is_win(0, 0) and not ev.is_no_damage(0, 0)   # simultaneous death is not a win
    assert not ev.is_win(6, 28) and not ev.is_no_damage(6, 28)  # boss alive
    assert ev.is_no_damage(0, 20, initial_hp=20)                # relative to the initial HP


# ----------------------------------------------------------------------------------------------
# Sticky actions per frame (stub env, no emulator)
# ----------------------------------------------------------------------------------------------
class RecordingEnv(gym.Env):
    """Records the actions it receives."""
    observation_space = spaces.Box(0, 1, (1,), dtype=np.float32)
    action_space = spaces.MultiDiscrete([2, 5, 2])

    def __init__(self):
        self.actions = []

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.array(action).copy())
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class ScriptedRng:
    """rng.random() replays a fixed list."""
    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0)


ACTIONS = [np.array(a) for a in ([1, 0, 0], [0, 1, 1], [1, 2, 0], [0, 4, 1])]


def test_sticky_p0_is_a_pass_through_and_never_draws():
    base = RecordingEnv()
    env = ev.StickyFrames(base, 0.0, ScriptedRng([]))  # an empty script raises if a number is drawn
    env.reset()
    for a in ACTIONS:
        env.step(a)
    assert [a.tolist() for a in base.actions] == [a.tolist() for a in ACTIONS]


def test_sticky_p1_repeats_the_previous_action_from_the_start():
    base = RecordingEnv()
    env = ev.StickyFrames(base, 1.0, np.random.default_rng(0))
    env.reset()
    for a in ACTIONS:
        env.step(a)
    # the "previous action" before the first frame is no-op, and it is then repeated forever
    assert all(a.tolist() == [0, 0, 0] for a in base.actions)


def test_sticky_repeats_the_executed_previous_action_and_reset_clears_it():
    p = 0.5
    base = RecordingEnv()
    # draws: keep, stick, stick, keep  (stick when draw < p)
    env = ev.StickyFrames(base, p, ScriptedRng([0.9, 0.1, 0.1, 0.9, 0.1]))
    env.reset()
    for a in ACTIONS:
        env.step(a)
    # frame 2 and 3 repeat ACTIONS[0] (the executed previous action, not ACTIONS[1]); frame 4 is new
    assert [a.tolist() for a in base.actions] == [ACTIONS[0].tolist()] * 3 + [ACTIONS[3].tolist()]
    env.reset()
    env.step(ACTIONS[1])  # draw 0.1 < p right after a reset: previous action is no-op again
    assert base.actions[-1].tolist() == [0, 0, 0]


def test_sticky_frequency():
    base = RecordingEnv()
    env = ev.StickyFrames(base, 0.25, np.random.default_rng(1))
    env.reset()
    requests = [np.array([i, 0, 0]) for i in range(4000)]  # all distinct: a sticky frame is always visible
    for a in requests:
        env.step(a)
    stuck = np.mean([not np.array_equal(e, a) for e, a in zip(base.actions, requests)])
    assert 0.25 * 0.9 < stuck < 0.25 * 1.1


# ----------------------------------------------------------------------------------------------
# Emulator: the eval stack, frame counting, no-op starts, reproducibility
# ----------------------------------------------------------------------------------------------
class ScriptedModel:
    """Stand-in for a PPO model: samples random actions with torch (seeded per episode by run_episode)."""
    def predict(self, obs, deterministic=False):
        a = [int(torch.randint(0, n, (1,))) for n in (2, 5, 2)]
        return np.array([a]), None


class ZeroModel:
    def predict(self, obs, deterministic=False):
        return np.zeros((1, 3), dtype=np.int64), None


@pytest.fixture
def eval_env(rom_available):
    made = []

    def _make(*args, **kwargs):
        while made:
            made.pop().venv.close()
        made.append(ev.make_eval_env(*args, **kwargs))
        return made[-1]

    yield _make
    while made:
        made.pop().venv.close()


def reference_rollout(actions, **make_kwargs):
    """rollout() on src.env.make_env's stack; the emulator is closed before returning (one per process)."""
    from src.env import make_env
    env = make_env(**make_kwargs)
    try:
        return rollout(env, actions)
    finally:
        env.close()


def rollout(env, actions):
    """Step a gym env through `actions`; return (obs stack, rewards, infos)."""
    env.reset()
    obs_list, rewards, infos = [], [], []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        obs_list.append(obs)
        rewards.append(r)
        infos.append(info)
        if term or trunc:
            break
    return np.array(obs_list), rewards, infos


def test_eval_stack_matches_make_env_without_sticky(eval_env):
    rng = np.random.default_rng(0)
    actions = [rng.integers([0, 0, 0], [2, 5, 2]) for _ in range(120)]
    ref_obs, ref_r, ref_info = reference_rollout(actions, state="Gutsman-boss")
    ee = eval_env("Gutsman-boss", sticky=0.0)
    obs, r, info = rollout(ee.env, actions)
    assert np.array_equal(obs, ref_obs)
    assert r == ref_r
    assert [(i["hp"], i["boss_hp"]) for i in info] == [(i["hp"], i["boss_hp"]) for i in ref_info]


def test_eval_stack_with_sticky_p1_equals_noop_actions(eval_env):
    rng = np.random.default_rng(0)
    actions = [rng.integers([0, 0, 0], [2, 5, 2]) for _ in range(60)]
    ref_obs, ref_r, _ = reference_rollout([np.zeros(3, dtype=int)] * len(actions), state="Gutsman-boss")
    ee = eval_env("Gutsman-boss", sticky=1.0)
    obs, r, _ = rollout(ee.env, actions)
    assert np.array_equal(obs, ref_obs) and r == ref_r


def test_frame_counter_counts_real_frames(eval_env):
    ee = eval_env("Gutsman-boss")
    ee.env.reset()
    for _ in range(10):
        ee.env.step(np.zeros(3, dtype=int))
    assert ee.counter.frames >= 40    # never fewer than 4 frames per step (longer while blinking)


def test_run_episode_timeout_frames_and_noop_start(eval_env):
    # 80 frames per episode = 20 steps; no boss kill is possible, no damage in idle
    ee = eval_env("Gutsman-boss", episode_frames=80)
    noop_rng = np.random.default_rng(5)
    expected_noop = int(np.random.default_rng(5).integers(0, 11))
    r = ev.run_episode(ZeroModel(), ee, True, noop_max=10, noop_rng=noop_rng,
                       sticky_rng=np.random.default_rng(0))
    assert r["noop_steps"] == expected_noop
    assert r["steps"] == 20 and r["frames"] == 80          # counted from the emulator, incl. the no-op start
    assert r["outcome"] == "timeout" and not r["win"] and r["frames_to_kill"] is None
    assert r["hp_initial"] == 28 and r["hp_lost"] == 28 - r["hp_final"]
    # no-op starts of 0: the policy acts from the first step
    r0 = ev.run_episode(ZeroModel(), ee, True, noop_max=0, noop_rng=np.random.default_rng(5),
                        sticky_rng=np.random.default_rng(0))
    assert r0["noop_steps"] == 0


def test_evaluate_is_reproducible_and_reports_all_metrics(rom_available):
    kwargs = dict(episodes=3, deterministic=False, sticky=0.25, noop_max=5, episode_frames=200,
                  seed=7, verbose=False)
    a = ev.evaluate(ScriptedModel(), "Gutsman-boss", **kwargs)
    b = ev.evaluate(ScriptedModel(), "Gutsman-boss", **kwargs)
    assert a["episodes"] == b["episodes"]
    assert json.dumps(ev.json_safe(a), allow_nan=False) == json.dumps(ev.json_safe(b), allow_nan=False)
    assert len(a["episodes"]) == 3 and a["aggregate"]["n_episodes"] == 3
    for e in a["episodes"]:
        for key in ("win", "no_damage", "hp_lost", "damage_events", "frames", "frames_to_kill", "reward"):
            assert key in e
        assert e["steps"] <= 50 and e["hp_lost"] >= 0 and e["damage_events"] >= 0
    assert a["config"]["sticky"] == 0.25 and a["config"]["noop_max"] == 5


# ----------------------------------------------------------------------------------------------
# No-damage stop criterion : synthetic episodes through a real PPO + ValidationCallback
# ----------------------------------------------------------------------------------------------
class FakeBossEnv(gym.Env):
    """Ends after 3 steps with the given final (boss_hp, hp); info carries hp/boss_hp like BossWrapper."""
    observation_space = spaces.Box(0, 1, (4,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self, boss_hp, hp, final_reward=0.0):
        self.boss_hp, self.hp, self.final_reward = boss_hp, hp, final_reward
        self.t = 0

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t >= 3
        info = {"hp": self.hp if done else 28, "boss_hp": self.boss_hp if done else 28}
        return np.zeros(4, dtype=np.float32), (self.final_reward if done else 0.0), done, False, info


def fake_venv(**kwargs):
    return VecMonitor(DummyVecEnv([lambda: FakeBossEnv(**kwargs)]), info_keywords=("hp", "boss_hp"))


def train_with_callback(boss_hp, hp, final_reward=0.0):
    """Run PPO with the ValidationCallback; return (num_timesteps at the end, callback)."""
    env, eval_env = fake_venv(boss_hp=boss_hp, hp=hp, final_reward=final_reward), \
        fake_venv(boss_hp=boss_hp, hp=hp, final_reward=final_reward)
    cb = ValidationCallback(eval_env, eval_freq=16, n_eval_episodes=2, stop_on_no_damage=True, verbose=0)
    model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, n_epochs=1, device="cpu", seed=0)
    model.learn(total_timesteps=128, callback=cb)
    return model.num_timesteps, cb


@pytest.mark.parametrize("boss_hp,hp,stops,label", [
    (0, 28, True, "win without damage"),
    (0, 20, False, "win with damage"),
    (12, 0, False, "defeat"),
    (0, 0, False, "simultaneous death"),
    (12, 28, False, "timeout with the boss alive"),
])
def test_info_stop_criterion(boss_hp, hp, stops, label):
    steps, cb = train_with_callback(boss_hp, hp)
    assert (steps < 128) == stops, label
    assert len(cb.last_outcomes) == 2  # every validation episode was classified from info["episode"]
    assert all(o["no_damage"] for o in cb.last_outcomes) == stops
    assert all(o["win"] for o in cb.last_outcomes) == (boss_hp == 0 and hp > 0)
