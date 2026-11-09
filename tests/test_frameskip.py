"""Tests for FrameskipWrapper.

Wrapper-logic tests use a fake env (no ROM needed). Emulator tests need the ROM and use the
`require_rom` fixture from conftest.py.
"""

import hashlib
import json
import os

import gymnasium as gym
import numpy as np
import pytest

from src.wrappers import FrameskipWrapper

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class FrameLog(gym.Wrapper):
    """Sits below FrameskipWrapper: counts emulator frames and keeps the raw RGB frames."""

    def __init__(self, env):
        super().__init__(env)
        self.n = 0
        self.frames = []

    def reset(self, **kwargs):
        self.n = 0
        self.frames = []
        return self.env.reset(**kwargs)

    def step(self, action):
        self.n += 1
        result = self.env.step(action)
        self.frames.append(result[0].copy())
        return result


class FakeEnv(gym.Env):
    """Emits random RGB frames; `blink_counter` follows a scripted sequence (1 -> invisible)."""

    def __init__(self, blink_seq=None, terminate_at=None):
        self.observation_space = gym.spaces.Box(0, 255, (8, 8, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self._rng = np.random.default_rng(1)
        self._blink_seq = blink_seq or []
        self._terminate_at = terminate_at
        self.t = 0

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return self._rng.integers(0, 256, (8, 8, 3), dtype=np.uint8), {"blink_counter": 0}

    def step(self, action):
        self.t += 1
        obs = self._rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        blink = self._blink_seq[self.t - 1] if self.t - 1 < len(self._blink_seq) else 0
        term = self._terminate_at is not None and self.t >= self._terminate_at
        return obs, 1.0, term, False, {"blink_counter": blink}


def _wrapped(**fake_kwargs):
    log = FrameLog(FakeEnv(**fake_kwargs))
    return FrameskipWrapper(log, skip=4), log


def test_extends_while_blinking():
    env, log = _wrapped(blink_seq=[1] * 200)
    env.reset()
    env.step(0)
    assert log.n == 4 + 60  # 4 fixed + safety cap of 60 extra frames


# ------------------------------------------------------------------ emulator tests (ROM needed)

def _find_frameskip(env):
    e = env
    while not isinstance(e, FrameskipWrapper):
        e = e.env
    return e


def _make_logged_env(state="Fireman-boss", **kwargs):
    """make_env with a FrameLog inserted right below the FrameskipWrapper."""
    from src.env import make_env

    env = make_env(state=state, **kwargs)
    fs = _find_frameskip(env)
    log = FrameLog(fs.env)
    fs.env = log
    return env, log


def _random_action(rng):
    return np.array([rng.integers(2), rng.integers(5), rng.integers(2)])


def record_trajectory(steps=1500, **make_env_kwargs):
    """Per-step [reward, terminated, truncated, info, obs sha256, frames] on Fireman, seed 0."""
    env, log = _make_logged_env(**make_env_kwargs)
    rng = np.random.default_rng(0)
    env.reset()
    out = []
    for _ in range(steps):
        log.n = 0
        obs, r, term, trunc, info = env.step(_random_action(rng))
        out.append([float(r), bool(term), bool(trunc),
                    {k: (v.item() if hasattr(v, "item") else v) for k, v in sorted(info.items())},
                    hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest(), log.n])
        if term or trunc:
            env.reset()
    env.close()
    return out


# Info keys recorded in the fixture that were removed from the wrapper's info afterwards
# (`eye_state`). Every other field of the fixture is still compared exactly.
REMOVED_INFO_KEYS = {"eye_state"}


def _load_expected():
    with open(os.path.join(DATA_DIR, "fireman_legacy_trajectory.json")) as f:
        expected = json.load(f)
    for row in expected:
        row[3] = {k: v for k, v in row[3].items() if k not in REMOVED_INFO_KEYS}
    return expected


@pytest.mark.usefixtures("require_rom")
def test_reproduces_recorded_trajectory():
    # Pinned trajectory: the emulator, the wrappers and the blink extension must not change.
    expected = _load_expected()
    got = record_trajectory()
    assert len(got) == len(expected) == 1500
    for i, (g, e) in enumerate(zip(got, expected)):
        assert g == e, f"trajectory diverged at step {i}: {g} != {e}"
    # the fixture exercises the blink extension (some steps > 4 frames)
    assert max(x[5] for x in got) > 4
