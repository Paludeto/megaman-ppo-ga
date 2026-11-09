"""
Reproducibility helpers of src.train: LR schedule (horizon, clamp),
VecNormalize checkpoint paths/roundtrip and the run metadata. None of these needs the ROM.
"""

import argparse
import os
import sys

import gymnasium as gym
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src import train

LR = 2.5e-4


def sb3_progress(step, total):
    """progress_remaining exactly as SB3 computes it."""
    return 1.0 - float(step) / float(total)


def test_lr_at_step_linear_and_clamped():
    assert train.lr_at_step(0, LR, 1000) == LR
    assert train.lr_at_step(500, LR, 1000) == pytest.approx(LR / 2)
    assert train.lr_at_step(1000, LR, 1000) == 0.0
    # past the horizon: constant at the minimum (0), never negative
    assert train.lr_at_step(1500, LR, 1000) == 0.0
    assert train.lr_at_step(10**9, LR, 1000) == 0.0


def test_default_horizon_matches_progress_remaining_exactly():
    # horizon == timesteps on a fresh run is the previous behavior: progress_remaining * LR (bitwise)
    total = 40_000_000
    sched = train.linear_schedule(LR, total, total)
    for step in (0, 1, 16_384, 1_400_000, 20_000_000, 39_999_999, total):
        p = sb3_progress(step, total)
        assert sched(p) == p * LR


def test_horizon_is_independent_of_the_timestep_ceiling():
    # ceiling 1M but horizon 4M: at the ceiling the LR is still 3/4 of the initial one
    total, horizon = 1_000_000, 4_000_000
    sched = train.linear_schedule(LR, horizon, total)
    assert sched(sb3_progress(0, total)) == LR
    assert sched(sb3_progress(total, total)) == pytest.approx(LR * 0.75)
    # a horizon shorter than the ceiling clamps to 0 once passed
    sched = train.linear_schedule(LR, 500_000, total)
    assert sched(sb3_progress(250_000, total)) == pytest.approx(LR / 2)
    assert sched(sb3_progress(500_000, total)) == 0.0
    assert sched(sb3_progress(900_000, total)) == 0.0


@pytest.mark.parametrize("model_path,expected", [
    ("/x/ck/run_500000_steps.zip", "/x/ck/run_vecnormalize_500000_steps.pkl"),
    ("/x/ck/run_500000_steps", "/x/ck/run_vecnormalize_500000_steps.pkl"),
    ("/x/ck/Gutsman_v2_1600000_steps.zip", "/x/ck/Gutsman_v2_vecnormalize_1600000_steps.pkl"),
    ("/x/models/final_model.zip", "/x/models/final_model_vecnormalize.pkl"),
])
def test_vecnormalize_path_for(model_path, expected):
    assert train.vecnormalize_path_for(model_path) == expected


def test_vecnormalize_roundtrip(tmp_path):
    def wrapped():
        return VecNormalize(DummyVecEnv([lambda: gym.make("CartPole-v1")]), norm_obs=False, norm_reward=True, gamma=0.99)

    env = wrapped()
    env.reset()
    for _ in range(200):
        env.step([env.action_space.sample()])
    path = str(tmp_path / "final_model_vecnormalize.pkl")
    env.save(path)

    restored = VecNormalize.load(path, DummyVecEnv([lambda: gym.make("CartPole-v1")]))
    assert restored.ret_rms.count == env.ret_rms.count > 1
    assert restored.ret_rms.var == pytest.approx(env.ret_rms.var)
    assert restored.norm_reward and not restored.norm_obs


def test_run_meta_contents(tmp_path):
    args = argparse.Namespace(tag="t", seed=1, lr_horizon=None, deterministic_torch=False, timesteps=10)
    meta = train.collect_run_meta(args, "cpu", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for lib in ("python", "torch", "stable_baselines3", "stable_retro", "gymnasium", "numpy"):
        assert meta["versions"][lib]
    assert meta["args"]["seed"] == 1
    assert meta["recipe"]["LR"] == train.LR and "GAMMA" in meta["recipe"]
    assert set(meta["git"]) == {"commit", "dirty"}
    assert "gpu" in meta["hardware"]

    assert os.path.basename(train.write_run_meta(meta, str(tmp_path))) == "run_meta.json"
