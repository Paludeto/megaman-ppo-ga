"""Tests for the GA baselines. Emulator tests are skipped when rom.nes is absent."""
import collections
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baselines.ga import common, deep_ga, ga_ol  # noqa: E402

ROM = os.path.join(ROOT, "custom_integrations", "MegaMan-v1-Nes", "rom.nes")
needs_rom = pytest.mark.skipif(not os.path.exists(ROM), reason="rom.nes not present")


def test_decode_gene_covers_the_action_space():
    acts = {tuple(common.decode_gene(g)) for g in range(common.N_GENES)}
    assert len(acts) == 20
    assert all(0 <= a[0] < 2 and 0 <= a[1] < 5 and 0 <= a[2] < 2 for a in acts)


def test_next_population_elitism_and_sizes():
    rng = np.random.default_rng(0)
    pop = rng.integers(0, 20, size=(20, 30))
    keys = [(float(i), 0, 0) for i in range(20)]
    children, elites = ga_ol.next_population(pop, keys, rng, elite=3, trunc=0.25, p_cross=0.5, p_mut=0.05)
    assert children.shape == pop.shape
    assert elites == [19, 18, 17]
    assert (children[:3] == pop[[19, 18, 17]]).all()
    assert children.min() >= 0 and children.max() < 20


def test_seed_chain_rebuild_is_deterministic_and_prefix_cache_is_exact():
    model = deep_ga.build_model()
    chain = [123, 456, 789, 1011]
    a = deep_ga.rebuild(chain, 0.005, model)
    b = deep_ga.rebuild(chain, 0.005, model)
    assert np.array_equal(a, b)
    cache = collections.OrderedDict()
    deep_ga.rebuild(chain[:2], 0.005, model, cache)
    c = deep_ga.rebuild(chain, 0.005, model, cache)      # starts from the cached 2-seed prefix
    assert np.array_equal(a, c)
    # the mutation really is theta_{g-1} + sigma * eps(seed_g)
    d = deep_ga.rebuild(chain[:3], 0.005, model)
    assert np.allclose(a - d, 0.005 * deep_ga.noise(1011, a.shape[0]))
    assert not np.array_equal(a, d)


def test_deep_ga_next_generation_appends_one_seed():
    rng = np.random.default_rng(0)
    chains = [[i] for i in range(10)]
    keys = [(float(i), 0, 0) for i in range(10)]
    kids, elites = deep_ga.next_generation(chains, keys, rng, n_parents=3, elite=1)
    assert kids[0] == [9] and elites == [9]
    assert all(len(k) == 2 and k[0] in (9, 8, 7) for k in kids[1:])


@needs_rom
def test_fitness_env_is_the_evaluation_task_and_deterministic():
    env, counter = common.make_task_env("gutsman", "eval")
    try:
        genome = np.random.default_rng(0).integers(0, 20, size=450)

        def act(i, obs):
            return common.decode_gene(genome[i // 4])
        m1 = common.run_episode(env, counter, act)
        m2 = common.run_episode(env, counter, act)
        assert m1 == m2                                    # one rollout per individual is exact
        assert m1["steps"] <= 1800 and m1["frames"] >= 4 * m1["steps"]
        # evaluation reward = d * (hit events - damage events), no bonus
        assert abs(m1["reward"] - 0.05 * (m1["hits"] - m1["damage_events"])) < 1e-6
    finally:
        env.close()


def test_train_task_is_the_ppo_training_reward():
    from src import train
    k = common.task_reward_kwargs("train")
    assert k == {"d": train.D, "damage_penalty_mult": train.DAMAGE_PENALTY_MULT, "win_bonus": train.WIN_BONUS}
    assert common.task_reward_kwargs("eval") == {"d": 0.05, "damage_penalty_mult": 1.0, "win_bonus": 0.0}
    with pytest.raises(ValueError):
        common.task_reward_kwargs("shaped")


@needs_rom
def test_train_task_plays_the_same_trajectory_with_the_training_reward():
    """Same genome, both tasks: identical trajectory; only the reward differs (m=2 and win bonus)."""
    genome = np.random.default_rng(0).integers(0, 20, size=450)

    def act(i, obs):
        return common.decode_gene(genome[i // 4])
    out = {}
    for task in common.TASKS:
        env, counter = common.make_task_env("gutsman", task)
        try:
            out[task] = common.run_episode(env, counter, act)
        finally:
            env.close()
    tr, ev = out["train"], out["eval"]
    assert {k: v for k, v in tr.items() if k != "reward"} == {k: v for k, v in ev.items() if k != "reward"}
    assert tr["damage_events"] > 0                         # the m=2 term is exercised
    expected = 0.05 * tr["hits"] - 0.10 * tr["damage_events"] + 0.5 * tr["win"]
    assert abs(tr["reward"] - expected) < 1e-6


@needs_rom
def test_sticky_patch_matches_reference_stack():
    """The patched stack must equal the reference stack (StickyFrames below ActionSkip/Frameskip)."""
    import gymnasium as gym
    import stable_retro
    from gymnasium.wrappers import TimeLimit
    import src.env as E
    from src.wrappers import ActionSkipWrapper, FrameskipWrapper, BossWrapper

    class StickyFrames(gym.Wrapper):
        def __init__(s, env, p, rng):
            super().__init__(env)
            s.p, s.rng, s.prev = p, rng, np.zeros(3, dtype=int)

        def reset(s, **kw):
            s.prev = np.zeros(3, dtype=int)
            return s.env.reset(**kw)

        def step(s, a):
            a = np.array(a)
            if s.rng.random() < s.p:
                a = s.prev.copy()
            s.prev = a.copy()
            return s.env.step(a)

    E._register_custom_integrations()
    raw = stable_retro.make(game="MegaMan-v1-Nes", state="Gutsman-boss",
                            inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
                            use_restricted_actions=stable_retro.Actions.MULTI_DISCRETE,
                            obs_type=stable_retro.Observations.IMAGE, render_mode=None)
    ref = StickyFrames(raw, 0.25, np.random.default_rng(7))
    ref = BossWrapper(TimeLimit(FrameskipWrapper(ActionSkipWrapper(ref), 4), 1800), d=0.05)
    genome = np.random.default_rng(1).integers(0, 20, size=450)
    try:
        ref.reset()
        ref_r, i, done = 0.0, 0, False
        while not done:
            _, r, te, tr, info = ref.step(common.decode_gene(genome[i // 4]))
            i += 1
            ref_r += r
            done = te or tr
        ref_steps, ref_hp, ref_boss = i, info["hp"], info["boss_hp"]
    finally:
        ref.close()

    env, counter = common.make_task_env("gutsman", "eval")
    try:
        m = common.run_episode(env, counter, lambda k, o: common.decode_gene(genome[k // 4]),
                               sticky_p=0.25, rng=np.random.default_rng(7))
    finally:
        env.close()
    assert m["steps"] == ref_steps
    assert abs(m["reward"] - ref_r) < 1e-6
    assert 28 - m["hp_lost"] == ref_hp and m["win"] == (ref_boss == 0)


@needs_rom
def test_deep_ga_frame_stack_matches_vecframestack():
    """The manual stack used by the Deep GA must equal the PPO pipeline's VecFrameStack+VecTransposeImage."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
    from src.env import make_env
    actions = np.random.default_rng(3).integers(0, 20, size=12)

    venv = VecTransposeImage(VecFrameStack(DummyVecEnv([lambda: make_env(state="Gutsman-boss")]), 4,
                                           channels_order="last"))
    ref = [venv.reset()[0].copy()]
    for g in actions:
        ref.append(venv.step(common.decode_gene(g)[None])[0][0].copy())
    venv.close()

    env, counter = common.make_task_env("gutsman", "eval")
    stack = np.zeros((4, 84, 84), dtype=np.uint8)
    mine = []

    def on_obs(obs, is_reset):
        if is_reset:
            stack[:] = 0
        else:
            stack[:-1] = stack[1:]
        stack[-1] = obs[..., 0]
        mine.append(stack.copy())
    try:
        common.run_episode(env, counter, lambda i, o: common.decode_gene(actions[min(i, len(actions) - 1)]),
                           on_obs=on_obs)
    finally:
        env.close()
    for a, b in zip(ref, mine):                            # first 13 stacks (reset + 12 steps)
        assert np.array_equal(a, b)


@needs_rom
def test_deep_ga_episode_is_deterministic():
    import torch
    torch.set_num_threads(1)
    W = {}
    W["env"], W["counter"] = common.make_task_env("gutsman", "eval")
    try:
        a = deep_ga.run_chain_episode(W, [11, 22], 0.005, 0.0, np.random.default_rng(0), 0)
        b = deep_ga.run_chain_episode(W, [11, 22], 0.005, 0.0, np.random.default_rng(0), 0)
        assert a == b and a["steps"] > 0
    finally:
        W["env"].close()
