"""
Tests of the fair GA x PPO comparison: sticky actions in the training env, the GA policies evaluated through
src/eval.py, and the GA search loop of baselines.ga.fair (budget, PPO-style validation, sticky regime).
Emulator tests are skipped when rom.nes is absent.
"""
import json
import os

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from baselines.ga import common, fair
from baselines.ga.policy import DeepGAPolicy, OpenLoopPolicy, load_run_policy
from src.wrappers import StickyFrames


class RecordingEnv(gym.Env):
    observation_space = spaces.Box(0, 1, (1,), dtype=np.float32)
    action_space = spaces.MultiDiscrete([2, 5, 2])

    def __init__(self):
        self.actions = []

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.array(action).copy())
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def _sticky_trace(seed, n=300):
    base = RecordingEnv()
    env = StickyFrames(base, 0.5)
    env.reset(seed=seed)
    for i in range(n):
        env.step(np.array([i % 2, i % 5, (i // 2) % 2]))
    return np.array(base.actions)


# ----------------------------------------------------------------------------------------------
# Pure
# ----------------------------------------------------------------------------------------------
def test_sticky_reset_seed_reseeds_the_rng():
    a, b, c = _sticky_trace(3), _sticky_trace(3), _sticky_trace(4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sticky_reset_without_seed_keeps_the_injected_rng():
    base = RecordingEnv()
    env = StickyFrames(base, 1.0, np.random.default_rng(0))
    rng = env.rng
    env.reset()                       # what src/eval.py's DummyVecEnv does (seed=None)
    assert env.rng is rng


def test_eval_task_reward_is_the_unshaped_reward():
    assert fair.eval_task_reward({"hits": 14, "damage_events": 0}) == pytest.approx(0.70)
    assert fair.eval_task_reward({"hits": 24, "damage_events": 3}) == pytest.approx(1.05)


def test_fitness_key_orders_like_common_for_single_episodes():
    ms = [{"reward": 0.5, "win": False, "frames_to_kill": None},
          {"reward": 0.5, "win": True, "frames_to_kill": 900},
          {"reward": 0.5, "win": True, "frames_to_kill": 700},
          {"reward": 0.6, "win": False, "frames_to_kill": None}]
    assert sorted(range(4), key=lambda i: fair.fitness_key(ms[i])) == \
        sorted(range(4), key=lambda i: common.fitness_key(ms[i]))
    assert fair.fitness_key(None) < fair.fitness_key(ms[0])


def test_aggregate_means_rates_and_sums_steps():
    ms = [{"reward": 1.0, "win": True, "flawless": False, "hits": 14, "damage_events": 1, "hp_lost": 4,
           "died": False, "steps": 300, "frames": 1200, "frames_to_kill": 1200},
          {"reward": 0.0, "win": False, "flawless": False, "hits": 6, "damage_events": 3, "hp_lost": 28,
           "died": True, "steps": 200, "frames": 800, "frames_to_kill": None}]
    a = fair.aggregate(ms)
    assert a["reward"] == 0.5 and a["win"] == 0.5 and a["hits"] == 10
    assert a["steps"] == 500 and a["frames"] == 2000 and a["frames_to_kill"] == 1200 and a["episodes"] == 2
    assert fair.aggregate(ms[:1])["win"] is True


def test_best_rule_is_strictly_higher_like_evalcallback():
    assert fair.is_new_best(None, -1.0)
    best = {"val_reward": 0.5}
    assert not fair.is_new_best(best, 0.5)
    assert fair.is_new_best(best, 0.55)


def test_open_loop_policy_steps_through_the_genome_and_resets():
    pol = OpenLoopPolicy([0, 19, 7], k=2)
    acts = [tuple(pol.predict(np.zeros((1, 4, 84, 84)))[0][0]) for _ in range(7)]
    exp = [tuple(common.decode_gene(g)) for g in (0, 0, 19, 19, 7, 7, 7)]   # the last gene is held
    assert acts == exp
    pol.reset_episode()
    assert tuple(pol.predict(np.zeros((1, 4, 84, 84)))[0][0]) == tuple(common.decode_gene(0))


def test_deep_ga_policy_matches_the_ga_worker_network():
    import torch
    from baselines.ga import deep_ga
    chain = [12345, 678]
    pol = DeepGAPolicy(chain, 0.005)
    model = deep_ga.build_model().eval()
    theta = deep_ga.rebuild(chain, 0.005, model)
    torch.nn.utils.vector_to_parameters(torch.from_numpy(theta), model.parameters())
    obs = np.random.default_rng(0).integers(0, 256, (8, 4, 84, 84), dtype=np.uint8)
    with torch.no_grad():
        logits = model(torch.from_numpy(obs).float() / 255.0).numpy()
    exp = np.stack([deep_ga.act_from_logits(r) for r in logits])
    assert np.array_equal(pol.predict(obs)[0], exp)


# ----------------------------------------------------------------------------------------------
# Emulator
# ----------------------------------------------------------------------------------------------
def _rollout(env, actions, seed=None):
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for a in actions:
        obs, _, term, trunc, _ = env.step(a)
        out.append(obs)
        if term or trunc:
            break
    return np.array(out)


def test_make_env_sticky_is_seeded_and_off_by_default(open_env):
    rng = np.random.default_rng(0)
    actions = [rng.integers([0, 0, 0], [2, 5, 2]) for _ in range(150)]
    plain = _rollout(open_env(state="Gutsman-boss"), actions)
    zero = _rollout(open_env(state="Gutsman-boss", sticky=0.0), actions, seed=1)
    a = _rollout(open_env(state="Gutsman-boss", sticky=0.25), actions, seed=1)
    b = _rollout(open_env(state="Gutsman-boss", sticky=0.25), actions, seed=1)
    assert np.array_equal(plain, zero)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, plain)


@pytest.mark.parametrize("variant", ["deep-ga", "ga-ol"])
def test_ga_policy_through_src_eval_reproduces_the_ga_rollout(rom_available, variant):
    """The GA's own rollout (baselines.ga.common, evaluation task) and src.eval.evaluate() agree exactly."""
    from baselines.ga import deep_ga
    from src import eval as ev
    env, counter = common.make_task_env("gutsman", "eval")
    try:
        if variant == "deep-ga":
            chain = [987654, 321]
            ref = deep_ga.run_chain_episode({"env": env, "counter": counter}, chain, 0.005, 0.0,
                                            np.random.default_rng(0), 0)
            policy = DeepGAPolicy(chain, 0.005)
        else:
            genome = np.random.default_rng(3).integers(0, common.N_GENES, 450)
            ref = common.run_episode(env, counter, lambda i, obs: common.decode_gene(genome[min(i // 4, 449)]),
                                     0.0, np.random.default_rng(0), 0)
            policy = OpenLoopPolicy(genome, 4)
    finally:
        env.close()
    res = ev.evaluate(policy, "Gutsman-boss", episodes=1, deterministic=True, sticky=0.0, noop_max=0,
                      episode_frames=7200, seed=0, verbose=False)["episodes"][0]
    assert res["win"] == ref["win"]
    assert res["hp_lost"] == ref["hp_lost"]
    assert res["frames"] == ref["frames"]
    assert res["reward"] == pytest.approx(ref["reward"], abs=1e-6)   # VecEnv rewards are float32


def test_fair_search_det_regime_budget_validation_and_outputs(rom_available, tmp_path):
    out = str(tmp_path / "deep")
    s = fair.main(["--variant", "deep-ga", "--boss", "gutsman", "--max-env-steps", "6000",
                   "--validate-every", "1500", "--pop", "6", "--procs", "2", "--seed", "1", "--out-dir", out,
                   "--snapshot-at", "2000", "4000", "99999"])
    assert s["regime"] == "det" and s["stopped_by"] in ("budget", "no_damage")
    assert s["env_steps"] >= 6000 or s["stopped_by"] == "no_damage"
    assert s["env_steps"] - 6000 < 2 * 1800                        # at most one chunk of `procs` episodes
    hist = json.load(open(os.path.join(out, "history.json")))
    vals = json.load(open(os.path.join(out, "validations.json")))
    best = json.load(open(os.path.join(out, "best.json")))
    assert len(vals) >= 1
    first_max = next(v for v in vals if v["val_reward"] == max(x["val_reward"] for x in vals))
    assert best["generation"] == first_max["generation"]
    assert hist[-1]["cum_env_steps"] == s["env_steps"]
    for which in ("best", "final"):
        policy, rec = load_run_policy(out, which)
        assert rec["variant"] == "deep-ga" and isinstance(policy, DeepGAPolicy)
    for step in (2000, 4000):            # snapshot = top individual of the generation crossing the step
        snap = json.load(open(os.path.join(out, f"snapshot_{step}.json")))
        assert snap["cum_env_steps"] >= step
        crossing = next(h for h in hist if h["cum_env_steps"] >= step)
        assert snap["generation"] == crossing["generation"]
    _, rec = load_run_policy(out, "snapshot_99999")   # never reached: the final individual
    assert rec["snapshot_fallback"] and rec["generation"] == json.load(open(os.path.join(out, "final.json")))["generation"]


def test_fair_search_sticky_regime_and_ga_evaluate(rom_available, tmp_path):
    from baselines.ga import evaluate
    out = str(tmp_path / "ol")
    s = fair.main(["--variant", "ga-ol", "--boss", "gutsman", "--max-env-steps", "5000", "--pop", "4",
                   "--elite", "1", "--train-sticky", "0.25", "--episodes-per-eval", "2", "--procs", "2",
                   "--final-estimate-episodes", "2", "--seed", "2", "--out-dir", out])
    assert s["regime"] == "sticky" and s["n_validations"] == 0 and s["best"] is None
    assert s["final_estimate"]["episodes"] == 2
    assert not os.path.exists(os.path.join(out, "best.json"))
    res = evaluate.main(["--run", out, "--which", "final", "--episodes", "2", "--sticky", "0.25", "--seed", "123",
                         "--json", str(tmp_path / "ev.json")])
    assert res["config"]["variant"] == "ga-ol" and res["aggregate"]["n_episodes"] == 2
    assert json.load(open(tmp_path / "ev.json"))["config"]["sticky"] == 0.25


# ----------------------------------------------------------------------------------------------
# Campaign design (experiments/fair): the job graph encodes the protocol
# ----------------------------------------------------------------------------------------------
def _real_jobs(monkeypatch):
    monkeypatch.delenv("FAIR_SMOKE", raising=False)
    import importlib
    from experiments.fair import config, jobs
    importlib.reload(config)
    importlib.reload(jobs)
    return config, jobs, jobs.build_jobs()


def test_job_graph_counts_and_dependencies(monkeypatch):
    C, J, js = _real_jobs(monkeypatch)
    kinds = {}
    for j in js:
        kinds[j.kind] = kinds.get(j.kind, 0) + 1
    # docs/EXPERIMENTS.md: tuning of the 3 methods (4 configs x 6 bosses; PPO tuning runs are kind "ppo") + GA K
    # tuning (2 K x 6 bosses x 2 GAs); PPO 6 bosses x 2 regimes x 5 seeds; GAs 2 x 2 regimes x 6 x 5; 180 runs x (1 eval + 5 curve points)
    assert kinds == {"tune": 72, "select": 5, "ppo": 24 + 60, "ga": 120, "eval": 180, "curve": 900,
                     "aggregate": 1}
    assert C.conditions("ppo")[-1] == "stickystoch" and "stickystoch" not in C.conditions("deep-ga")
    assert all(len(C.DET_GRID[m]) == 4 for m in C.METHODS)          # the same tuning effort for all
    assert set(C.TUNE_BOSSES) == set(C.BOSSES) and not set(C.FOLDS["A"]) & set(C.FOLDS["B"])
    by_id = {j.id: j for j in js}
    for j in js:
        if j.kind == "eval":
            _, regime, method, boss, seed, model = j.id.split("/")
            assert j.deps == (J.train_id(regime, method, boss, int(seed[1:])),)
            assert model in C.MODELS[regime]
        if j.kind == "ga":
            assert j.deps == (f"select/{j.id.split('/')[1]}/{j.id.split('/')[2]}",)
        if j.kind == "ppo" and not j.id.startswith("tune/"):
            assert j.deps == ("select/det/ppo",)
        if j.kind == "curve":
            _, regime, method, boss, seed, step = j.id.split("/")
            assert j.deps == (J.train_id(regime, method, boss, int(seed[1:])),)
            assert int(step) in C.curve_steps(boss)
    assert set(by_id["aggregate"].deps) == {j.id for j in js if j.kind in ("eval", "curve")}
    for boss in C.BOSSES:      # every PPO curve step is a CheckpointCallback step
        assert all(st % C.PPO_CHECKPOINT_EVERY == 0 for st in C.curve_steps(boss))


def test_regime_flags_in_commands(monkeypatch):
    C, J, _ = _real_jobs(monkeypatch)
    hp = dict(C.DET_GRID["ppo"][1][1])
    det = J._ppo_train("t", "cutman", 1, hp, "det", 3_000_000)
    sticky = J._ppo_train("t", "cutman", 1, hp, "sticky", 3_000_000)
    assert "--timesteps 3000000" in det and "--lr-horizon 40000000" in det and "--lr 0.0001 --ent-coef 0.005" in det
    assert "--train-sticky" not in det and "--no-early-stop" not in det
    assert "--train-sticky 0.25 --no-early-stop" in sticky
    tune = J._ppo_tune_cmd("lr10_e5", "cutman", hp)
    assert "--seed 1000" in tune and "experiments.fair.ppo_summary fairtune_lr10_e5_cutman cutman" in tune
    assert C.ppo_tag("det", "cutman", 1, "lr25_e5") == "fair_det_cutman_s1"       # default keeps the old tag
    assert C.ppo_tag("det", "cutman", 1, "lr10_e5") == "fair_det_cutman_s1_lr10_e5"
    hp = dict(C.DET_GRID["deep-ga"][0][1])
    g_det = J._ga_cmd("deep-ga", "bombman", 0, "/tmp/x", hp, "det", 1)
    g_sticky = J._ga_cmd("deep-ga", "bombman", 0, "/tmp/x", hp, "sticky", 4)
    assert "--max-env-steps 3000000" in g_det and "--train-sticky" not in g_det
    assert "--train-sticky 0.25 --episodes-per-eval 4 --no-early-stop" in g_sticky
    assert "--snapshot-at" not in g_det and "/eval/" not in g_det          # tuning runs: no snapshots
    g_camp = J._ga_cmd("deep-ga", "bombman", 0, "/tmp/x", hp, "det", 1, snapshots=True)
    assert g_camp.endswith("--snapshot-at 500000 1000000 1500000 2000000 2500000")
    assert g_camp.startswith("rm -rf ") and "/eval/det/deep-ga/bombman/s0" in g_camp


def test_fold_selection_is_held_out(monkeypatch, tmp_path):
    """A boss gets the config selected on the OTHER fold: its own tuning runs never choose its config."""
    C, J, _ = _real_jobs(monkeypatch)
    from experiments.fair import tune_select as T
    monkeypatch.setattr(C, "CAMPAIGN", str(tmp_path))
    best = {"A": "s002_p96", "B": "s005_p256"}     # the config that wins on each fold's own bosses
    for cid, _ in C.DET_GRID["deep-ga"]:
        for b in C.TUNE_BOSSES:
            d = C.tune_dir("det", "deep-ga", cid, b)
            os.makedirs(d)
            win = cid == best[C.fold_of(b)]
            json.dump({"first_no_damage": {"cum_env_steps": 1_000_000} if win else None}, open(f"{d}/summary.json", "w"))
            json.dump([{"val_reward": 0.5}], open(f"{d}/validations.json", "w"))
    T.select("det", "deep-ga")
    for b in C.BOSSES:
        assert C.chosen_config("det", "deep-ga", b)["config_id"] == best[C.other_fold(C.fold_of(b))]


def test_tune_select_ranks_share_ties():
    from experiments.fair.tune_select import ranks
    assert ranks([(1, -5, 1.0), (0, 0, 2.0), (1, -5, 1.0), (1, -3, 0.5)]) == [2.5, 4.0, 2.5, 1.0]
    assert (1, -3_000_000, 0.0) > (0, 0, 1.9)      # reaching the no-damage stop beats any validation reward


def test_agarwal_metrics():
    from experiments.fair import stats as S
    assert S.iqm([1, 2, 3, 4]) == 2.5 and S.iqm(list(range(10))) == 4.5
    assert S.prob_improvement([[1, 1]], [[0, 0]]) == 1.0 and S.prob_improvement([[0.5]], [[0.5]]) == 0.5
    pt, lo, hi = S.aggregate_ci([[1, 1, 1], [0, 0, 0]], np.mean, n_boot=200)
    assert pt == lo == hi == 0.5          # no within-boss variation: the stratified bootstrap is exact
    d, lo, hi = S.difference_ci([[1, 1, 1]], [[0, 0, 0]], S.iqm, n_boot=200)
    assert d == lo == hi == 1.0
