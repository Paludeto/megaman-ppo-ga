"""
Module: experiments.fair.config
Description: The fixed design of the fair GA x PPO comparison (docs/EXPERIMENTS.md:
             full design on the 3M budget - two training regimes, 5 seeds, GA tuning). Every number
             here is decided BEFORE any campaign result; changing one after results exist breaks the protocol.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable                                        # the interpreter that runs the scheduler
CAMPAIGN = os.path.join(ROOT, "campaign", "fair")          # outputs (not versioned)

# Robot Masters and the agent-step budget: the SAME 3M steps for every boss and every method (compute
# constraint). The stop rule is the first no-damage validation, checked every 100k steps.
BUDGET = 3_000_000
BOSSES = {b: {"state": s, "cap": BUDGET} for b, s in (
    ("gutsman", "Gutsman-boss"), ("cutman", "Cutman-boss"), ("bombman", "Bombman-boss"),
    ("elecman", "Elecman-boss"), ("iceman", "Iceman-boss"), ("fireman", "Fireman-boss"))}
EPISODE_FRAMES = 7200
LR_HORIZON = 40_000_000           # the LR schedule does not depend on the budget

# Training regimes. det = deterministic setting (deterministic env in training, early stop at the first no-damage
# validation). sticky = Machado et al. 2018 (sticky actions p=0.25 in training, no early stop, final model).
# Sticky actions are the primary EVALUATION condition in both regimes.
REGIMES = ("det", "sticky")
STICKY_P = 0.25

# Seeds: 5 per method, boss and regime.
PPO_SEEDS = (666, 1, 2, 3, 4)
GA_SEEDS = (0, 1, 2, 3, 4)

METHODS = ("ppo", "deep-ga", "ga-ol")
GA_VARIANTS = ("deep-ga", "ga-ol")

# Model evaluated: the final one (the policy at the no-damage stop, or at the budget), in both regimes.
MODELS = {"det": ("final",), "sticky": ("final",)}
PRIMARY_MODEL = {"det": "final", "sticky": "final"}

# Evaluation: src.eval.evaluate() for every model, seed 123 (same per-episode streams for all methods).
# name -> (sticky p, no-op max, episodes, stochastic policy). "stickystoch" (sampled policy under sticky) is PPO-only and secondary:
# the GAs never train a stochastic policy, but PPO's greedy policy is not the one it was trained with.
EVAL_SEED = 123
CONDITIONS = {
    "det1": (0.0, 0, 1, False),
    "sticky": (0.25, 0, 30, False),
    "stickynoop": (0.25, 30, 30, False),
    "noop": (0.0, 30, 30, False),
    "stickystoch": (0.25, 0, 30, True),
}
PRIMARY_CONDITION = "sticky"

# Learning curves: the policy every 500k steps below the budget (PPO: its CheckpointCallback file; GA: the
# top individual of the generation that crosses the step), evaluated in the primary condition. A run that
# stopped earlier (no-damage stop) keeps its stopped policy for the later points.
PPO_CHECKPOINT_EVERY = 500_000
CURVE_CONDITION = "sticky"


def curve_steps(boss):
    return list(range(PPO_CHECKPOINT_EVERY, BOSSES[boss]["cap"], PPO_CHECKPOINT_EVERY))


def snapshot_name(step):
    return f"snapshot_{step}"


# Hyperparameter tuning, the SAME effort for the three methods (docs/EXPERIMENTS.md): a grid of 4 configs each,
# one tuning seed (not a campaign seed), the campaign budget, det regime, and one criterion for all:
#   (reached the no-damage stop, fewer steps to it, final deterministic validation reward)
# i.e. the training-time validation every method already runs - never the evaluation metric.
# Held-out bosses: 2-fold cross-validation over bosses. Every config is run on every boss; a boss is trained
# in the campaign with the config chosen on the OTHER fold (mean rank over its 3 bosses; ties go to the
# earlier config, the defaults come first). So no boss takes part in choosing its own hyperparameters.
# Sticky regime: the det choice; the GAs also tune K (episodes per individual, a noise-handling knob that PPO
# does not have) with the same folds; criterion: mean training reward of the final individual over 30
# sticky episodes (seeds 900000+).
FOLDS = {"A": ("gutsman", "cutman", "elecman"), "B": ("bombman", "iceman", "fireman")}
TUNE_BOSSES = FOLDS["A"] + FOLDS["B"]
TUNE_SEED = 1000
TUNE_ESTIMATE_EPISODES = 30
DET_GRID = {
    "ppo": [
        ("lr25_e5", {"lr": 2.5e-4, "ent_coef": 5e-3}),
        ("lr10_e5", {"lr": 1e-4, "ent_coef": 5e-3}),
        ("lr25_e10", {"lr": 2.5e-4, "ent_coef": 1e-2}),
        ("lr10_e10", {"lr": 1e-4, "ent_coef": 1e-2}),
    ],
    "deep-ga": [
        ("s005_p96", {"sigma": 0.005, "pop": 96, "parents": 10, "elite": 1}),
        ("s002_p96", {"sigma": 0.002, "pop": 96, "parents": 10, "elite": 1}),
        ("s005_p256", {"sigma": 0.005, "pop": 256, "parents": 20, "elite": 1}),
        ("s002_p256", {"sigma": 0.002, "pop": 256, "parents": 20, "elite": 1}),
    ],
    "ga-ol": [
        ("m020", {"p_mut": 0.02, "pop": 240, "k": 4, "elite": 6, "trunc": 0.2, "p_cross": 0.5}),
        ("m005", {"p_mut": 0.005, "pop": 240, "k": 4, "elite": 6, "trunc": 0.2, "p_cross": 0.5}),
        ("m020_p96", {"p_mut": 0.02, "pop": 96, "k": 4, "elite": 6, "trunc": 0.2, "p_cross": 0.5}),
        ("m005_p96", {"p_mut": 0.005, "pop": 96, "k": 4, "elite": 6, "trunc": 0.2, "p_cross": 0.5}),
    ],
}
DEFAULT_CONFIG = {m: g[0][0] for m, g in DET_GRID.items()}
STICKY_K_GRID = (1, 4)            # episodes per individual in the sticky regime (GAs; on top of the det choice)


def fold_of(boss):
    return next(f for f, bs in FOLDS.items() if boss in bs)


def other_fold(fold):
    return next(f for f in FOLDS if f != fold)

# Scheduler resources, sized for a 16-core / 32-thread machine with one GPU; adjust them to your hardware.
# Slots approximate real CPU use: a PPO run (16 env processes that idle during the GPU updates) keeps ~7-8 threads busy, a GA run
# (12 emulator processes) ~11, an evaluation 1. Strict priority PPO > GA > evaluations: when a job of a
# higher class is ready and under its cap but does not fit, nothing of a lower class starts, so the slots
# it needs free up (the PPO path is the longest). Up to GPU_SLOTS PPO runs while GA work remains, then
# GPU_SLOTS_NO_GA; up to MAX_GA GA runs. Concurrency changes only the wall-clock time, never a result.
CPU_SLOTS = 40
GPU_SLOTS = 2
GPU_SLOTS_NO_GA = 3
MAX_GA = 3
MAX_GA_WITH_PPO = 1  # measured: a 2nd GA run slows each PPO run from ~75k to ~50k steps/min (SMT sharing)
NICE_NON_PPO = 10    # GA runs and evaluations yield the CPU to PPO (see scheduler.py)
SLOTS = {"ppo": 8, "ga": 12, "eval": 1, "select": 0, "aggregate": 1}

# FAIR_SMOKE=1: the whole job graph in miniature (2 bosses, 2 seeds, tiny budgets, 2 episodes per
# condition) in campaign/fair_smoke, to test the pipeline end to end. Never used for results.
SMOKE = os.environ.get("FAIR_SMOKE") == "1"
if SMOKE:
    CAMPAIGN = os.path.join(ROOT, "campaign", "fair_smoke")
    BOSSES = {"gutsman": {"state": "Gutsman-boss", "cap": 110_000},
              "cutman": {"state": "Cutman-boss", "cap": 110_000}}
    PPO_SEEDS, GA_SEEDS = (666, 1), (0, 1)
    FOLDS = {"A": ("gutsman",), "B": ("cutman",)}
    TUNE_BOSSES = ("gutsman", "cutman")
    DET_GRID = {v: g[:2] for v, g in DET_GRID.items()}
    STICKY_K_GRID = (1, 2)
    TUNE_ESTIMATE_EPISODES = 3
    CONDITIONS = {k: (p, n, min(e, 2), st) for k, (p, n, e, st) in CONDITIONS.items()}

    def curve_steps(boss):      # noqa: F811 - two points inside the tiny budget
        return [50_000, 100_000]

assert SMOKE or all(st % PPO_CHECKPOINT_EVERY == 0 for b in BOSSES for st in curve_steps(b)), "curve steps vs checkpoints"


def ppo_tag(regime, boss, seed, config_id=None):
    """PPO run tag; the default config keeps the historical tag (so its runs are reused), others get a suffix."""
    base = f"fairsmoke_{regime}_{boss}_s{seed}" if SMOKE else f"fair_{regime}_{boss}_s{seed}"
    return base if config_id in (None, DEFAULT_CONFIG["ppo"]) else f"{base}_{config_id}"


def ppo_tune_tag(config_id, boss):
    return f"{'fairsmoke' if SMOKE else 'fair'}tune_{config_id}_{boss}"


def chosen_config(regime, method, boss):
    """{"config_id", "args"[, "K"]} chosen for `boss` (on the other fold); None before the selection ran."""
    import json
    path = chosen_path(regime, method)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("by_boss", {}).get(boss)


def ppo_run_tag(regime, boss, seed):
    """Tag of the campaign PPO run of this cell (its config is the det choice, also in the sticky regime)."""
    ch = chosen_config("det", "ppo", boss)
    return ppo_tag(regime, boss, seed, ch["config_id"] if ch else None)


def ga_run_dir(regime, variant, boss, seed):
    return os.path.join(CAMPAIGN, "runs", regime, variant, boss, f"s{seed}")


def tune_dir(regime, variant, config_id, boss):
    return os.path.join(CAMPAIGN, "tuning", regime, variant, config_id, boss)


def chosen_path(regime, method):
    return os.path.join(CAMPAIGN, "tuning", f"chosen_{regime}_{method}.json")


def select_id(regime, method):
    return f"select/{regime}/{method}"


def eval_path(regime, method, boss, seed, model, condition):
    return os.path.join(CAMPAIGN, "eval", regime, method, boss, f"s{seed}", f"{model}__{condition}.json")


def seeds(method):
    return PPO_SEEDS if method == "ppo" else GA_SEEDS


def conditions(method):
    return [c for c in CONDITIONS if c != "stickystoch" or method == "ppo"]
