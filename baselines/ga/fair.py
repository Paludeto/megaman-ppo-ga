"""
Module: baselines.ga.fair
Description: GA search loop of the fair GA x PPO comparison (docs/EXPERIMENTS.md), for both
             variants (deep-ga, ga-ol). It gives the GAs what src/train.py gives PPO:

  Budget        --max-env-steps agent steps (the PPO cap of the boss); the steps of every rollout count.
                When the remaining budget could end inside a generation, the generation runs in chunks of
                `procs` episodes and stops at the budget, so the overshoot is at most one chunk (PPO's is
                at most one rollout of 16 x 1024 steps). Unevaluated individuals of that last generation
                are dropped.
  Regime det    deterministic env: one exact rollout per individual, elites keep their
                metrics. Every --validate-every steps (100k = PPO's eval_freq x n_envs) the current elite
                is validated like PPO's ValidationCallback: eval-task reward of its deterministic episode
                (+d per hit, -d per damage, no bonus; exact from the rollout), best = strictly highest
                (the first one on ties, as EvalCallback), stop at the first no-damage win
                (--no-early-stop disables the stop).
  Regime sticky (--train-sticky P; Machado et al. 2018) per-frame sticky actions in every rollout; each
                individual plays --episodes-per-eval K episodes with fresh seeds and its fitness is the
                mean; every individual, elites included, is re-evaluated each generation. No validation
                and no early stop (the model reported is the final one).
  Fitness       undiscounted return of the PPO training reward (src/train.py; baselines.ga.common).

Outputs (--out-dir): config.json, history.json, validations.json, best.json (best-validation model; regime
det), final.json (top individual of the last generation), snapshot_<step>.json (--snapshot-at: the top
individual of the generation that crosses each step, for learning curves) and summary.json. The final evaluation is not
done here: baselines/ga/evaluate.py runs src/eval.py on best.json/final.json, exactly as for PPO.

Usage:
  python -m baselines.ga.fair --variant deep-ga --boss cutman --max-env-steps 3000000 --seed 0 --out-dir DIR
  python -m baselines.ga.fair --variant ga-ol --boss cutman --max-env-steps 3000000 --train-sticky 0.25 \
      --episodes-per-eval 4 --no-early-stop --seed 0 --out-dir DIR
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.ga import common, deep_ga, ga_ol  # noqa: E402

VALIDATE_EVERY = 100_000          # PPO: eval_freq = 100000 // n_envs calls of n_envs steps
SEED_MAX = 2 ** 31 - 1
ESTIMATE_SEED0 = 900_000          # seeds of the post-search estimate (tuning only), disjoint from eval's


# ----------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ----------------------------------------------------------------------------------------------
def eval_task_reward(m):
    """
    Evaluation-task reward of a deterministic rollout: +d per hit, -d per damage event, no bonus.
    BossWrapper counts both events per agent step, exactly as common.run_episode does, so this equals
    the reward PPO's validation env would give to the same trajectory.
    """
    return round(common.D * m["hits"] - common.D * m["damage_events"], 6)


def fitness_key(m):
    """Larger is better: mean training reward, then win (rate), then fewer frames to the kill."""
    if m is None:
        return (-math.inf, -math.inf, -math.inf)
    return (m["reward"], float(m["win"]), -(m["frames_to_kill"] or 0))


def aggregate(ms):
    """Metrics of one individual over its K episodes (K == 1: the episode itself)."""
    if len(ms) == 1:
        return dict(ms[0], episodes=1)
    kills = [m["frames_to_kill"] for m in ms if m["frames_to_kill"] is not None]

    def mean(k):
        return float(np.mean([float(m[k]) for m in ms]))
    return {
        "reward": round(mean("reward"), 6), "win": mean("win"), "flawless": mean("flawless"),
        "hits": mean("hits"), "damage_events": mean("damage_events"), "hp_lost": mean("hp_lost"),
        "died": mean("died"), "steps": sum(m["steps"] for m in ms), "frames": sum(m["frames"] for m in ms),
        "frames_to_kill": float(np.mean(kills)) if kills else None, "episodes": len(ms),
    }


def is_new_best(best, val_reward):
    """EvalCallback's rule: a validation replaces the best only if its reward is STRICTLY higher."""
    return best is None or val_reward > best["val_reward"]


# ----------------------------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------------------------
class DeepGA:
    kind = "deep"
    name = "deep-ga"

    def __init__(self, args, rng):
        self.args, self.rng = args, rng

    def init_population(self):
        return [[int(self.rng.integers(0, SEED_MAX))] for _ in range(self.args.pop)]

    def payload(self, ind, sticky_p, seed):
        return (list(ind), self.args.sigma, sticky_p, seed, 0)

    def next_generation(self, pop, keys):
        return deep_ga.next_generation(pop, keys, self.rng, self.args.parents, self.args.elite)

    def record(self, ind):
        return {"chain": [int(s) for s in ind], "sigma": self.args.sigma}


class OpenLoopGA:
    kind = "ol"
    name = "ga-ol"

    def __init__(self, args, rng):
        self.args, self.rng = args, rng
        self.length = int(np.ceil(ga_ol.episode_steps(args.boss) / args.k))

    def init_population(self):
        pop = self.rng.integers(0, common.N_GENES, size=(self.args.pop, self.length))
        return [[int(g) for g in row] for row in pop]

    def payload(self, ind, sticky_p, seed):
        return (list(ind), self.args.k, sticky_p, seed, 0)

    def next_generation(self, pop, keys):
        a = self.args
        children, elites = ga_ol.next_population(np.asarray(pop), keys, self.rng, a.elite, a.trunc,
                                                 a.p_cross, a.p_mut)
        return [[int(g) for g in row] for row in children], elites

    def record(self, ind):
        return {"genome": [int(g) for g in ind], "k": self.args.k}


# ----------------------------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------------------------
def _dump(out_dir, name, obj):
    tmp = os.path.join(out_dir, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, os.path.join(out_dir, name))


def evaluate_generation(pool, var, pop, todo, sticky_p, k_eps, ep_rng, budget_left, procs, episode_steps):
    """
    Run K episodes of every individual in `todo` (in order: the elites, rows 0..elite-1, come first).
    Returns ({index: aggregated metrics}, steps, frames). If the budget could end inside this
    generation, episodes run in chunks of `procs` and stop once `budget_left` is used up; individuals
    whose K episodes did not all run are left out.
    """
    jobs = []
    for i in todo:
        for _ in range(k_eps):
            seed = int(ep_rng.integers(0, SEED_MAX)) if sticky_p > 0 else 0
            jobs.append((i, var.payload(pop[i], sticky_p, seed)))
    results = {}
    if budget_left >= len(jobs) * episode_steps:            # cannot run out: one parallel map
        for j, m in enumerate(pool.map(var.kind, [p for _, p in jobs])):
            results[j] = m
    else:
        used, j = 0, 0
        while j < len(jobs) and used < budget_left:
            chunk = jobs[j:j + procs]
            for q, m in enumerate(pool.map(var.kind, [p for _, p in chunk])):
                results[j + q] = m
                used += m["steps"]
            j += len(chunk)
    per = {}
    for j, (i, _) in enumerate(jobs):
        if j in results:
            per.setdefault(i, []).append(results[j])
    complete = {i: aggregate(ms) for i, ms in per.items() if len(ms) == k_eps}
    steps = sum(m["steps"] for m in results.values())
    frames = sum(m["frames"] for m in results.values())
    return complete, steps, frames


def search(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)                      # GA operators
    ep_rng = np.random.default_rng([args.seed, 0x5EED])         # episode seeds (sticky regime only)
    var = DeepGA(args, rng) if args.variant == "deep-ga" else OpenLoopGA(args, rng)
    sticky_p = float(args.train_sticky)
    sticky = sticky_p > 0.0
    k_eps = args.episodes_per_eval if sticky else 1
    stop_on_no_damage = not (sticky or args.no_early_stop)
    episode_steps = ga_ol.episode_steps(args.boss)

    config = dict(vars(args), regime="sticky" if sticky else "det", episodes_per_eval_effective=k_eps,
                  stop_on_no_damage=stop_on_no_damage, episode_steps=episode_steps,
                  episode_frames=common.BOSSES[args.boss]["frames"],
                  fitness=common.task_description("train"))
    if args.variant == "deep-ga":
        config.update(n_params=int(sum(p.numel() for p in deep_ga.build_model().parameters())),
                      architecture="NatureCNN(512) + Linear(512, 9)")
    else:
        config.update(genome_length=var.length)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    pool = common.EvalPool(args.boss, args.procs, task="train")
    pop = var.init_population()
    metrics = [None] * len(pop)
    history, validations = [], []
    best = final = first_no_damage = None
    cum_steps = cum_frames = gen = 0
    next_val = args.validate_every
    snapshots = sorted(set(args.snapshot_at or []))
    stopped_by = None
    t0 = time.time()
    while True:
        todo = list(range(len(pop))) if sticky else [i for i, m in enumerate(metrics) if m is None]
        complete, gen_steps, gen_frames = evaluate_generation(
            pool, var, pop, todo, sticky_p, k_eps, ep_rng, args.max_env_steps - cum_steps, args.procs,
            episode_steps)
        for i in todo:
            metrics[i] = complete.get(i)
        cum_steps += gen_steps
        cum_frames += gen_frames
        keys = [fitness_key(m) for m in metrics]
        evaluated = [i for i, m in enumerate(metrics) if m is not None]
        if not evaluated:            # the budget ended before any individual finished its K episodes
            stopped_by = "budget"
            break
        bi = max(evaluated, key=lambda i: keys[i])
        top = metrics[bi]
        final_ind = pop[bi]
        final = {"variant": var.name, "generation": gen, "cum_env_steps": cum_steps, "metrics": top,
                 "individual": var.record(final_ind)}
        rec = {
            "generation": gen, "best_reward": top["reward"], "best_win": top["win"],
            "best_flawless": top["flawless"], "best_hits": top["hits"],
            "best_damage_events": top["damage_events"], "best_frames_to_kill": top["frames_to_kill"],
            "mean_reward": float(np.mean([metrics[i]["reward"] for i in evaluated])),
            "n_evaluated": len(evaluated), "gen_env_steps": gen_steps, "cum_env_steps": cum_steps,
            "cum_env_frames": cum_frames, "elapsed_s": round(time.time() - t0, 1),
        }
        while snapshots and cum_steps >= snapshots[0]:
            _dump(out_dir, f"snapshot_{snapshots.pop(0)}.json", final)
        if not sticky and cum_steps >= next_val:
            while next_val <= cum_steps:
                next_val += args.validate_every
            v = {"variant": var.name, "generation": gen, "cum_env_steps": cum_steps,
                 "val_reward": eval_task_reward(top), "no_damage": bool(top["flawless"]),
                 "metrics": top, "individual": var.record(pop[bi])}
            validations.append({k: v[k] for k in ("generation", "cum_env_steps", "val_reward", "no_damage")})
            if is_new_best(best, v["val_reward"]):
                best = v
            if v["no_damage"] and first_no_damage is None:
                first_no_damage = {"generation": gen, "cum_env_steps": cum_steps}
            rec["validation"] = validations[-1]
            if v["no_damage"] and stop_on_no_damage:
                stopped_by = "no_damage"
        history.append(rec)
        _dump(out_dir, "history.json", history)
        _dump(out_dir, "validations.json", validations)
        _dump(out_dir, "final.json", final)
        if best is not None:
            _dump(out_dir, "best.json", best)
        print(f"gen {gen:4d} best={top['reward']:.3f} hits={top['hits']} dmg={top['damage_events']} "
              f"win={top['win']} mean={rec['mean_reward']:.3f} steps={cum_steps} t={rec['elapsed_s']}s"
              + (f" val={rec['validation']['val_reward']:.2f}" if "validation" in rec else ""), flush=True)
        gen += 1
        if stopped_by is None:
            if cum_steps >= args.max_env_steps:
                stopped_by = "budget"
            elif (time.time() - t0) / 60.0 >= args.time_budget_min:
                stopped_by = "time"
            elif gen >= args.max_generations:
                stopped_by = "generations"
        if stopped_by is not None:
            break
        old = metrics
        pop, elites = var.next_generation(pop, keys)
        metrics = [old[i] for i in elites] + [None] * (len(pop) - len(elites))

    summary = {
        "variant": var.name, "boss": args.boss, "regime": config["regime"], "seed": args.seed,
        "stopped_by": stopped_by, "generations": gen, "env_steps": cum_steps, "env_frames": cum_frames,
        "budget": args.max_env_steps, "search_seconds": round(time.time() - t0, 1),
        "first_no_damage": first_no_damage, "n_validations": len(validations),
        "best": None if best is None else {k: best[k] for k in ("generation", "cum_env_steps", "val_reward",
                                                                "no_damage")},
        "final": {k: final[k] for k in ("generation", "cum_env_steps", "metrics")},
    }
    if args.final_estimate_episodes > 0:
        # Post-search estimate of the final individual on the training task (tuning criterion of the
        # sticky regime). Uncounted, like PPO's validation episodes; seeds disjoint from src/eval.py's.
        p = sticky_p
        ms = pool.map(var.kind, [var.payload(final_ind, p, ESTIMATE_SEED0 + e)
                                 for e in range(args.final_estimate_episodes)])
        summary["final_estimate"] = {"episodes": len(ms), "sticky_p": p, **aggregate(ms)}
    pool.close()
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"search done: {stopped_by}, {gen} generations, {cum_steps} env steps, "
          f"{summary['search_seconds']:.0f}s", flush=True)
    return summary


def build_parser():
    ap = argparse.ArgumentParser(description="GA search of the fair GA x PPO comparison")
    ap.add_argument("--variant", choices=("deep-ga", "ga-ol"), required=True)
    ap.add_argument("--boss", choices=list(common.BOSSES), required=True)
    ap.add_argument("--max-env-steps", type=int, required=True, help="agent-step budget (the PPO cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-sticky", type=float, default=0.0, help="per-frame sticky p in training (regime sticky)")
    ap.add_argument("--episodes-per-eval", type=int, default=1, help="K episodes per individual (regime sticky)")
    ap.add_argument("--no-early-stop", action="store_true", help="regime det: do not stop at a no-damage validation")
    ap.add_argument("--validate-every", type=int, default=VALIDATE_EVERY)
    ap.add_argument("--snapshot-at", type=int, nargs="*", default=None,
                    help="agent steps at which the current top individual is saved (learning curves)")
    ap.add_argument("--final-estimate-episodes", type=int, default=0,
                    help="after the search, N training-task episodes of the final individual (tuning)")
    ap.add_argument("--time-budget-min", type=float, default=24 * 60.0, help="safety cap only")
    ap.add_argument("--max-generations", type=int, default=10 ** 6)
    ap.add_argument("--procs", type=int, default=common.MAX_PROCS)
    # Deep GA
    ap.add_argument("--pop", type=int, default=None, help="default: 96 (deep-ga), 240 (ga-ol)")
    ap.add_argument("--parents", type=int, default=10)
    ap.add_argument("--elite", type=int, default=None, help="default: 1 (deep-ga), 6 (ga-ol)")
    ap.add_argument("--sigma", type=float, default=0.005)
    # GA-OL
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--trunc", type=float, default=0.2)
    ap.add_argument("--p-cross", type=float, default=0.5)
    ap.add_argument("--p-mut", type=float, default=0.02)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.pop is None:
        args.pop = 96 if args.variant == "deep-ga" else 240
    if args.elite is None:
        args.elite = 1 if args.variant == "deep-ga" else 6
    if not 0.0 <= args.train_sticky <= 1.0:
        ap.error("--train-sticky must be in [0, 1]")
    if args.episodes_per_eval < 1:
        ap.error("--episodes-per-eval must be >= 1")
    if args.train_sticky == 0.0 and args.episodes_per_eval != 1:
        ap.error("--episodes-per-eval > 1 only makes sense with --train-sticky (the env is deterministic)")
    if args.procs > common.MAX_PROCS:
        ap.error(f"at most {common.MAX_PROCS} emulator processes")
    return search(args)


if __name__ == "__main__":
    main()
