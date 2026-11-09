"""
Module: experiments.fair.random_baseline
Description: Exploratory floor: a policy that samples each action uniformly at every agent step, through
             src.eval.evaluate() with the campaign's evaluation streams (5 policy seeds, 30 episodes per boss
             and condition). Writes campaign/fair/random_baseline.json. It is not a job of the scheduler.

Usage: python -m experiments.fair.random_baseline [--procs 12]
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

from experiments.fair import config as C

SEEDS = range(5)
EPISODES = 30
CONDITIONS = {"sticky": (0.25, 0), "stickynoop": (0.25, 30), "noop": (0.0, 30)}


class RandomPolicy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        return np.array([[self.rng.integers(2), self.rng.integers(5), self.rng.integers(2)]]), None


def job(task):
    from src import eval as ev
    boss, seed, cond = task
    p, noop = CONDITIONS[cond]
    res = ev.evaluate(RandomPolicy(seed), C.BOSSES[boss]["state"], episodes=EPISODES, sticky=p, noop_max=noop,
                      seed=C.EVAL_SEED, verbose=False)
    a = res["aggregate"]
    return {"boss": boss, "seed": seed, "condition": cond, "wins": a["win"]["k"], "no_damage": a["no_damage"]["k"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()
    tasks = [(b, s, c) for b in C.BOSSES for s in SEEDS for c in CONDITIONS]
    with mp.get_context("spawn").Pool(args.procs) as pool:
        out = pool.map(job, tasks)
    os.makedirs(C.CAMPAIGN, exist_ok=True)
    with open(os.path.join(C.CAMPAIGN, "random_baseline.json"), "w") as f:
        json.dump(out, f, indent=1)
    for cond in CONDITIONS:
        wins = {b: sum(o["wins"] for o in out if o["boss"] == b and o["condition"] == cond) for b in C.BOSSES}
        print(cond, wins, f"of {len(SEEDS) * EPISODES} each")


if __name__ == "__main__":
    main()
