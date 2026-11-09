"""
Module: baselines.ga.evaluate
Description: Evaluates a GA individual saved by baselines.ga.fair (best.json or final.json) with
             src.eval.evaluate(): the same env stack, perturbation protocol (sticky actions, no-op
             starts) and per-episode seeds that src/eval.py uses for PPO. The CLI mirrors src/eval.py and
             the JSON has the same format, so PPO and GA results are read by the same code.

Usage:
  python -m baselines.ga.evaluate --run DIR --which best --sticky 0.25 --episodes 30 --seed 123 --json out.json
"""

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.ga import common  # noqa: E402
from baselines.ga.policy import load_run_policy  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a GA individual with src/eval.py's protocol")
    ap.add_argument("--run", required=True, help="baselines.ga.fair output directory")
    ap.add_argument("--which", required=True, help="best, final or snapshot_<step>")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--sticky", type=float, default=0.0)
    ap.add_argument("--noop-max", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(1)
    from src import eval as ev

    with open(os.path.join(args.run, "config.json")) as f:
        cfg = json.load(f)
    policy, record = load_run_policy(args.run, args.which)
    boss = common.BOSSES[cfg["boss"]]
    print(f"Evaluating {record['variant']} {args.which} (gen {record['generation']}) on {boss['state']}: "
          f"{args.episodes} episodes (sticky={args.sticky}, noop_max={args.noop_max}, seed={args.seed})")
    result = ev.evaluate(policy, boss["state"], episodes=args.episodes, deterministic=True, sticky=args.sticky,
                         noop_max=args.noop_max, episode_frames=boss["frames"],
                         seed=args.seed)
    result["config"].update(model=os.path.join(args.run, f"{args.which}.json"), variant=record["variant"],
                            generation=record["generation"], cum_env_steps=record["cum_env_steps"],
                            snapshot_fallback=record.get("snapshot_fallback", False))
    print("-" * 72)
    print(ev.format_summary(result["aggregate"]))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(ev.json_safe(result), f, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    main()
