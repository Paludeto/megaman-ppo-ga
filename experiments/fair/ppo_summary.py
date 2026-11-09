"""
Module: experiments.fair.ppo_summary
Description: Summary of a PPO tuning run in the format experiments.fair.tune_select reads for every method:
               stopped_no_damage  the run stopped at a no-damage validation before the budget (det regime: the
                                  only reason a run ends before the budget);
               env_steps          agent steps of its final model;
               final_val_reward   its last deterministic validation reward on the evaluation task (TensorBoard
                                  eval/mean_reward of ValidationCallback, every 100k steps).

Usage: python -m experiments.fair.ppo_summary <tag> <boss> <out.json>
"""

import glob
import json
import os
import sys
import zipfile

from experiments.fair import config as C


def summarize(tag, boss):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    final = os.path.join(C.ROOT, "models", f"{tag}_best", "final_model.zip")
    with zipfile.ZipFile(final) as z:
        steps = int(json.loads(z.read("data"))["num_timesteps"])
    dirs = sorted(glob.glob(os.path.join(C.ROOT, "logs", tag, "PPO_*")))
    ea = EventAccumulator(dirs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    vals = ea.Scalars("eval/mean_reward")
    return {"tag": tag, "boss": boss, "env_steps": steps, "budget": C.BOSSES[boss]["cap"],
            "stopped_no_damage": steps < C.BOSSES[boss]["cap"],
            "final_val_reward": float(vals[-1].value), "n_validations": len(vals)}


def main(tag, boss, out):
    s = summarize(tag, boss)
    with open(out, "w") as f:
        json.dump(s, f, indent=1)
    print(s)


if __name__ == "__main__":
    main(*sys.argv[1:])
