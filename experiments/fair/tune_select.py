"""
Module: experiments.fair.tune_select
Description: Picks a method's hyperparameters for one regime from its tuning runs (docs/EXPERIMENTS.md).

  Score of a tuning run (the same for the three methods; the training-time validation, never the evaluation):
    det:    (reached the no-damage stop, fewer steps to it, final deterministic validation reward);
    sticky: mean training reward of the final individual over 30 sticky episodes (GA K tuning only).
  Held-out bosses: configs are ranked per boss (ties share the mean rank); on each fold the lowest mean rank
  over its bosses wins (ties: the earlier config, the defaults come first). A boss gets the config selected
  on the OTHER fold. Writes campaign/fair/tuning/chosen_<regime>_<method>.json:
    {"selected_on": {fold: {"config_id", "args"[, "K"]}}, "by_boss": {boss: {...}}, "table": {...}}

Usage: python -m experiments.fair.tune_select det|sticky <method>
"""

import json
import os
import sys

from experiments.fair import config as C


def _load(path):
    with open(path) as f:
        return json.load(f)


def det_score_ga(tune_dir):
    s = _load(os.path.join(tune_dir, "summary.json"))
    vals = _load(os.path.join(tune_dir, "validations.json"))
    fnd = s["first_no_damage"]
    reached = fnd is not None
    return (int(reached), -fnd["cum_env_steps"] if reached else 0, vals[-1]["val_reward"] if vals else float("-inf"))


def det_score_ppo(tune_dir):
    s = _load(os.path.join(tune_dir, "summary.json"))
    reached = bool(s["stopped_no_damage"])
    return (int(reached), -s["env_steps"] if reached else 0, s["final_val_reward"])


def sticky_score(tune_dir):
    return (_load(os.path.join(tune_dir, "summary.json"))["final_estimate"]["reward"],)


def ranks(scores):
    """Rank 1 = best; tied scores share their mean rank."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for q in range(i, j + 1):
            out[order[q]] = (i + j) / 2 + 1
        i = j + 1
    return out


def select(regime, method):
    if regime == "det":
        configs = [(cid, dict(hp)) for cid, hp in C.DET_GRID[method]]
        score = det_score_ppo if method == "ppo" else det_score_ga
    else:
        assert method in C.GA_VARIANTS, "only the GAs have a sticky (K) tuning"
        configs = [(f"K{k}", None) for k in C.STICKY_K_GRID]
        score = sticky_score
    table = {cid: {"boss": {}} for cid, _ in configs}
    for boss in C.TUNE_BOSSES:
        scores = [score(C.tune_dir(regime, method, cid, boss)) for cid, _ in configs]
        for (cid, _), sc, r in zip(configs, scores, ranks(scores)):
            table[cid]["boss"][boss] = {"score": list(sc), "rank": r}

    det = _load(C.chosen_path("det", method)) if regime == "sticky" else None
    selected_on = {}
    for fold, bosses in C.FOLDS.items():
        mean_rank = [sum(table[cid]["boss"][b]["rank"] for b in bosses) / len(bosses) for cid, _ in configs]
        for (cid, _), mr in zip(configs, mean_rank):
            table[cid].setdefault("mean_rank", {})[fold] = mr
        i = min(range(len(configs)), key=lambda q: (mean_rank[q], q))
        cid, hp = configs[i]
        if regime == "det":
            selected_on[fold] = {"config_id": cid, "args": hp}
        else:   # K chosen on this fold, on top of the det config chosen on this same fold
            base = det["selected_on"][fold]
            k = int(cid[1:])
            selected_on[fold] = {"config_id": f"{base['config_id']}_K{k}", "args": base["args"], "K": k}
    by_boss = {b: selected_on[C.other_fold(C.fold_of(b))] for b in C.BOSSES}
    os.makedirs(os.path.dirname(C.chosen_path(regime, method)), exist_ok=True)
    with open(C.chosen_path(regime, method), "w") as f:
        json.dump({"selected_on": selected_on, "by_boss": by_boss, "table": table}, f, indent=1)
    for fold, e in selected_on.items():
        print(f"{regime} {method}: selected on fold {fold} {C.FOLDS[fold]} -> {e['config_id']} "
              f"(used for fold {C.other_fold(fold)})")


if __name__ == "__main__":
    select(sys.argv[1], sys.argv[2])
