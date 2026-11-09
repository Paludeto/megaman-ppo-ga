"""
Module: experiments.fair.jobs
Description: Job graph of the fair GA x PPO comparison (docs/EXPERIMENTS.md):
               tuning of the three methods (det, 4 configs x 6 bosses) -> selection per method on the other
               fold of bosses -> GA K tuning (sticky) -> selection -> training (PPO and both GAs, both regimes,
               5 seeds) -> evaluation through
               src/eval.py -> aggregation.
             Every job is a shell command built when the job STARTS, so the GA jobs read the
             hyperparameters selected by the tuning stage at that moment. The list order is the
             scheduler's priority (evaluations follow their training job).
"""

import json
import os
import shlex
from dataclasses import dataclass

from experiments.fair import config as C


@dataclass
class Job:
    id: str
    kind: str          # tune | select | ppo | ga | eval | curve | aggregate
    deps: tuple
    build: object      # () -> shell command

    @property
    def slots(self):
        return C.SLOTS["ppo" if self.kind == "ppo" else "ga" if self.kind in ("ga", "tune")
                       else "eval" if self.kind in ("eval", "curve") else self.kind]


def _q(path):
    return shlex.quote(str(path))


def ga_cli_args(variant, hp):
    keys = ("sigma", "pop", "parents", "elite") if variant == "deep-ga" else ("p_mut", "pop", "k", "elite",
                                                                              "trunc", "p_cross")
    return " ".join(f"--{k.replace('_', '-')} {hp[k]}" for k in keys)


def chosen(regime, method, boss):
    """Config chosen for `boss` (on the other fold): {"config_id", "args"[, "K"]}."""
    ch = C.chosen_config(regime, method, boss)
    if ch is None:
        raise RuntimeError(f"no {regime} selection for {method}/{boss} yet")
    return ch


def selected_on(regime, method, fold):
    """Config selected ON `fold` (used for the other fold's bosses)."""
    with open(C.chosen_path(regime, method)) as f:
        return json.load(f)["selected_on"][fold]


def _ga_cmd(variant, boss, seed, out_dir, hp, regime, k_eps, extra="", snapshots=False):
    sticky = (f" --train-sticky {C.STICKY_P} --episodes-per-eval {k_eps} --no-early-stop"
              if regime == "sticky" else "")
    # a campaign run (snapshots) is retrained from scratch: drop evaluations of any earlier run of the cell
    ev = os.path.join(C.CAMPAIGN, "eval", regime, variant, boss, f"s{seed}")
    return ((f"rm -rf {_q(ev)} && " if snapshots else "") + f"rm -rf {_q(out_dir)} && mkdir -p {_q(out_dir)} && "
            f"{C.PY} -m baselines.ga.fair --variant {variant} --boss {boss} "
            f"--max-env-steps {C.BOSSES[boss]['cap']} --seed {seed} --out-dir {_q(out_dir)} --procs 12 "
            f"{ga_cli_args(variant, hp)}{sticky}{extra}"
            + (" --snapshot-at " + " ".join(str(s) for s in C.curve_steps(boss)) if snapshots else ""))


def _ppo_train(tag, boss, seed, hp, regime, timesteps):
    sticky = f" --train-sticky {C.STICKY_P} --no-early-stop" if regime == "sticky" else ""
    return (f"rm -rf models/{tag}_best logs/{tag} checkpoints/{tag} && "
            f"{C.PY} src/train.py --tag {tag} --state {C.BOSSES[boss]['state']} --seed {seed} "
            f"--timesteps {timesteps} --lr-horizon {C.LR_HORIZON} --lr {hp['lr']} --ent-coef {hp['ent_coef']}{sticky}")


def _ppo_complete(tag, boss):
    d = os.path.join(C.ROOT, "models", f"{tag}_best")
    return os.path.exists(os.path.join(d, "final_model.zip")) and all(
        os.path.exists(os.path.join(d, f"snapshot_{st}_model.zip")) for st in C.curve_steps(boss))


def _ppo_cmd(regime, boss, seed):
    """Campaign PPO run with the config chosen for this boss. A finished run of the same config (the default
    one, from before the tuning existed) is reused."""
    ch = chosen("det", "ppo", boss)
    tag = C.ppo_tag(regime, boss, seed, ch["config_id"])
    if _ppo_complete(tag, boss):
        return f"echo reuse finished run {tag}"
    ev = os.path.join(C.CAMPAIGN, "eval", regime, "ppo", boss, f"s{seed}")
    return (f"rm -rf {_q(ev)} && " + _ppo_train(tag, boss, seed, ch["args"], regime, C.BOSSES[boss]["cap"])
            + f" && {_keep_snapshots(tag, boss)} && rm -rf checkpoints/{tag}")


def _ppo_tune_cmd(cid, boss, hp):
    tag = C.ppo_tune_tag(cid, boss)
    out = C.tune_dir("det", "ppo", cid, boss)
    return (f"mkdir -p {_q(out)} && " + _ppo_train(tag, boss, C.TUNE_SEED, hp, "det", C.BOSSES[boss]["cap"])
            + f" && {C.PY} -m experiments.fair.ppo_summary {tag} {boss} {_q(os.path.join(out, 'summary.json'))}"
            + f" && rm -rf checkpoints/{tag}")


def _keep_snapshots(tag, boss):
    """
    Move the CheckpointCallback files of the learning-curve steps next to the models
    (models/<tag>_best/snapshot_<step>_model.zip). A step the run never reached (it stopped early) gets
    a copy of the final model, the policy the run ended with, and a .fallback marker.
    """
    d = f"models/{tag}_best"
    parts = []
    for st in C.curve_steps(boss):
        src, dst = f"checkpoints/{tag}/{tag}_{st}_steps.zip", f"{d}/snapshot_{st}_model.zip"
        parts.append(f"if [ -f {src} ]; then mv {src} {dst}; else cp {d}/final_model.zip {dst} && "
                     f"touch {d}/snapshot_{st}.fallback; fi")
    return "{ " + "; ".join(parts) + "; }"


def _eval_cmd(regime, method, boss, seed, model):
    cmds = []
    for cond in C.conditions(method):
        out = C.eval_path(regime, method, boss, seed, model, cond)
        if os.path.exists(out):
            continue
        p, noop, n, stoch = C.CONDITIONS[cond]
        if method == "ppo":
            zip_path = os.path.join("models", f"{C.ppo_run_tag(regime, boss, seed)}_best", f"{model}_model.zip")
            cmds.append(f"{C.PY} src/eval.py --model {zip_path} --state {C.BOSSES[boss]['state']} "
                        f"--episode-frames {C.EPISODE_FRAMES} --seed {C.EVAL_SEED} --device cpu --episodes {n} "
                        f"{'--stochastic' if stoch else '--deterministic'} --sticky {p} --noop-max {noop} "
                        f"--json {_q(out)}")
        else:
            cmds.append(f"{C.PY} -m baselines.ga.evaluate --run {_q(C.ga_run_dir(regime, method, boss, seed))} "
                        f"--which {model} --episodes {n} --sticky {p} --noop-max {noop} --seed {C.EVAL_SEED} "
                        f"--json {_q(out)}")
    return " && ".join(cmds) if cmds else "true"


def _curve_cmd(regime, method, boss, seed, step):
    out = C.eval_path(regime, method, boss, seed, C.snapshot_name(step), C.CURVE_CONDITION)
    if os.path.exists(out):
        return "true"
    p, noop, n, _ = C.CONDITIONS[C.CURVE_CONDITION]
    if method == "ppo":
        zip_path = os.path.join("models", f"{C.ppo_run_tag(regime, boss, seed)}_best", f"snapshot_{step}_model.zip")
        return (f"{C.PY} src/eval.py --model {zip_path} --state {C.BOSSES[boss]['state']} "
                f"--episode-frames {C.EPISODE_FRAMES} --seed {C.EVAL_SEED} --device cpu --episodes {n} "
                f"--deterministic --sticky {p} --noop-max {noop} --json {_q(out)}")
    return (f"{C.PY} -m baselines.ga.evaluate --run {_q(C.ga_run_dir(regime, method, boss, seed))} "
            f"--which {C.snapshot_name(step)} --episodes {n} --sticky {p} --noop-max {noop} "
            f"--seed {C.EVAL_SEED} --json {_q(out)}")


def train_id(regime, method, boss, seed):
    return f"train/{regime}/{method}/{boss}/s{seed}"


def build_jobs():
    jobs = []

    def evals_of(regime, method, boss, seed):
        tid = train_id(regime, method, boss, seed)
        for model in C.MODELS[regime]:
            jobs.append(Job(f"eval/{regime}/{method}/{boss}/s{seed}/{model}", "eval", (tid,),
                            lambda r=regime, m=method, b=boss, s=seed, mo=model: _eval_cmd(r, m, b, s, mo)))
        for step in C.curve_steps(boss):
            jobs.append(Job(f"curve/{regime}/{method}/{boss}/s{seed}/{step}", "curve", (tid,),
                            lambda r=regime, m=method, b=boss, s=seed, st=step: _curve_cmd(r, m, b, s, st)))

    # 1. Tuning, det regime, the three methods: every config on every boss (PPO first: its path is the longest)
    tune_det = {m: [] for m in C.METHODS}
    for method in C.METHODS:
        for cid, hp in C.DET_GRID[method]:
            for boss in C.TUNE_BOSSES:
                jid = f"tune/det/{method}/{cid}/{boss}"
                tune_det[method].append(jid)
                if method == "ppo":
                    jobs.append(Job(jid, "ppo", (), lambda b=boss, h=hp, c=cid: _ppo_tune_cmd(c, b, h)))
                else:
                    jobs.append(Job(jid, "tune", (), lambda v=method, b=boss, h=hp, c=cid: _ga_cmd(
                        v, b, C.TUNE_SEED, C.tune_dir("det", v, c, b), h, "det", 1)))
    for method in C.METHODS:
        jobs.append(Job(C.select_id("det", method), "select", tuple(tune_det[method]),
                        lambda m=method: f"{C.PY} -m experiments.fair.tune_select det {m}"))

    # 2. GA K tuning, sticky regime: on each tuning boss, on top of the det config selected on ITS fold
    if "sticky" in C.REGIMES:
        for variant in C.GA_VARIANTS:
            tune_sticky = []
            for k in C.STICKY_K_GRID:
                for boss in C.TUNE_BOSSES:
                    jid = f"tune/sticky/{variant}/K{k}/{boss}"
                    tune_sticky.append(jid)
                    jobs.append(Job(jid, "tune", (C.select_id("det", variant),), lambda v=variant, b=boss, k=k: _ga_cmd(
                        v, b, C.TUNE_SEED, C.tune_dir("sticky", v, f"K{k}", b),
                        selected_on("det", v, C.fold_of(b))["args"], "sticky", k,
                        extra=f" --final-estimate-episodes {C.TUNE_ESTIMATE_EPISODES}")))
            jobs.append(Job(C.select_id("sticky", variant), "select", tuple(tune_sticky),
                            lambda v=variant: f"{C.PY} -m experiments.fair.tune_select sticky {v}"))

    # 3. Campaign: seed by seed (complete seed sets appear early), both regimes, every boss and method
    for i in range(len(C.GA_SEEDS)):
        for regime in C.REGIMES:
            for boss in C.BOSSES:
                ppo_seed = C.PPO_SEEDS[i]
                jobs.append(Job(train_id(regime, "ppo", boss, ppo_seed), "ppo", (C.select_id("det", "ppo"),),
                                lambda r=regime, b=boss, s=ppo_seed: _ppo_cmd(r, b, s)))
                evals_of(regime, "ppo", boss, ppo_seed)
                for variant in C.GA_VARIANTS:
                    seed = C.GA_SEEDS[i]
                    jobs.append(Job(train_id(regime, variant, boss, seed), "ga", (C.select_id(regime, variant),),
                                    lambda r=regime, v=variant, b=boss, s=seed: _ga_cmd(
                                        v, b, s, C.ga_run_dir(r, v, b, s), chosen(r, v, b)["args"], r,
                                        chosen(r, v, b).get("K", 1), snapshots=True)))
                    evals_of(regime, variant, boss, seed)

    # 4. Aggregation
    jobs.append(Job("aggregate", "aggregate", tuple(j.id for j in jobs if j.kind in ("eval", "curve")),
                    lambda: f"{C.PY} -m experiments.fair.aggregate"))
    ids = [j.id for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    known = set(ids)
    assert all(d in known for j in jobs for d in j.deps), "unknown dependency"
    return jobs
