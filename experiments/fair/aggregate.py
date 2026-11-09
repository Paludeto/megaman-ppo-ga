"""
Module: experiments.fair.aggregate
Description: Collects every evaluation and training record of the fair comparison and applies the
             statistics (experiments.fair.stats). Writes campaign/fair/results.json,
             campaign/fair/agarwal.json and campaign/fair/RESULTS.md (tables).

             Primary outcome: win rate under sticky actions (p=0.25, 30 episodes) per training seed, for the final model of each regime. Secondary: no-damage wins, the other perturbation conditions, the deterministic episode.

Usage: python -m experiments.fair.aggregate [--partial]   (--partial: use whatever seeds exist)
"""

import argparse
import json
import os
import zipfile

import numpy as np

from experiments.fair import config as C
from experiments.fair import stats as S

LABEL = {"ppo": "PPO", "deep-ga": "Deep GA", "ga-ol": "GA malha aberta"}
REGIME_LABEL = {"det": "treino determinístico", "sticky": "treino com sticky (p=0,25)"}


def _load(path):
    with open(path) as f:
        return json.load(f)


def ppo_steps(zip_path):
    """num_timesteps stored by SB3 in the model zip (the step at which the model was saved)."""
    with zipfile.ZipFile(zip_path) as z:
        return int(json.loads(z.read("data"))["num_timesteps"])


def training_record(regime, method, boss, seed):
    if method == "ppo":
        tag = C.ppo_run_tag(regime, boss, seed)
        d = os.path.join(C.ROOT, "models", f"{tag}_best")
        if not os.path.exists(os.path.join(d, "final_model.zip")):
            return None
        rec = {"final_steps": ppo_steps(os.path.join(d, "final_model.zip"))}
        log = os.path.join(C.CAMPAIGN, "logs", f"train__{regime}__ppo__{boss}__s{seed}.log")
        rec["stopped_early"] = os.path.exists(log) and "Stopping training: no-damage win" in open(log).read()
        return rec
    d = C.ga_run_dir(regime, method, boss, seed)
    if not os.path.exists(os.path.join(d, "summary.json")):
        return None
    s = _load(os.path.join(d, "summary.json"))
    return {"final_steps": s["env_steps"], "best_steps": (s["best"] or {}).get("cum_env_steps"),
            "stopped_early": s["stopped_by"] == "no_damage", "stopped_by": s["stopped_by"],
            "first_no_damage": s["first_no_damage"], "seconds": s["search_seconds"],
            "generations": s["generations"]}


def collect():
    rows = []
    for regime in C.REGIMES:
        for boss in C.BOSSES:
            for method in C.METHODS:
                for seed in C.seeds(method):
                    for model in C.MODELS[regime]:
                        conds = {}
                        for cond in C.conditions(method):
                            p = C.eval_path(regime, method, boss, seed, model, cond)
                            if not os.path.exists(p):
                                continue
                            a = _load(p)["aggregate"]
                            conds[cond] = {"wins": a["win"]["k"], "n": a["win"]["n"],
                                           "no_damage": a["no_damage"]["k"], "hp_lost": a["hp_lost"]["mean"],
                                           "deaths": a["deaths"], "timeouts": a["timeouts"]}
                        if conds:
                            rows.append({"regime": regime, "boss": boss, "method": method, "seed": seed,
                                         "model": model, "conditions": conds,
                                         "training": training_record(regime, method, boss, seed)})
    return rows


def scores(rows, regime, boss, method, model, cond, key="wins"):
    """Per-seed (count, episodes) of `key` in a condition, in seed order."""
    out = []
    for r in rows:
        if (r["regime"], r["boss"], r["method"], r["model"]) == (regime, boss, method, model) and cond in r["conditions"]:
            c = r["conditions"][cond]
            out.append((c[key], c["n"]))
    return out


def analyse(rows, partial):
    res = {}
    need = len(C.GA_SEEDS)
    for regime in C.REGIMES:
        for model in C.MODELS[regime]:
            for cond in ("sticky", "stickynoop", "noop"):
                for key in ("wins", "no_damage"):
                    block = {"per_boss": {}, "tests": {}}
                    for boss in C.BOSSES:
                        pb = {}
                        for method in C.METHODS:
                            kn = scores(rows, regime, boss, method, model, cond, key)
                            if kn:
                                x = [k / n for k, n in kn]
                                pb[method] = {"n_seeds": len(x), "per_seed": x, "per_seed_k": [k for k, _ in kn],
                                              "episodes": kn[0][1], "mean": float(np.mean(x)),
                                              "ci95": list(S.bootstrap_ci(x))}
                        block["per_boss"][boss] = pb
                    for other in C.GA_VARIANTS:
                        pairs, bosses, ps, diffs = [], [], [], []
                        for boss in C.BOSSES:
                            pb = block["per_boss"][boss]
                            if "ppo" not in pb or other not in pb:
                                continue
                            a, b = pb["ppo"]["per_seed"], pb[other]["per_seed"]
                            if not partial and (len(a) < need or len(b) < need):
                                continue
                            if len(a) < 2 or len(b) < 2:
                                continue
                            d, p = S.perm_test_exact(a, b)
                            pairs.append((a, b))
                            bosses.append(boss)
                            ps.append(p)
                            diffs.append(d)
                        if not pairs:
                            continue
                        adj = S.holm(ps)
                        overall = S.perm_test_stratified(pairs) if len(pairs) > 1 else (diffs[0], ps[0])
                        block["tests"][f"ppo_vs_{other}"] = {
                            "per_boss": {b: {"diff": d, "p": p, "p_holm": h}
                                         for b, d, p, h in zip(bosses, diffs, ps, adj)},
                            "overall": {"diff": overall[0], "p": overall[1], "bosses": bosses}}
                    # Holm over the two primary comparisons (PPO vs each GA)
                    keys = sorted(block["tests"])
                    for k, h in zip(keys, S.holm([block["tests"][k]["overall"]["p"] for k in keys])):
                        block["tests"][k]["overall"]["p_holm"] = h
                    res[f"{regime}/{model}/{cond}/{key}"] = block
    return res


def fmt_pct(x):
    return f"{100 * x:.0f}%"


def write_markdown(rows, res, path):
    L = ["# Comparação justa PPO × GA — resultados (gerado por experiments/fair/aggregate.py)", ""]
    for regime in C.REGIMES:
        model = C.PRIMARY_MODEL[regime]
        for key, what in (("wins", "Vitórias"), ("no_damage", "Vitórias sem dano")):
            block = res.get(f"{regime}/{model}/sticky/{key}")
            if not block:
                continue
            L += [f"## {what} sob sticky — {REGIME_LABEL[regime]}, modelo `{model}`", "",
                  "Média entre sementes [IC 95% bootstrap]; entre parênteses, as vitórias de cada semente em 30 "
                  "episódios. p = permutação exata (Holm sobre os 6 chefes).", "",
                  "| Chefe | PPO | Deep GA | GA malha aberta | p PPO×Deep GA (Holm) | p PPO×GA-OL (Holm) |",
                  "|---|---|---|---|---|---|"]
            for boss in C.BOSSES:
                pb = block["per_boss"][boss]
                cells = []
                for m in C.METHODS:
                    if m in pb:
                        e = pb[m]
                        seeds = " ".join(str(k) for k in e["per_seed_k"])
                        cells.append(f"{fmt_pct(e['mean'])} [{fmt_pct(e['ci95'][0])}–{fmt_pct(e['ci95'][1])}] ({seeds})")
                    else:
                        cells.append("—")
                tcells = []
                for other in C.GA_VARIANTS:
                    t = block["tests"].get(f"ppo_vs_{other}", {}).get("per_boss", {}).get(boss)
                    tcells.append("—" if t is None else f"{t['p']:.3f} ({t['p_holm']:.3f})")
                L.append(f"| {boss} | " + " | ".join(cells + tcells) + " |")
            for other in C.GA_VARIANTS:
                t = block["tests"].get(f"ppo_vs_{other}")
                if t:
                    L.append(f"\nGeral PPO × {LABEL[other]}: diferença média {100 * t['overall']['diff']:+.1f} p.p., "
                             f"p = {t['overall']['p']:.2g}, p Holm = {t['overall']['p_holm']:.2g} (permutação estratificada, "
                             f"{len(t['overall']['bosses'])} chefes).")
            L.append("")
    # secondary: PPO greedy (argmax, the policy compared with the GAs) vs sampled, both under sticky
    L += ["## PPO guloso × amostrado sob sticky (secundário, descritivo)", "",
          "O PPO é comparado aos GAs com a política gulosa (argmax), mas foi treinado amostrando. Média entre "
          "sementes da fração de vitórias em 30 episódios sob sticky.", "",
          "| Regime | Chefe | PPO guloso | PPO amostrado |", "|---|---|---|---|"]
    for regime in C.REGIMES:
        for boss in C.BOSSES:
            cells = []
            for cond in ("sticky", "stickystoch"):
                kn = scores(rows, regime, boss, "ppo", C.PRIMARY_MODEL[regime], cond)
                cells.append(f"{fmt_pct(np.mean([k / n for k, n in kn]))} (n={len(kn)})" if kn else "—")
            L.append(f"| {regime} | {boss} | " + " | ".join(cells) + " |")
    L.append("")
    # hyperparameters chosen (on the other fold of bosses)
    L += ["## Hiperparâmetros escolhidos (validação cruzada em 2 folds de chefes)", "",
          "Cada chefe usa a configuração escolhida no OUTRO fold: " +
          "; ".join(f"fold {f} = {', '.join(bs)}" for f, bs in C.FOLDS.items()) + ".", "",
          "| Regime | Método | " + " | ".join(C.BOSSES) + " |", "|---|---|" + "---|" * len(C.BOSSES)]
    for regime in C.REGIMES:
        for m in C.METHODS:
            cells = []
            for b in C.BOSSES:
                ch = C.chosen_config(regime, m, b) or (C.chosen_config("det", m, b) if m == "ppo" else None)
                cells.append(ch["config_id"] if ch else "—")
            L.append(f"| {regime} | {LABEL[m]} | " + " | ".join(cells) + " |")
    L.append("")
    # training cost
    L += ["## Custo de treino (passos do modelo final; mediana entre sementes)", "",
          "| Regime | Chefe | Orçamento | PPO | Deep GA | GA malha aberta | Parou por abate sem dano (PPO / Deep GA / GA-OL) |",
          "|---|---|---|---|---|---|---|"]
    for regime in C.REGIMES:
        for boss in C.BOSSES:
            cells, stops = [], []
            for m in C.METHODS:
                tr = [r["training"] for r in rows if (r["regime"], r["boss"], r["method"], r["model"]) ==
                      (regime, boss, m, "final") and r["training"]]
                cells.append(f"{np.median([t['final_steps'] for t in tr]) / 1e6:.1f}M" if tr else "—")
                stops.append(f"{sum(t['stopped_early'] for t in tr)}/{len(tr)}" if tr else "—")
            L.append(f"| {regime} | {boss} | {C.BOSSES[boss]['cap'] / 1e6:g}M | " + " | ".join(cells) +
                     f" | {' / '.join(stops)} |")
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------------------------
# Learning curves: sticky win rate at 25/50/75% of the budget and the final model
# ----------------------------------------------------------------------------------------------
def curves(rows):
    """{regime: {boss: {method: [{"step", "n_seeds", "per_seed", "mean", "ci95", "fallbacks"}]}}}; the last point
    (the budget) is the final model."""
    out = {}
    for regime in C.REGIMES:
        for boss in C.BOSSES:
            for method in C.METHODS:
                pts = []
                for step in C.curve_steps(boss):
                    x, fb = [], 0
                    for seed in C.seeds(method):
                        pth = C.eval_path(regime, method, boss, seed, C.snapshot_name(step), C.CURVE_CONDITION)
                        if not os.path.exists(pth):
                            continue
                        d = _load(pth)
                        x.append(d["aggregate"]["win"]["k"] / d["aggregate"]["win"]["n"])
                        if method == "ppo":
                            fb += os.path.exists(os.path.join(C.ROOT, "models", f"{C.ppo_run_tag(regime, boss, seed)}_best",
                                                              f"snapshot_{step}.fallback"))
                        else:
                            fb += bool(d["config"].get("snapshot_fallback"))
                    if x:
                        pts.append({"step": step, "n_seeds": len(x), "per_seed": x, "mean": float(np.mean(x)),
                                    "ci95": list(S.bootstrap_ci(x)), "fallbacks": fb})
                x = [k / n for k, n in scores(rows, regime, boss, method, "final", C.CURVE_CONDITION)]
                if x:
                    pts.append({"step": C.BOSSES[boss]["cap"], "n_seeds": len(x), "per_seed": x,
                                "mean": float(np.mean(x)), "ci95": list(S.bootstrap_ci(x)), "fallbacks": None})
                if pts:
                    out.setdefault(regime, {}).setdefault(boss, {})[method] = pts
    return out


def curves_markdown(cv):
    L = ["## Curvas de aprendizado: vitórias sob sticky ao longo do treino (o último ponto é o modelo final)", "",
         "Média entre sementes. Entre colchetes, quantas sementes já tinham parado por abate sem dano e usam "
         "a política em que pararam.", ""]
    for regime in C.REGIMES:
        steps = sorted({p["step"] for b in cv.get(regime, {}).values() for pts in b.values() for p in pts})
        L += [f"### {REGIME_LABEL[regime].capitalize()}", "",
              "| Chefe | Método | " + " | ".join(f"{st / 1e6:g}M" for st in steps) + " |",
              "|---|---|" + "---|" * len(steps)]
        for boss in C.BOSSES:
            for method in C.METHODS:
                pts = cv.get(regime, {}).get(boss, {}).get(method)
                if not pts:
                    continue
                by = {p["step"]: p for p in pts}
                cells = []
                for f in steps:
                    p = by.get(f)
                    cells.append("—" if p is None else fmt_pct(p["mean"]) + (f" [{p['fallbacks']}]" if p["fallbacks"] else ""))
                L.append(f"| {boss} | {LABEL[method]} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def early_budget_tests(cv, steps=(500_000, 1_000_000)):
    """Exploratory, uncorrected: stratified permutation test of PPO vs Deep GA at early budgets (sticky regime)."""
    out = {}
    if "sticky" not in cv:
        return out
    for step in steps:
        pairs = []
        for boss in C.BOSSES:
            by = {m: next((p for p in cv["sticky"][boss][m] if p["step"] == step), None) for m in ("ppo", "deep-ga")}
            if all(by.values()):
                pairs.append((by["ppo"]["per_seed"], by["deep-ga"]["per_seed"]))
        if pairs:
            diff, p = S.perm_test_stratified(pairs)
            out[str(step)] = {"diff_ppo_minus_deep_ga": diff, "p": p}
    return out


def agarwal(res):
    """Aggregate metrics of Agarwal et al. 2021 over the pooled (boss x seed) scores, stratified bootstrap CIs."""
    out = {}
    for regime in C.REGIMES:
        model = C.PRIMARY_MODEL[regime]
        for cond in ("sticky", "stickynoop", "noop"):
            block = res.get(f"{regime}/{model}/{cond}/wins")
            if not block:
                continue
            by = {m: [block["per_boss"][b][m]["per_seed"] for b in C.BOSSES if m in block["per_boss"][b]]
                  for m in C.METHODS}
            if any(len(v) != len(C.BOSSES) for v in by.values()):
                continue                       # every method needs every boss
            e = {"methods": {}, "vs_ppo": {}}
            for m in C.METHODS:
                e["methods"][m] = {"iqm": S.aggregate_ci(by[m], S.iqm), "mean": S.aggregate_ci(by[m], np.mean)}
            for other in C.GA_VARIANTS:
                e["vs_ppo"][other] = {"iqm_diff": S.difference_ci(by["ppo"], by[other], S.iqm),
                                      "prob_improvement": S.prob_improvement_ci(by["ppo"], by[other])}
            out[f"{regime}/{model}/{cond}"] = e
    return out


def agarwal_markdown(ag):
    def ci(t, pct=True):
        f = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.2f}")
        return f"{f(t[0])} [{f(t[1])}–{f(t[2])}]"
    L = ["## Métricas agregadas (Agarwal et al., NeurIPS 2021)", "",
         "Sobre os escores de todas as execuções (chefe × semente; escore = vitórias/30). IQM = média interquartil. "
         "Entre colchetes, IC 95% por bootstrap estratificado por chefe (10.000 reamostragens). "
         "P(PPO > X) = probabilidade de uma execução do PPO superar uma de X no mesmo chefe (0,5 = indistinguíveis).", ""]
    names = {"sticky": "sticky", "stickynoop": "sticky + no-op", "noop": "só no-op"}
    for key, e in ag.items():
        regime, cond = key.split("/")[0], key.split("/")[-1]
        treino = "treino determinístico" if regime == "det" else "treino com sticky (p=0,25)"
        L += [f"### {treino.capitalize()} — condição {names.get(cond, cond)}", "", "| Método | IQM | Média |", "|---|---|---|"]
        for m in C.METHODS:
            L.append(f"| {LABEL[m]} | {ci(e['methods'][m]['iqm'])} | {ci(e['methods'][m]['mean'])} |")
        L += ["", "| Comparação | Diferença de IQM (PPO − X) | P(PPO > X) |", "|---|---|---|"]
        for other in C.GA_VARIANTS:
            v = e["vs_ppo"][other]
            d = v["iqm_diff"]
            L.append(f"| PPO × {LABEL[other]} | {100 * d[0]:+.0f} p.p. [{100 * d[1]:+.0f} – {100 * d[2]:+.0f}] | "
                     f"{ci(v['prob_improvement'], pct=False)} |")
        L.append("")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="use whatever seeds exist")
    args = ap.parse_args(argv)
    rows = collect()
    res = analyse(rows, args.partial)
    cv = curves(rows)
    early = early_budget_tests(cv)
    suffix = "_partial" if args.partial else ""
    with open(os.path.join(C.CAMPAIGN, f"results{suffix}.json"), "w") as f:
        json.dump({"rows": rows, "analysis": res, "curves": cv, "early_budget_tests": early}, f, indent=1)
    md = os.path.join(C.CAMPAIGN, f"RESULTS{suffix}.md")
    write_markdown(rows, res, md)
    ag = agarwal(res)
    with open(os.path.join(C.CAMPAIGN, f"agarwal{suffix}.json"), "w") as f:
        json.dump(ag, f, indent=1)
    with open(md, "a") as f:
        f.write("\n" + "\n".join(agarwal_markdown(ag)) + "\n")
    with open(md, "a") as f:
        f.write("\n" + "\n".join(curves_markdown(cv)) + "\n")
    print(f"{len(rows)} model records; wrote results{suffix}.json, agarwal{suffix}.json, RESULTS{suffix}.md")


if __name__ == "__main__":
    main()
