"""
Module: experiments.fair.stats
Description: Statistics of the fair comparison, fixed before any result:
  - unit of analysis = the training seed (5 per method, boss and regime); a seed's score is its win rate
    over the 30 evaluation episodes of a condition;
  - per boss: mean over seeds with a 95% percentile bootstrap interval over seeds (10,000 resamples) and
    the EXACT two-sided permutation test of the difference of means (all C(10,5) = 252 splits);
  - Holm correction over the 6 bosses, per pair of methods and regime;
  - over all bosses: stratified permutation test (labels permuted within each boss) of the mean over
    bosses of the per-boss difference, 100,000 Monte Carlo permutations.
"""

import itertools

import numpy as np

BOOT = 10_000
MC = 100_000
RNG_SEED = 20261002


def bootstrap_ci(x, n_boot=BOOT, seed=RNG_SEED):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def perm_test_exact(a, b):
    """Two-sided exact permutation test of mean(a) - mean(b). Returns (difference, p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n, obs = len(a), a.mean() - b.mean()
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        d = pooled[mask].mean() - pooled[~mask].mean()
        count += abs(d) >= abs(obs) - 1e-12
        total += 1
    return float(obs), count / total


def perm_test_stratified(pairs, n_perm=MC, seed=RNG_SEED):
    """
    pairs = [(a_boss, b_boss), ...]. Statistic = mean over bosses of (mean(a) - mean(b)); labels are
    permuted within each boss. Two-sided Monte Carlo p with the +1 correction.
    """
    rng = np.random.default_rng(seed)
    pooled = [np.concatenate([np.asarray(a, float), np.asarray(b, float)]) for a, b in pairs]
    sizes = [len(a) for a, _ in pairs]
    obs = float(np.mean([np.mean(a) - np.mean(b) for a, b in pairs]))
    stats = np.zeros(n_perm)
    for k, (p, n) in enumerate(zip(pooled, sizes)):
        perm = np.argsort(rng.random((n_perm, len(p))), axis=1)
        x = p[perm]
        stats += x[:, :n].mean(axis=1) - x[:, n:].mean(axis=1)
    stats /= len(pairs)
    p = (1 + np.sum(np.abs(stats) >= abs(obs) - 1e-12)) / (n_perm + 1)
    return obs, float(p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


# ----------------------------------------------------------------------------------------------
# Aggregate metrics of Agarwal et al. (NeurIPS 2021, "Deep RL at the Edge of the Statistical Precipice"):
# interquartile mean (IQM), mean and probability of improvement over the pooled (boss x seed) scores, with
# 95% intervals from a bootstrap STRATIFIED by boss (seeds resampled with replacement within each boss).
# ----------------------------------------------------------------------------------------------
def iqm(x):
    """Mean of the middle 50% of the scores (25% trimmed from each end, fractional at the edges)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    lo, hi = 0.25 * n, 0.75 * n
    w = np.clip(np.minimum(np.arange(1, n + 1), hi) - np.maximum(np.arange(n), lo), 0, None)
    return float(np.sum(w * x) / np.sum(w))


def prob_improvement(a_by_task, b_by_task):
    """P(X > Y) + 0.5 P(X = Y), X a run of A and Y a run of B on the same task, averaged over tasks."""
    ps = []
    for a, b in zip(a_by_task, b_by_task):
        a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None, :]
        ps.append(float(np.mean((a > b) + 0.5 * (a == b))))
    return float(np.mean(ps))


def _strat_resample(by_task, rng):
    return [np.asarray(t, float)[rng.integers(0, len(t), len(t))] for t in by_task]


def aggregate_ci(by_task, fn, n_boot=BOOT, seed=RNG_SEED):
    """Point estimate and stratified-bootstrap 95% CI of fn(pooled scores)."""
    rng = np.random.default_rng(seed)
    point = fn(np.concatenate([np.asarray(t, float) for t in by_task]))
    boots = [fn(np.concatenate(_strat_resample(by_task, rng))) for _ in range(n_boot)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def difference_ci(a_by_task, b_by_task, fn, n_boot=BOOT, seed=RNG_SEED):
    """fn(A) - fn(B) with a stratified bootstrap (A and B resampled independently) 95% CI."""
    rng = np.random.default_rng(seed)
    pool = lambda ts: np.concatenate([np.asarray(t, float) for t in ts])  # noqa: E731
    point = fn(pool(a_by_task)) - fn(pool(b_by_task))
    boots = [fn(pool(_strat_resample(a_by_task, rng))) - fn(pool(_strat_resample(b_by_task, rng)))
             for _ in range(n_boot)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def prob_improvement_ci(a_by_task, b_by_task, n_boot=BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    point = prob_improvement(a_by_task, b_by_task)
    boots = [prob_improvement(_strat_resample(a_by_task, rng), _strat_resample(b_by_task, rng))
             for _ in range(n_boot)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
