"""
Module: baselines.ga.ga_ol
Description: Operators of GA-OL, the open-loop genetic algorithm. The genome is a sequence of
             MultiDiscrete([2,5,2]) actions in blocks of k agent steps; there is NO perception: the
             individual replays the same actions regardless of what happens on screen (the "Brute" idea
             of Machado et al. 2018). The search loop is baselines.ga.fair.
"""


import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.ga import common  # noqa: E402


def episode_steps(boss):
    """Agent steps of the TimeLimit (max_episode_frames // frameskip)."""
    return common.BOSSES[boss]["frames"] // common.FRAMESKIP


def next_population(pop, keys, rng, elite, trunc, p_cross, p_mut):
    """
    Build the next generation. `pop` is an (N, L) int array, `keys` the fitness keys (larger is
    better). The top `elite` individuals are copied unchanged; every other child is produced from two
    parents drawn uniformly from the top `trunc` fraction (2-point crossover with prob p_cross, then
    per-gene resampling with prob p_mut). Returns (children, elite_indices) where `children` has N rows
    and the first `elite` rows are the elites.
    """
    n, length = pop.shape
    order = sorted(range(n), key=lambda i: keys[i], reverse=True)
    n_parents = max(2, int(round(trunc * n)))
    parents = order[:n_parents]
    children = np.empty_like(pop)
    for j in range(elite):
        children[j] = pop[order[j]]
    for j in range(elite, n):
        a, b = rng.choice(parents, size=2, replace=True)
        child = pop[a].copy()
        if rng.random() < p_cross:
            c1, c2 = sorted(rng.integers(0, length + 1, size=2))
            child[c1:c2] = pop[b][c1:c2]
        mask = rng.random(length) < p_mut
        child[mask] = rng.integers(0, common.N_GENES, size=int(mask.sum()))
        children[j] = child
    return children, order[:elite]
