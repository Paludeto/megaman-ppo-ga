"""
Module: baselines.ga.deep_ga
Description: Deep GA operators (Such et al. 2017, "Deep neuroevolution"). The genome is the set of weights
             of a NatureCNN (the PPO input: 4x84x84 uint8, grayscale, frameskip 4) followed by a linear
             head with 2+5+2 logits (B, D-pad, A); the action is the per-head argmax. A genome is stored
             as a chain of seeds: theta_g = theta_(g-1) + sigma * eps(seed_g). The search loop is
             baselines.ga.fair.
"""


import argparse
import collections
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.ga import common  # noqa: E402

LOGITS = (2, 5, 2)
_CACHE_MAX = 16


def build_model():
    """NatureCNN (SB3, features_dim=512) + linear head with sum(LOGITS)=9 logits. CPU, float32."""
    import gymnasium as gym
    import torch
    from stable_baselines3.common.torch_layers import NatureCNN
    space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
    return torch.nn.Sequential(NatureCNN(space, features_dim=512), torch.nn.Linear(512, sum(LOGITS)))


def init_vector(model, seed):
    """theta_0: PyTorch default init of the architecture under torch.manual_seed(seed), flattened."""
    import torch
    torch.manual_seed(int(seed))
    fresh = build_model()
    return torch.nn.utils.parameters_to_vector(fresh.parameters()).detach().numpy().astype(np.float32)


def noise(seed, n):
    """eps(seed) ~ N(0, I) of length n, float32, from a seeded generator."""
    return np.random.default_rng(int(seed)).standard_normal(n, dtype=np.float32)


def rebuild(chain, sigma, model, cache=None):
    """
    Rebuild theta from a seed chain. `cache` (OrderedDict prefix-tuple -> vector) lets a worker start
    from the longest prefix it already holds (a child differs from its parent by one seed).
    """
    chain = tuple(int(s) for s in chain)
    start, theta = 0, None
    if cache is not None:
        for L in range(len(chain), 0, -1):
            if chain[:L] in cache:
                start, theta = L, cache[chain[:L]]
                cache.move_to_end(chain[:L])
                break
    if theta is None:
        theta = init_vector(model, chain[0])
        start = 1
        if cache is not None:
            cache[chain[:1]] = theta
    for L in range(start, len(chain)):
        theta = theta + sigma * noise(chain[L], theta.shape[0])
        if cache is not None:
            cache[chain[:L + 1]] = theta
    if cache is not None:
        while len(cache) > _CACHE_MAX:
            cache.popitem(last=False)
    return theta


def act_from_logits(logits):
    """Per-head argmax over the [B(2), D-pad(5), A(2)] logits -> np.ndarray([b, d, a])."""
    out, i = [], 0
    for n in LOGITS:
        out.append(int(logits[i:i + n].argmax()))
        i += n
    return np.array(out, dtype=np.int64)


def run_chain_episode(W, chain, sigma, sticky_p, rng, noop_max):
    """Worker entry: rebuild the policy of `chain` and run one episode."""
    import torch
    if "model" not in W:
        W["model"] = build_model().eval()
        W["cache"] = collections.OrderedDict()
    model = W["model"]
    theta = rebuild(chain, sigma, model, W["cache"])
    torch.nn.utils.vector_to_parameters(torch.from_numpy(theta), model.parameters())

    stack = np.zeros((4, 84, 84), dtype=np.uint8)   # same layout as VecFrameStack + VecTransposeImage

    def on_obs(obs, is_reset):
        if is_reset:
            stack[:] = 0
        else:
            stack[:-1] = stack[1:]
        stack[-1] = obs[..., 0]

    def act(i, obs):
        with torch.no_grad():
            logits = model(torch.from_numpy(stack).float().unsqueeze(0) / 255.0)[0].numpy()
        return act_from_logits(logits)

    return common.run_episode(W["env"], W["counter"], act, sticky_p, rng, noop_max, on_obs=on_obs)


def next_generation(chains, keys, rng, n_parents, elite=1):
    """
    Truncation selection with elitism: the top `elite` chains are copied unchanged; every other child
    is a uniformly chosen parent among the top `n_parents` plus one new mutation seed.
    Returns (children_chains, elite_source_indices).
    """
    n = len(chains)
    order = sorted(range(n), key=lambda i: keys[i], reverse=True)
    parents = order[:n_parents]
    children = [list(chains[order[j]]) for j in range(elite)]
    for _ in range(n - elite):
        p = int(rng.choice(parents))
        children.append(list(chains[p]) + [int(rng.integers(0, 2 ** 31 - 1))])
    return children, order[:elite]
