"""
Module: baselines.ga.policy
Description: The GA individuals as policies with SB3's predict() interface, so that src/eval.py evaluates
             them with exactly the code, wrapper stack and per-episode seeds it uses for PPO.

  - DeepGAPolicy: NatureCNN + Linear(512, 9) rebuilt from a seed chain (baselines.ga.deep_ga); the
    action is the per-head argmax, i.e. what PPO's predict(deterministic=True) does with the same
    network. It reads the (n, 4, 84, 84) uint8 stack that VecFrameStack + VecTransposeImage produce.
  - OpenLoopPolicy: a GA-OL genome (one gene per k agent steps); it ignores the observation.
    src/eval.py calls reset_episode() after every reset and predict() once per policy step (no-op
    starts excluded), the same step indexing as baselines.ga.common.run_episode.
"""

import json
import os

import numpy as np

from baselines.ga import common, deep_ga


class DeepGAPolicy:
    def __init__(self, chain, sigma):
        import torch
        self._torch = torch
        self.model = deep_ga.build_model().eval()
        theta = deep_ga.rebuild(chain, sigma, self.model)
        torch.nn.utils.vector_to_parameters(torch.from_numpy(theta), self.model.parameters())

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        torch = self._torch
        with torch.no_grad():
            logits = self.model(torch.as_tensor(np.asarray(obs)).float() / 255.0).numpy()
        return np.stack([deep_ga.act_from_logits(row) for row in logits]), None


class OpenLoopPolicy:
    def __init__(self, genome, k):
        self.genome = np.asarray(genome, dtype=np.int64)
        self.k = int(k)
        self.i = 0

    def reset_episode(self):
        self.i = 0

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        gene = self.genome[min(self.i // self.k, len(self.genome) - 1)]
        self.i += 1
        n = len(obs) if obs is not None else 1
        return np.repeat(common.decode_gene(gene)[None], n, axis=0), None


def policy_from_record(record):
    """Policy of an individual record written by baselines.ga.fair ({"variant", "individual": {...}})."""
    ind = record["individual"]
    if record["variant"] == "deep-ga":
        return DeepGAPolicy(ind["chain"], ind["sigma"])
    if record["variant"] == "ga-ol":
        return OpenLoopPolicy(ind["genome"], ind["k"])
    raise ValueError(f"unknown variant {record['variant']!r}")


def load_run_policy(run_dir, which):
    """
    (policy, record) of `which` ("best", "final" or "snapshot_<step>") in a baselines.ga.fair run
    directory. A snapshot the run never reached (it stopped early) is the final individual, the policy
    the run ended with; the record then carries "snapshot_fallback": True.
    """
    path = os.path.join(run_dir, f"{which}.json")
    fallback = which.startswith("snapshot_") and not os.path.exists(path)
    with open(os.path.join(run_dir, "final.json") if fallback else path) as f:
        record = json.load(f)
    if fallback:
        record["snapshot_fallback"] = True
    return policy_from_record(record), record
