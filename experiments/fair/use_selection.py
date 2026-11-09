"""
Module: experiments.fair.use_selection
Description: Installs the hyperparameters selected in the original campaign (experiments/fair/selection/) and
             marks the tuning and selection jobs as done, so the scheduler starts at the final runs. PPO
             tuning runs on the GPU without deterministic kernels, so re-running the tuning can select other
             configurations; this fixes the configurations of the final runs instead.

Usage: python -m experiments.fair.use_selection     (then start experiments.fair.scheduler)
"""

import glob
import os
import shutil

from experiments.fair import config as C
from experiments.fair.jobs import build_jobs
from experiments.fair.scheduler import MARKERS, marker

SELECTION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selection")


def main():
    os.makedirs(os.path.join(C.CAMPAIGN, "tuning"), exist_ok=True)
    os.makedirs(MARKERS, exist_ok=True)
    for path in glob.glob(os.path.join(SELECTION, "chosen_*.json")):
        shutil.copy(path, os.path.join(C.CAMPAIGN, "tuning", os.path.basename(path)))
    done = [j for j in build_jobs() if j.id.startswith(("tune/", "select/"))]
    for job in done:
        open(marker(job.id, "done"), "w").close()
    print(f"installed the selection and marked {len(done)} tuning and selection jobs as done")


if __name__ == "__main__":
    main()
