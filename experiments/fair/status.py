"""
Module: experiments.fair.status
Description: Progress of the fair-comparison campaign: jobs by kind and state, what is running (with the
             last log line), and the failures.

Usage: python -m experiments.fair.status
"""

import collections
import os
import time

from experiments.fair import config as C
from experiments.fair.jobs import build_jobs
from experiments.fair.scheduler import LOGS, MARKERS, marker, _name


def main():
    jobs = build_jobs()
    started = {}
    log_path = os.path.join(C.CAMPAIGN, "scheduler.log")
    if os.path.exists(log_path):
        for line in open(log_path):
            parts = line.split()
            if len(parts) >= 3 and parts[2].startswith("SCHEDULER_START"):
                started = {}   # jobs started by an earlier (killed) scheduler are not running
            elif len(parts) >= 4 and parts[2] == "START":
                started[parts[3]] = " ".join(parts[:2]).strip("[]")
    counts = collections.defaultdict(collections.Counter)
    running, failed = [], []
    for j in jobs:
        if os.path.exists(marker(j.id, "done")):
            st = "done"
        elif os.path.exists(marker(j.id, "failed")):
            st = "failed"
            failed.append(j.id)
        elif j.id in started:
            st = "running"
            running.append(j)
        else:
            st = "pending"
        counts[j.kind][st] += 1
    print(f"{'kind':10s} {'done':>6s} {'running':>8s} {'pending':>8s} {'failed':>7s}")
    for kind in ("tune", "select", "ppo", "ga", "eval", "curve", "aggregate"):
        c = counts[kind]
        print(f"{kind:10s} {c['done']:6d} {c['running']:8d} {c['pending']:8d} {c['failed']:7d}")
    print("\nrunning:")
    for j in running:
        lp = os.path.join(LOGS, f"{_name(j.id)}.log")
        last = ""
        if os.path.exists(lp):
            lines = [l for l in open(lp, errors="replace").read().splitlines() if l.strip()]
            last = lines[-1][:110] if lines else ""
        print(f"  {j.id}  since {started[j.id]}\n      {last}")
    if failed:
        print("\nfailed:", *failed, sep="\n  ")
    pid_file = os.path.join(C.CAMPAIGN, "scheduler.pid")
    if os.path.exists(pid_file):
        pid = int(open(pid_file).read().split()[0])
        alive = os.path.exists(f"/proc/{pid}")
        print(f"\nscheduler pid {pid}: {'running' if alive else 'NOT running'}  ({time.strftime('%Y-%m-%d %H:%M')})")


if __name__ == "__main__":
    main()
