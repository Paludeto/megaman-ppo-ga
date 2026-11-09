"""
Module: experiments.fair.scheduler
Description: Runs the job graph of experiments.fair.jobs on this machine with CPU/GPU slot accounting.

  - Resumable: a job is done when campaign/fair/markers/<id>.done exists (written on exit code 0). A
    failure writes <id>.failed and blocks the jobs that depend on it; delete the .failed marker and
    restart to retry. Jobs that were running when the scheduler stopped are simply run again.
  - Mix: up to GPU_SLOTS PPO runs (GPU_SLOTS_NO_GA once no GA work remains) and MAX_GA GA-type runs,
    within CPU_SLOTS. Strict class priority PPO > GA > evaluations; within a class, the job-list order.
    A ready job of a higher class that is under its cap but does not fit holds back every lower class,
    so small evaluations cannot keep taking the slots a PPO run is waiting for.
  - Stop: kill the process group in campaign/fair/scheduler.pid (experiments/fair/stop.sh); SIGTERM
    stops the children too.

Start (detached, survives the terminal):
  setsid nohup python -m experiments.fair.scheduler >> campaign/fair/scheduler.out 2>&1 &
"""

import os
import signal
import subprocess
import sys
import time

from experiments.fair import config as C
from experiments.fair.jobs import build_jobs

MARKERS = os.path.join(C.CAMPAIGN, "markers")
LOGS = os.path.join(C.CAMPAIGN, "logs")
LOG = os.path.join(C.CAMPAIGN, "scheduler.log")
GA_KINDS = ("ga", "tune")


def _name(jid):
    return jid.replace("/", "__")


def marker(jid, state):
    return os.path.join(MARKERS, f"{_name(jid)}.{state}")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def job_class(job):
    """Scheduling priority: 0 = PPO, 1 = GA, 2 = everything else (evaluations, curves, selections)."""
    return 0 if job.kind == "ppo" else 1 if job.kind in GA_KINDS else 2


def main():
    os.makedirs(MARKERS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(C.CAMPAIGN, "scheduler.pid"), "w") as f:
        f.write(f"{os.getpid()} {os.getpgid(0)}\n")
    jobs = build_jobs()
    running = {}                       # id -> (job, Popen, t0, logfile)
    reported_blocked = set()

    def stop(signum, frame):
        log(f"SCHEDULER_STOP signal={signum}; terminating {len(running)} running job(s)")
        for _, p, _, _ in running.values():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(1)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    commit = subprocess.run(["git", "-C", C.ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", C.ROOT, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    log(f"SCHEDULER_START pid={os.getpid()} jobs={len(jobs)} commit={commit} dirty={bool(dirty)}")

    while True:
        for jid, (job, p, t0, lf) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            del running[jid]
            dt = (time.time() - t0) / 60.0
            state = "done" if rc == 0 else "failed"
            with open(marker(jid, state), "w") as f:
                f.write(f"rc={rc} minutes={dt:.1f} commit={commit}\n")
            log(f"{'DONE' if rc == 0 else 'FAIL'} {jid} rc={rc} {dt:.1f} min")

        done = {j.id for j in jobs if os.path.exists(marker(j.id, "done"))}
        failed = {j.id for j in jobs if os.path.exists(marker(j.id, "failed"))}
        pending = [j for j in jobs if j.id not in done and j.id not in failed and j.id not in running]
        blocked = {j.id for j in pending if any(d in failed for d in j.deps)}
        for jid in blocked - reported_blocked:
            log(f"BLOCKED {jid} (a dependency failed)")
        reported_blocked |= blocked
        pending = [j for j in pending if j.id not in blocked]
        if not pending and not running:
            log(f"SCHEDULER_END done={len(done)} failed={len(failed)} blocked={len(blocked)}")
            return
        ready = [j for j in pending if all(d in done for d in j.deps)]

        cpu = sum(job.slots for job, _, _, _ in running.values())
        n_ppo = sum(job.kind == "ppo" for job, _, _, _ in running.values())
        n_ga = sum(job.kind in GA_KINDS for job, _, _, _ in running.values())
        ga_left = n_ga > 0 or any(j.kind in GA_KINDS for j in pending)
        ppo_cap = C.GPU_SLOTS if ga_left else C.GPU_SLOTS_NO_GA
        ppo_left = n_ppo > 0 or any(j.kind == "ppo" for j in pending)
        ga_cap = C.MAX_GA_WITH_PPO if ppo_left else C.MAX_GA
        held = set()                       # classes held back by a waiting higher-class job
        for j in sorted(ready, key=lambda j: job_class(j)):
            cls = job_class(j)
            if cls in held:
                continue
            if j.kind == "ppo" and n_ppo >= ppo_cap:
                continue
            if j.kind in GA_KINDS and n_ga >= ga_cap:
                continue
            if cpu + j.slots > C.CPU_SLOTS:
                held |= {c for c in (1, 2) if c > cls}
                continue
            try:
                cmd = j.build()
            except Exception as e:  # e.g. the selected hyperparameters are missing: fail this job only
                with open(marker(j.id, "failed"), "w") as f:
                    f.write(f"build error: {e!r}\n")
                log(f"FAIL {j.id} (could not build the command: {e!r})")
                continue
            lf = open(os.path.join(LOGS, f"{_name(j.id)}.log"), "a")
            lf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {cmd}\n")
            lf.flush()
            env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1" if j.kind == "eval" else
                       os.environ.get("OMP_NUM_THREADS", ""))
            if not env["OMP_NUM_THREADS"]:
                del env["OMP_NUM_THREADS"]
            # PPO steps its 16 envs in lockstep, so it stalls whenever any env waits for a core: every
            # other job runs at a lower CPU priority and only takes the cycles PPO leaves idle.
            argv = ["bash", "-c", cmd] if j.kind == "ppo" else ["nice", "-n", str(C.NICE_NON_PPO), "bash", "-c", cmd]
            p = subprocess.Popen(argv, cwd=C.ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
            running[j.id] = (j, p, time.time(), lf)
            cpu += j.slots
            n_ppo += j.kind == "ppo"
            n_ga += j.kind in GA_KINDS
            log(f"START {j.id} (cpu {cpu}/{C.CPU_SLOTS}, ppo {n_ppo}, ga {n_ga})")
        time.sleep(5)


if __name__ == "__main__":
    main()
