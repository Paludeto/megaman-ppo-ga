# Comparison campaign: PPO, Deep GA and open-loop GA

*Read this in another language: **English** · [Português](EXPERIMENTS.pt-BR.md)*

`experiments/fair/` trains and evaluates the three methods under one protocol. All constants are in
`experiments/fair/config.py`.

## Protocol

| Item | Value |
|------|-------|
| Bosses | Gutsman, Cutman, Bombman, Elecman, Iceman, Fireman |
| Budget | 3,000,000 agent steps per run; all rollout steps count |
| Seeds | PPO 666, 1, 2, 3, 4; GAs 0, 1, 2, 3, 4 |
| Episode | at most 7200 emulator frames (1800 agent steps) |
| Final runs | 3 methods × 6 bosses × 2 regimes × 5 seeds = 180 |

**Training regimes**

- `sticky` (primary): sticky actions with `p = 0.25` per emulator frame in every training rollout, no
  early stop.
- `det`: deterministic training that stops at the first no-damage validation, checked every 100k steps.

In both regimes the evaluated model is the final one: the policy at the stop or at the budget. For the
GAs it is the top individual of the last generation, without re-evaluation. PPO's validation episodes
do not count against the budget, because `src/train.py` plays them in a separate environment. The GAs
read validation from their own rollouts.

**Hyperparameter tuning.** Each method has a grid of 4 configurations (`DET_GRID`), run on all 6 bosses
with seed 1000 in the `det` regime and the campaign budget. Runs are ordered by (reached the no-damage
stop, fewer steps to it, final validation reward). Configurations are compared by mean rank over a fold
of bosses, A = {Gutsman, Cutman, Elecman} and B = {Bombman, Iceman, Fireman}, and each boss trains with
the choice of the **other** fold.

In the `sticky` regime the GAs also tune `K ∈ {1, 4}`, the episodes per individual, which has no PPO
counterpart. They use the same folds and seed; the criterion is the mean training reward of the final
individual over 30 sticky episodes.

**Evaluation.** `src.eval.evaluate()` runs for all methods, with seed 123 and the same per-episode
streams.

| Condition | Sticky `p` | No-op max | Episodes | Policy |
|-----------|:----------:|:---------:|:--------:|--------|
| `det1` | 0 | 0 | 1 | greedy |
| `sticky` (primary) | 0.25 | 0 | 30 | greedy |
| `stickynoop` | 0.25 | 30 | 30 | greedy |
| `noop` | 0 | 30 | 30 | greedy |
| `stickystoch` (PPO only) | 0.25 | 0 | 30 | sampled |

Learning curves evaluate the policy under `sticky` every 500k steps.

**Statistics** (`experiments/fair/stats.py`). The unit of analysis is the training seed, scored by its
win rate over the 30 episodes of a condition.

- Per boss: mean with a percentile bootstrap 95% interval (10,000 resamples) and the exact two-sided
  permutation test over all 252 splits, with Holm correction over the 6 bosses per pair of methods and
  regime. With 5 seeds per method, the smallest possible Holm-adjusted `p` is 0.048.
- Over all bosses (primary test): stratified permutation test of the mean per-boss difference (100,000
  permutations), with Holm correction over the two comparisons (PPO vs each GA). Its resolution limit
  is `p = 2e-5` after correction.
- Interquartile mean and probability of improvement, with stratified bootstrap intervals.
- Exploratory and uncorrected: PPO vs Deep GA at 0.5M and 1M steps in the `sticky` regime
  (`early_budget_tests` in `results.json`), and the random-policy floor (`random_baseline.py`).

## Reproducing the campaign

```bash
conda env create -f environment.yml && conda activate megaman-ppo     # exact versions
cp /path/to/your/megaman.nes custom_integrations/MegaMan-v1-Nes/rom.nes

python -m experiments.fair.use_selection    # use the original campaign's hyperparameters (skips 96 tuning runs)
mkdir -p campaign/fair
setsid nohup python -m experiments.fair.scheduler >> campaign/fair/scheduler.out 2>&1 &

python -m experiments.fair.status           # jobs by state, what is running, failures
experiments/fair/stop.sh                    # stop; running jobs are redone on the next start
python -m experiments.fair.random_baseline  # after the campaign: exploratory random-policy floor
```

Run these from the repository root, in the conda environment: the jobs use the interpreter that starts
the scheduler. The last job is `experiments.fair.aggregate`, which writes the tables and statistics;
`python -m experiments.fair.aggregate --partial` computes them from the seeds finished so far.

`experiments/fair/selection/` holds the configurations chosen in the original campaign. Skip
`use_selection` to rerun the tuning as well; the campaign then has 1362 jobs (96 tuning runs, 5
selections, 180 final runs, 180 evaluations, 900 learning-curve points and the aggregation), and PPO
tuning may select other configurations (see below).

The scheduler is resumable. A job is done when `campaign/fair/markers/<id>.done` exists; a failure
writes `<id>.failed` and blocks its dependents until that file is deleted. `CPU_SLOTS`, `GPU_SLOTS`,
`MAX_GA` and `SLOTS` in `config.py` are sized for the machine below, so adjust them for yours. They
change the wall-clock time, not the results. On that machine the scheduler log of the campaign spans
about 33 hours.

To test the whole pipeline in miniature (2 bosses, 2 seeds, tiny budgets, output in
`campaign/fair_smoke`), which is never meant for results:

```bash
mkdir -p campaign/fair_smoke
FAIR_SMOKE=1 python -m experiments.fair.scheduler
```

### What was checked

With this code and the locked environment, on the machine of the original campaign (AMD Ryzen 9 7950X,
NVIDIA RTX 4070 Ti SUPER, driver 580.178.04, Linux), a sample of the campaign was executed again and
compared with its original outputs:

| Check | Result |
|-------|--------|
| 3 GA runs to 3M steps (Deep GA `det` and `sticky`, open-loop GA `sticky`) | history, snapshots and final individuals identical |
| 1 PPO run to 500k steps (`sticky`) | policy weights identical to the original checkpoint |
| 5 evaluations of saved models (PPO greedy and sampled, both GAs) | identical episode by episode |
| Statistics recomputed from the original evaluations | identical |
| Random-policy floor | identical |

The full campaign was not run again with this exact code. Results are only expected to be bit-identical
under the same conditions: the GAs run on CPU and depend on the emulator, seeds and library versions,
while PPO also depends on the GPU, driver and CUDA kernels, because the campaign does not pass
`--deterministic-torch`. On other hardware, expect PPO to diverge from the original runs; compare it
statistically, over seeds. With `--deterministic-torch`, two short PPO runs on that machine had identical
weights, but that mode is not the one used in the campaign.

## Outputs

Everything below is generated and git-ignored.

| Path | Content |
|------|---------|
| `campaign/fair/tuning/` | tuning runs and `chosen_<regime>_<method>.json` |
| `campaign/fair/runs/<regime>/<method>/<boss>/s<seed>/` | GA runs |
| `models/fair_<regime>_<boss>_s<seed>*_best/` | PPO final model and learning-curve snapshots |
| `campaign/fair/eval/` | one JSON per model and condition |
| `campaign/fair/results.json`, `agarwal.json` | per-seed scores, tests, learning curves, IQM and probability of improvement |
| `campaign/fair/RESULTS.md` | tables, with labels in Portuguese |
| `campaign/fair/random_baseline.json` | random-policy wins per boss, seed and condition |
| `campaign/fair/logs/`, `markers/`, `scheduler.log` | per-job logs and state |

## Interpretation limits

- Step budgets count agent steps, not emulator frames or compute.
- Each boss starts from one fixed save state, and each condition has 30 episodes.
- PPO is trained with a sampled policy and evaluated greedily by default. `stickystoch` measures the
  sampled policy.
- `K` tuning and the frame tie-break exist only in the GAs.
