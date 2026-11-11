# Mega Man (NES) boss fights: PPO, Deep GA and open-loop GA

*Read this in another language: **English** · [Português](README.pt-BR.md)*

Code of a comparison of three methods on the 6 Robot Master fights of *Mega Man* (NES): Gutsman,
Cutman, Bombman, Elecman, Iceman and Fireman.

| Method | What is learned | Observation | Code |
|--------|-----------------|-------------|------|
| **PPO** (stable-baselines3, `CnnPolicy`) | network weights, by gradient | 84×84×4 grayscale frame stack | `src/` |
| **Deep GA** (Such et al. 2017) | weights of the same CNN, by mutation and selection | the same frame stack | `baselines/ga/` |
| **Open-loop GA** (the "Brute" idea of Machado et al. 2018) | a fixed sequence of actions | none: it never sees the screen | `baselines/ga/` |

`experiments/fair/` runs them under one protocol: same step budget, reward, tuning effort and
evaluator. **To reproduce the campaign, see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).**

- **Actions:** `MultiDiscrete([2, 5, 2])` = `[B, D-pad, A]`; the D-pad values are none, UP, DOWN,
  LEFT, RIGHT.
- **Evaluation reward:** `r = d · (hits − damage)`, with `d = 0.05`. Hits and damage are *events* (a
  step in which the boss or Mega Man lost HP), not HP magnitudes.
- **Training reward:** the same, with a 2× damage penalty and `+0.5` for the kill. The GAs use its
  undiscounted return as fitness.
- **Win:** `boss_health` (RAM address 1729) reaches 0 with Mega Man alive.

## Setup

```bash
conda env create -f environment.yml && conda activate megaman-ppo
cp /path/to/your/megaman.nes custom_integrations/MegaMan-v1-Nes/rom.nes
tail -c +17 custom_integrations/MegaMan-v1-Nes/rom.nes | sha1sum
# must be: 6047e52929dfe8ed4708d325766ccb8d3d583c7d
pytest tests -q
```

`environment.yml` installs the exact versions in `requirements.lock.txt` (Python 3.12.13, Linux,
CUDA 13 wheels). A GPU is strongly recommended for PPO; the GAs run on CPU, with one emulator per
worker process (at most 12).

> The ROM is **not** distributed here. You need a legally obtained copy of *Mega Man (NES)*. `rom.sha`
> holds the SHA-1 **without** the 16-byte iNES header, the stable-retro convention. `make_env` verifies
> it and raises `RuntimeError` on a mismatch. The tests that need the emulator are skipped without
> `rom.nes`.

## Quick start

```bash
python src/train.py --tag bombman --state Bombman-boss --timesteps 3000000 --lr-horizon 40000000
python src/eval.py --model models/bombman_best/final_model.zip --state Bombman-boss \
  --episodes 30 --sticky 0.25 --seed 123

python -m baselines.ga.fair --variant deep-ga --boss bombman \
  --max-env-steps 3000000 --seed 0 --out-dir out/deep-ga
python -m baselines.ga.evaluate --run out/deep-ga --which final \
  --episodes 30 --sticky 0.25 --seed 123 --json out/deep-ga/eval.json
```

Use `--variant ga-ol` for the open-loop GA. Run every command from the repository root.

## Layout

```
src/
  wrappers.py        frameskip, reward (BossWrapper), 84×84 preprocessing
  env.py             make_env / make_venv: integration, ROM check, vectorization
  train.py           PPO training
  eval.py            evaluation: Wilson intervals, sticky actions, no-op starts
baselines/ga/
  fair.py            GA search loop (both variants)
  evaluate.py        evaluates a saved individual through src/eval.py
  deep_ga.py, ga_ol.py   genetic operators
  policy.py, common.py   individuals as policies; tasks, worker pool, episode runner
experiments/fair/
  config.py, jobs.py, scheduler.py, status.py, stop.sh   the campaign and its job graph
  tune_select.py, selection/, use_selection.py           hyperparameter selection
  aggregate.py, stats.py, ppo_summary.py                 statistics and tables
  random_baseline.py                                     exploratory random policy
tests/               pytest suite
custom_integrations/MegaMan-v1-Nes/   stable-retro integration (without rom.nes)
docs/                architecture guide and campaign guide
```

Generated at runtime and git-ignored: `logs/`, `checkpoints/`, `models/`, `out/`, `campaign/`.

## Train PPO

The recipe is the set of constants at the top of `src/train.py`. Every ~100k steps the policy plays
one deterministic validation episode on the evaluation task, and `best_model.zip` keeps the best
validation reward. Training ends at the first validation won **without damage** or at the
`--timesteps` ceiling; `--no-early-stop` always trains to the ceiling.

| Flag | Meaning |
|------|---------|
| `--tag`, `--state` | run name; save state, such as `Bombman-boss` (required) |
| `--timesteps`, `--n-envs`, `--seed` | ceiling (40M), parallel envs (16), seed (666) |
| `--lr`, `--ent-coef`, `--lr-horizon` | initial LR (2.5e-4), entropy coefficient (5e-3), step where the LR reaches 0 |
| `--train-sticky P`, `--no-early-stop` | sticky actions in training (validation stays deterministic); no early stop |
| `--episode-frames` | episode length in emulator frames (7200) |
| `--deterministic-torch` | deterministic torch and CUDA kernels (slower) |

Each run writes `logs/<tag>/` (TensorBoard and `run_meta.json`), `checkpoints/<tag>/` and
`models/<tag>_best/{best_model,final_model}.zip`. `run_meta.json` records library versions, git
commit, hardware, arguments and constants.

## Evaluate

`src/eval.py` reports win and no-damage win rates with 95% Wilson intervals, HP lost, damage events
and frames until the kill. Options: `--deterministic` (default) or `--stochastic`, `--sticky P` (each
emulator frame repeats the previous action with probability `P`), `--noop-max N`, `--seed`, `--device`,
`--episode-frames`, `--json PATH`. A single deterministic validation episode says nothing about
robustness; measure that here.

## Documentation

- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): protocol and reproduction of the campaign.
- [RESULTS.md](RESULTS.md): results of the original campaign, as reported in the paper.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the environment, PPO and GAs are built.
- [baselines/ga/README.md](baselines/ga/README.md): the genetic algorithms.

## License

See [LICENSE](LICENSE). The *Mega Man* ROM is not included and is not covered by this license.
