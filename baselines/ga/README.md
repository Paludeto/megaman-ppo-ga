# Genetic algorithms

*Read this in another language: **English** · [Português](README.pt-BR.md)*

Two genetic algorithms that play the same task as the PPO agent:

| Method | Genome | Perception | Reference |
|--------|--------|------------|-----------|
| **Open-loop GA** (GA-OL) | sequence of `MultiDiscrete([2,5,2])` actions in blocks of `k` agent steps | none: it replays the same actions whatever happens on screen | the "Brute" idea of Machado et al. 2018 |
| **Deep GA** | weights of a `NatureCNN` + linear head with 2+5+2 logits (1,688,745 parameters), action = per-head argmax | the same 4×84×84 uint8 frame stack as PPO | Such et al. 2017 |

Tasks are built with `src.env.make_env` (frameskip 4, 7200 emulator frames).

- **Fitness:** the undiscounted episode return of the PPO training reward: `+0.05` per boss hit,
  `-0.10` per damage event and `+0.5` for the kill. PPO's discount and reward normalization belong to
  the algorithm and are not applied.
- **Ties:** broken by win and then by fewer frames until the kill.
- **Evaluation:** `0.05 * (boss hits - damage events)`, through the same `src.eval.evaluate()` used for
  PPO. A win needs the boss dead with Mega Man alive.

Run everything from the repository root, with a legally obtained `rom.nes`. Each worker process owns
one emulator and one torch/OpenCV thread; at most 12 processes are allowed.

## Usage

`fair.py` counts the agent steps of every rollout against `--max-env-steps`.

```bash
# deterministic training: stops at the first no-damage validation
python -m baselines.ga.fair --variant deep-ga --boss cutman --max-env-steps 3000000 --seed 0 --out-dir out/dga

# sticky training: K episodes per individual, runs to the budget
python -m baselines.ga.fair --variant ga-ol --boss cutman --max-env-steps 3000000 --seed 0 --out-dir out/ol \
    --train-sticky 0.25 --episodes-per-eval 1 --no-early-stop

# evaluation of a saved individual (best, final or snapshot_<step>)
python -m baselines.ga.evaluate --run out/dga --which final --episodes 30 --sticky 0.25 --seed 123 --json eval.json
```

| Regime | Rollouts | Elites | Stop |
|--------|----------|--------|------|
| Deterministic | one exact episode per individual | keep their metrics | first no-damage validation, every `--validate-every` steps (100k), or the budget |
| Sticky (`--train-sticky P`) | `--episodes-per-eval K` episodes with fresh seeds; fitness is the mean | re-evaluated every generation | the budget |

The directory given by `--out-dir` receives `config.json`, `history.json`, `validations.json`,
`best.json`, `final.json`, `snapshot_<step>.json` (with `--snapshot-at`) and `summary.json`.

| | GA-OL | Deep GA |
|--|-------|---------|
| Population (`--pop`) | 240 | 96 |
| Selection | truncation at `--trunc` 0.2, `--elite` 6 | truncation at `--parents` 10, `--elite` 1 |
| Variation | two-point crossover (`--p-cross` 0.5), per-gene resampling (`--p-mut` 0.02) | Gaussian mutation (`--sigma` 0.005), no crossover |
| Genome | blocks of `--k` 4 agent steps | seed chain, `theta_g = theta_(g-1) + sigma * eps(seed_g)` |

These are the defaults. The hyperparameters selected for each boss are in
`experiments/fair/selection/`, and the protocol is in [docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md).
