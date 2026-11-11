# Results of the original campaign

These are the results of the campaign reported in the paper: 3 methods × 6 Robot Masters × 2 training
regimes × 5 seeds (180 final models), run with this code on one machine (AMD Ryzen 9 7950X, NVIDIA RTX
4070 Ti SUPER, driver 580.178.04, Linux, Python 3.12.13, torch 2.12.0, stable-retro 1.0.0). The protocol is
in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). All 1,362 jobs finished without failure.

Most tables come from `python -m experiments.fair.aggregate`, run on the campaign outputs. The random
floor comes from `python -m experiments.fair.random_baseline`. Trained models and per-episode records are
not in this repository. A new run on other hardware is not expected to give the same PPO numbers (see
*What was checked* in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)).

**Conventions.** The unit of analysis is the training seed, scored as its win rate over 30 evaluation
episodes. *Win*: the boss reaches zero health with Mega Man alive. *No-damage win*: a win without losing
health. PPO seeds are 666, 1, 2, 3 and 4; GA seeds are 0 to 4. Per-seed lists follow that order. PPO and
Deep GA are evaluated greedily. Boss order: Gutsman, Cutman, Bombman, Elecman, Iceman, Fireman.

Evaluation conditions:

| Condition | Meaning |
|-----------|---------|
| sticky (S) | sticky actions with p = 0.25 per emulator frame; **primary outcome** |
| sticky + no-op (S+N) | sticky actions plus 0–30 idle steps at the start |
| no-op (N) | 0–30 idle steps at the start, no sticky actions |
| deterministic | one episode, no perturbation |

## Selected hyperparameters

Each boss uses the configuration chosen on the other fold: fold A = Gutsman, Cutman, Elecman; fold B =
Bombman, Iceman, Fireman.

| Method | Grid | Used on Gutsman, Cutman, Elecman (chosen on B) | Used on Bombman, Iceman, Fireman (chosen on A) |
|--------|------|------------------------------------|------------------------------------|
| PPO | lr ∈ {2.5, 1}×10⁻⁴, entropy ∈ {0.005, 0.01} | `lr10_e5`: lr 1×10⁻⁴, entropy 0.005 | `lr25_e10`: lr 2.5×10⁻⁴, entropy 0.01 |
| Deep GA | σ ∈ {0.005, 0.002}, population ∈ {96, 256} | `s005_p96`; sticky: K = 1 | `s005_p96`; sticky: K = 4 |
| Open-loop GA | p_mut ∈ {0.02, 0.005}, population ∈ {240, 96} | `m020` (population 240); sticky: K = 1 | `m020` (population 240); sticky: K = 1 |

PPO uses the same configuration in both regimes. *K* (episodes per individual) was tuned only in the sticky
regime; the deterministic regime uses K = 1.

## Primary comparison

PPO minus each GA in mean win rate under sticky evaluation, over all bosses. The p-values come from a
two-sided permutation test stratified by boss (100,000 permutations), with Holm correction over the two
comparisons. 2×10⁻⁵ is the resolution limit after correction.

| Training | PPO − Deep GA | PPO − open-loop GA | p_Holm (both) |
|----------|---------------|--------------------|---------------|
| sticky (primary) | +43.7 points | +46.2 points | ≤ 2×10⁻⁵ |
| deterministic | +45.1 points | +60.0 points | ≤ 2×10⁻⁵ |

## Win rate per boss under sticky evaluation

Mean over 5 seeds [bootstrap 95% CI], then the wins of each seed out of 30. p_D and p_O: two-sided exact
5×5 permutation test, PPO against Deep GA and against the open-loop GA, Holm-adjusted over the six bosses.
The smallest possible adjusted p is 0.048. **Bold**: p < 0.05.

### Sticky training (primary)

| Boss | PPO | Deep GA | Open-loop GA | p_D | p_O |
|------|-----|---------|--------------|-----|-----|
| Gutsman | 99% [98, 100] (30 30 29 30 30) | 87% [83, 92] (26 25 26 24 29) | 71% [61, 82] (16 20 20 25 26) | **0.048** | **0.048** |
| Cutman | 100% [100, 100] (30 30 30 30 30) | 31% [24, 38] (6 8 10 13 9) | 37% [21, 50] (14 2 9 17 13) | **0.048** | **0.048** |
| Bombman | 100% [100, 100] (30 30 30 30 30) | 49% [48, 50] (15 15 15 14 15) | 47% [35, 57] (19 8 17 12 14) | **0.048** | **0.048** |
| Elecman | 53% [37, 74] (17 14 27 8 13) | 22% [11, 33] (5 11 3 3 11) | 0% [0, 0] (0 0 0 0 0) | 0.079 | **0.048** |
| Iceman | 67% [31, 91] (0 23 20 30 27) | 0% [0, 0] (0 0 0 0 0) | 0% [0, 0] (0 0 0 0 0) | 0.079 | 0.095 |
| Fireman | 100% [100, 100] (30 30 30 30 30) | 68% [50, 79] (23 22 22 25 10) | 87% [80, 94] (24 23 27 30 26) | **0.048** | 0.095 |
| **Mean** | **86%** | **43%** | **40%** | | |

### Deterministic training

| Boss | PPO | Deep GA | Open-loop GA | p_D | p_O |
|------|-----|---------|--------------|-----|-----|
| Gutsman | 83% [71, 93] (28 27 24 18 28) | 93% [88, 96] (28 28 25 29 29) | 35% [23, 45] (10 13 9 4 16) | 0.214 | **0.048** |
| Cutman | 99% [97, 100] (30 30 30 29 29) | 33% [27, 38] (9 12 8 8 12) | 9% [2, 20] (1 0 3 9 1) | **0.048** | **0.048** |
| Bombman | 92% [84, 98] (23 29 27 29 30) | 19% [13, 27] (4 3 5 10 6) | 8% [1, 15] (2 6 0 0 4) | **0.048** | **0.048** |
| Elecman | 35% [25, 48] (10 10 18 8 6) | 6% [3, 9] (1 2 0 3 3) | 0% [0, 0] (0 0 0 0 0) | **0.048** | **0.048** |
| Iceman | 45% [33, 62] (10 12 9 14 23) | 0% [0, 0] (0 0 0 0 0) | 0% [0, 0] (0 0 0 0 0) | **0.048** | **0.048** |
| Fireman | 99% [97, 100] (29 30 30 29 30) | 32% [19, 47] (18 3 10 8 9) | 41% [25, 59] (15 5 8 22 11) | **0.048** | **0.048** |
| **Mean** | **75%** | **30%** | **15%** | | |

No boss favors a GA significantly in either regime.

## Aggregate metrics

Over the 30 runs (boss × seed) of each method. IQM = interquartile mean. P(PPO > X) is the probability
that a PPO run beats a run of X on the same boss. Brackets: 95% CI from a bootstrap stratified by boss
(10,000 resamples).

| Training | Condition | PPO IQM | Deep GA IQM | Open-loop GA IQM | P(PPO > Deep GA) | P(PPO > open-loop GA) |
|----------|-----------|---------|-------------|------------------|------------------|-----------------------|
| sticky | S | 99% [96, 100] | 42% [36, 47] | 38% [31, 45] | 0.97 [0.92, 1.00] | 0.97 [0.92, 1.00] |
| sticky | S+N | 91% [88, 95] | 41% [35, 47] | 9% [6, 12] | 0.98 [0.95, 1.00] | 0.98 [0.95, 1.00] |
| sticky | N | 96% [91, 99] | 42% [34, 50] | 6% [3, 9] | 0.89 [0.85, 0.92] | 0.98 [0.95, 1.00] |
| deterministic | S | 85% [79, 90] | 21% [17, 26] | 9% [5, 13] | 0.87 [0.83, 0.92] | 1.00 [1.00, 1.00] |
| deterministic | S+N | 81% [75, 86] | 24% [20, 30] | 1% [0, 2] | 0.87 [0.81, 0.94] | 0.98 [0.95, 1.00] |
| deterministic | N | 92% [86, 96] | 40% [34, 47] | 5% [3, 8] | 0.83 [0.75, 0.89] | 1.00 [1.00, 1.00] |

## Deterministic episode against perturbations

Runs (of 30) that win the single deterministic episode, and mean win rate under each perturbation.

| Training | Method | Deterministic wins | S | S+N | N |
|----------|--------|--------------------|---|-----|---|
| deterministic | PPO | 29/30 | 75% | 71% | 84% |
| deterministic | Deep GA | 25/30 | 30% | 33% | 44% |
| deterministic | Open-loop GA | 20/30 | 15% | 5% | 9% |
| sticky | PPO | 27/30 | 86% | 81% | 85% |
| sticky | Deep GA | 13/30 | 43% | 42% | 46% |
| sticky | Open-loop GA | 11/30 | 40% | 11% | 10% |

Seeds (of 5) that win the deterministic episode, per boss:

| Training | Method | Gutsman | Cutman | Bombman | Elecman | Iceman | Fireman |
|----------|--------|---------|--------|---------|---------|--------|---------|
| deterministic | PPO | 5 | 5 | 5 | 4 | 5 | 5 |
| deterministic | Deep GA | 5 | 5 | 5 | 5 | 0 | 5 |
| deterministic | Open-loop GA | 5 | 5 | 5 | 0 | 0 | 5 |
| sticky | PPO | 5 | 5 | 5 | 3 | 4 | 5 |
| sticky | Deep GA | 5 | 2 | 1 | 1 | 0 | 4 |
| sticky | Open-loop GA | 3 | 2 | 1 | 0 | 0 | 5 |

## Win rate per boss and condition

Mean over 5 seeds, as S / S+N / N (%).

| Training | Method | Gutsman | Cutman | Bombman | Elecman | Iceman | Fireman |
|----------|--------|---------|--------|---------|---------|--------|---------|
| sticky | PPO | 99 / 99 / 100 | 100 / 96 / 99 | 100 / 97 / 95 | 53 / 46 / 63 | 67 / 53 / 57 | 100 / 95 / 98 |
| sticky | Deep GA | 87 / 89 / 100 | 31 / 23 / 21 | 49 / 70 / 65 | 22 / 7 / 17 | 0 / 0 / 0 | 68 / 64 / 71 |
| sticky | Open-loop GA | 71 / 18 / 13 | 37 / 13 / 8 | 47 / 7 / 7 | 0 / 0 / 0 | 0 / 0 / 0 | 87 / 28 / 32 |
| deterministic | PPO | 83 / 84 / 97 | 99 / 97 / 97 | 92 / 89 / 90 | 35 / 40 / 69 | 45 / 22 / 54 | 99 / 91 / 93 |
| deterministic | Deep GA | 93 / 93 / 100 | 33 / 21 / 40 | 19 / 45 / 71 | 6 / 7 / 15 | 0 / 0 / 0 | 32 / 31 / 36 |
| deterministic | Open-loop GA | 35 / 9 / 20 | 9 / 1 / 9 | 8 / 1 / 5 | 0 / 0 / 0 | 0 / 0 / 0 | 41 / 19 / 19 |

## Learning curves

Win rate under sticky evaluation at each checkpoint (30 episodes), mean over 5 seeds. A run that stopped
early keeps its stopped policy. The 3M column is the final model.

### Sticky training

| Boss | Method | 0.5M | 1M | 1.5M | 2M | 2.5M | 3M |
|------|--------|------|----|------|----|------|----|
| Gutsman | PPO | 6% | 57% | 92% | 96% | 99% | 99% |
| Gutsman | Deep GA | 87% | 84% | 85% | 85% | 84% | 87% |
| Gutsman | Open-loop GA | 33% | 36% | 55% | 66% | 70% | 71% |
| Cutman | PPO | 0% | 8% | 28% | 66% | 99% | 100% |
| Cutman | Deep GA | 21% | 31% | 25% | 21% | 25% | 31% |
| Cutman | Open-loop GA | 5% | 7% | 23% | 29% | 36% | 37% |
| Bombman | PPO | 24% | 96% | 96% | 99% | 100% | 100% |
| Bombman | Deep GA | 51% | 55% | 47% | 47% | 53% | 49% |
| Bombman | Open-loop GA | 8% | 15% | 20% | 33% | 40% | 47% |
| Elecman | PPO | 2% | 29% | 36% | 41% | 46% | 53% |
| Elecman | Deep GA | 19% | 21% | 21% | 21% | 23% | 22% |
| Elecman | Open-loop GA | 0% | 0% | 0% | 0% | 0% | 0% |
| Iceman | PPO | 0% | 0% | 1% | 23% | 45% | 67% |
| Iceman | Deep GA | 0% | 0% | 0% | 0% | 0% | 0% |
| Iceman | Open-loop GA | 0% | 0% | 0% | 0% | 0% | 0% |
| Fireman | PPO | 9% | 75% | 92% | 97% | 100% | 100% |
| Fireman | Deep GA | 87% | 89% | 75% | 75% | 74% | 68% |
| Fireman | Open-loop GA | 56% | 71% | 81% | 77% | 83% | 87% |

Averaged over bosses at 0.5M: Deep GA 44%, open-loop GA 17%, PPO 7%. Exploratory stratified permutation
tests of PPO against Deep GA, not corrected for multiplicity: Deep GA ahead at 0.5M (−37.4 points,
p ≤ 10⁻⁵); not distinguishable at 1M (−2.3 points, p = 0.66).

### Deterministic training

Brackets: seeds that had already stopped by that checkpoint.

| Boss | Method | 0.5M | 1M | 1.5M | 2M | 2.5M | 3M |
|------|--------|------|----|------|----|------|----|
| Gutsman | PPO | 1% | 45% | 83% [1] | 85% [4] | 83% [5] | 83% |
| Gutsman | Deep GA | 91% | 81% | 89% | 89% | 93% | 93% |
| Gutsman | Open-loop GA | 21% | 33% | 29% | 36% | 31% | 35% |
| Cutman | PPO | 0% | 1% | 53% | 91% [1] | 98% [3] | 99% |
| Cutman | Deep GA | 27% | 25% | 25% | 31% | 22% | 33% |
| Cutman | Open-loop GA | 2% | 5% | 5% | 8% | 9% | 9% |
| Bombman | PPO | 36% | 87% | 91% | 91% | 91% [2] | 92% |
| Bombman | Deep GA | 38% | 37% | 22% | 12% | 13% | 19% |
| Bombman | Open-loop GA | 5% | 13% | 9% | 10% | 9% | 8% |
| Elecman | PPO | 3% | 21% | 31% | 41% [1] | 31% [1] | 35% |
| Elecman | Deep GA | 12% | 11% | 10% | 11% | 4% | 6% |
| Elecman | Open-loop GA | 0% | 0% | 0% | 0% | 0% | 0% |
| Iceman | PPO | 0% | 0% | 15% | 53% | 49% [1] | 45% |
| Iceman | Deep GA | 0% | 0% | 0% | 0% | 0% | 0% |
| Iceman | Open-loop GA | 0% | 0% | 0% | 0% | 0% | 0% |
| Fireman | PPO | 36% | 96% | 98% | 99% | 99% | 99% |
| Fireman | Deep GA | 52% | 45% | 36% | 42% | 21% | 32% |
| Fireman | Open-loop GA | 33% | 51% | 39% | 41% | 40% | 41% |

## Training cost in the deterministic regime

The deterministic regime stops at the first no-damage win in a validation episode, checked every 100,000
steps. Every sticky run, and every GA run in both regimes, used about 3M steps.

| | Gutsman | Cutman | Bombman | Elecman | Iceman | Fireman |
|-|---------|--------|---------|---------|--------|---------|
| PPO runs that stopped early (of 5) | 5 | 5 | 3 | 1 | 1 | 0 |
| PPO median steps of the final model | 1.6M | 2.2M | 2.8M | 3.0M | 3.0M | 3.0M |
| PPO steps per seed (666, 1–4) | 1.8M, 1.5M, 1.6M, 1.3M, 2.3M | 2.8M, 2.0M, 2.5M, 2.2M, 1.8M | 2.8M, 2.1M, 2.4M, 3M, 3M | 3M, 1.9M, 3M, 3M, 3M | 3M, 3M, 3M, 3M, 2.0M | 3M (all) |

A stopped run is one whose final model has fewer than 3M steps. The *Parou por abate sem dano* column of
the generated `campaign/fair/RESULTS.md` reads this from the training log. In the original campaign the
seed-666 deterministic PPO runs have no such log, so that column undercounts them; the step counts above
are taken from the saved models.

## No-damage wins

PPO's no-damage win rate under sticky evaluation: mean [bootstrap 95% CI], then the no-damage wins of each
seed out of 30. Both GAs have 0 no-damage wins in every run, so p (exact permutation, Holm over six bosses)
is the same against each GA.

| Boss | Sticky training | p | Deterministic training | p |
|------|-----------------|---|------------------------|---|
| Gutsman | 38% [28, 48] (6 15 9 15 12) | **0.048** | 14% [5, 23] (2 9 6 0 4) | 0.238 |
| Cutman | 41% [14, 72] (1 13 30 5 13) | **0.048** | 31% [16, 49] (2 6 19 10 9) | **0.048** |
| Bombman | 5% [0, 12] (0 0 0 3 5) | 1.000 | 1% [0, 3] (0 1 1 0 0) | 1.000 |
| Elecman | 1% [0, 2] (1 0 0 0 0) | 1.000 | 0% [0, 0] (0 0 0 0 0) | 1.000 |
| Iceman | 0% [0, 0] (0 0 0 0 0) | 1.000 | 0% [0, 0] (0 0 0 0 0) | 1.000 |
| Fireman | 0% [0, 0] (0 0 0 0 0) | 1.000 | 0% [0, 0] (0 0 0 0 0) | 1.000 |

In total, PPO had 128 of 900 sticky episodes without damage under sticky training and 69 of 900 under
deterministic training. Each GA had 0 of 900 in each regime.

## PPO greedy against sampled

PPO is compared with the GAs using its greedy (argmax) policy, but it is trained by sampling. Mean win rate
under sticky evaluation (condition `stickystoch` for the sampled policy).

| Training | Policy | Gutsman | Cutman | Bombman | Elecman | Iceman | Fireman |
|----------|--------|---------|--------|---------|---------|--------|---------|
| sticky | greedy | 99% | 100% | 100% | 53% | 67% | 100% |
| sticky | sampled | 98% | 99% | 100% | 41% | 50% | 99% |
| deterministic | greedy | 83% | 99% | 92% | 35% | 45% | 99% |
| deterministic | sampled | 93% | 97% | 95% | 22% | 24% | 99% |

## Random-policy floor

Exploratory. Each action is drawn uniformly at every agent step: 5 seeds × 30 episodes per boss and
condition, 150 episodes in all. Wins:

| Condition | Gutsman | Cutman | Bombman | Elecman | Iceman | Fireman |
|-----------|---------|--------|---------|---------|--------|---------|
| sticky | 18 | 1 | 4 | 0 | 0 | 2 |
| sticky + no-op | 13 | 4 | 2 | 0 | 0 | 2 |
| no-op | 24 | 2 | 5 | 0 | 0 | 2 |

## Deep GA in the deterministic regime

These numbers come from the GA run histories (`campaign/fair/runs/`) and back the paper's discussion of
the Deep GA plateau. Ties in fitness are broken by win and then by fewer frames to the kill.

- On Bombman and Fireman, the best fitness is the same in all 5 seeds from 0.5M to 3M steps (0.9 and 0.8).
- In all 10 of those runs, once the best fitness is reached (within 0.25M steps), the frames to the kill
  of the top individual only go down. On Bombman they drop from 931–981 to 676–832.
- Over the same span, the sticky win rate drops on average (Bombman 38% to 19%, Fireman 52% to 32%) and
  varies widely. For one Fireman seed, sticky wins out of 30 at the six checkpoints are 8, 23, 14, 2, 2
  and 18.
- These data do not show that speed causes the drop. No run was made without the tie-break, the fastest
  Bombman run has the most sticky wins (10 of 30), and the open-loop GA, which uses the same rule, does not
  decline.
- Generations to 3M steps, sticky regime: 62–64 on Elecman, 97–99 on Gutsman, 108 on Cutman (K = 1);
  33 on Bombman, 39 on Fireman, 42 on Iceman (K = 4).
