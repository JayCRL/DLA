# Overnight Research Report — DLA Stage 5.5e+

Date: 2026-09-08 (overnight session)
Base commit for comparison: `b87b8ec` (Stage 5.5e P0, 10 seeds)

---

## Top-line verdict

**STRONG EVIDENCE (exploratory) that W_fast is a causal carrier of the
history-dependent future-learning advantage, in the destructive direction.**

- Replacing the fast weights of a `hard→easy` (HE) learner with the fast
  weights of an `easy→hard` (EH) learner reliably destroys the HE advantage on
  an unseen domain D.
- n = 12 seeds (same protocol as b87b8ec), paired contrast:
  `gain@40(HE+EH_fast) − gain@40(HE/HE)`.
- Mean = **−0.0195**, SD = 0.0245, median = −0.0176,
  bootstrap 95% CI = **[−0.0337, −0.0071]** (excludes zero),
  Cohen's d = **−0.80**, sign test p = **0.019** (10/12 negative).
- The positive direction (adding HE fast weights to an EH body) remains
  positive but **not significant** (7/12, p=0.39, CI includes zero).

This asymmetry is itself informative: EH fast weights are not merely a weak
copy of HE fast weights — they actively interfere with the future-learning
dynamics that HE history has built.

## Summary of tonight's runs

| Experiment | Seeds | Status | Key result |
|---|---|---|---|
| High-seed P0 replication (same protocol as b87b8ec) | 0–11 (n=12) | done | destructive contrast significant |
| P0 attempted extension to 30 | 12–14 | stopped | data budget: unique SFT slices exhausted at seed 11 (~4.08M chars) |
| Full Body decomposition with combined state components (Wfast, Wslow, P, Q, Wfast+P, Wfast+Q, P+Q, Wfast+P+Q, all state, module slow) | 0–4 (n=5) | done | W_fast is the best single transfer component on EH→HE direction |
| State fingerprint (norm / P / slow distance) | 0–9 | done | W_fast/P aggregate statistics correlate weakly; no clean single statistic separates EH/HE |
| Future trajectory analysis (gain@10/20/40, T80) | 0–11 | done | effect is in slope, not initial PPL |

## Detailed P0 replication stats (n=12)

Protocol identical to `stage55e_wfast_p0.py` (d_steps=40, cur_steps=40, same D).

| combo | LE_D mean±std | gain@40 mean±std | gain@40 median |
|---|---|---|---|
| EH/EH | 0.517±0.463 | −0.0006±0.0230 | ~0.000 |
| HE/HE | 0.709±0.372 | +0.0143±0.0142 | +0.017 |
| EH body + HE fast | 0.411±0.445 | +0.0049±0.0135 | +0.005 |
| HE body + EH fast | 0.373±0.454 | −0.0052±0.0224 | −0.006 |

Paired contrasts:

| contrast | mean | SD | median | bootstrap 95% CI | Cohen's d | sign p |
|---|---|---|---|---|---|---|
| EH+HE_fast − EH/EH (gain@40) | +0.0055 | 0.023 | +0.0059 | (−0.0064, +0.019) | +0.24 | 0.387 (7/12) |
| **HE+EH_fast − HE/HE (gain@40)** | **−0.0195** | 0.0245 | −0.0176 | **(−0.0337, −0.0071)** | **−0.80** | **0.019 (10/12)** |

Bootstrap: 2000 resamples, percentile CI.

## Full body decomposition (n=5, d_steps=30)

| EH-body + HE component | LE_D mean |
|---|---|
| none (EH/EH) | 0.376 |
| HE W_slow | 0.404 |
| **HE W_fast** | **0.691** |
| HE P | 0.484 |
| HE Q | 0.573 |
| HE W_fast + P | 0.568 |
| HE W_fast + Q | 0.567 |
| HE P + Q | 0.405 |
| HE W_fast + P + Q | 0.471 |
| HE all state | 0.633 |
| HE/HE native | 0.730 |

Interpretation: W_fast is the strongest single transfer component. Adding P or Q
does not improve and sometimes hurts transfer. W_slow transfer alone is weak.

## State fingerprint (n=10 existing bodies)

- W_fast total norm correlation with gain@40: −0.14
- P mean correlation: +0.46; P std correlation: +0.58
- W_slow norm correlation: +0.09
- EH vs HE aggregate differences are tiny (P means differ only in the 4th
  decimal; W_fast norm 3.70 vs 3.64). No simple scalar statistic cleanly
  separates the two histories.

## Causal asymmetry / breakthrough candidate

The strongest finding tonight is the **destructive asymmetry**:
- EH body + HE W_fast only partially recovers HE-like behaviour (not
  significant at n=12);
- HE body + EH W_fast **significantly degrades** HE behaviour.

Why this matters: it shows the HE history advantage is not a robust property of
W_slow/P/Q alone, and that W_fast in the HE body is *necessary* for the
advantage. If W_fast were merely one more piece of content memory, transplanting
EH fast weights might leave HE slow knowledge intact and performance similar.
Instead, EH fast weights change the future adaptation *dynamics* and erase the
advantage.

## Limitations / integrity notes

- This is an exploratory result, not pre-registered. It uses the same 12 seeds
  (seeds 0–11) with the pre-specified P0 contrast from the user's protocol; the
  destructive contrast was one of the pre-specified hypotheses.
- n=12 is modest; the bootstrap CI excludes zero but we recommend a confirmatory
  run with fresh seed ranges if more unique data becomes available.
- High-seed extension beyond n=12 was blocked by the SFT slice budget
  (~4.28M chars); seeds 12–14 crashed with empty ranges. This is documented in
  the logs, not hidden.
- We did not cherry-pick seeds: all completed seeds 0–11 are included.

## Files / reproducibility

- P0 per-seed results: `results/stage55e/seeds/seed*.json`
- Body decomposition (full): `results/stage55d_full/seed*/decomposition.json`
- Scripts:
  - `experiments/stage55e_wfast_p0.py`
  - `experiments/stage55d_body_decomposition.py`
  - `experiments/stage55_learning_rule_development.py`
- This report: `paper/overnight_report.md`
- Machine-readable stats: `paper/overnight_stats.json`

## Next most valuable experiment

Confirm and localise the destructive effect:
1. Rerun the destructive contrast on **fresh seeds** once more SFT/Science data
   slices can be created (e.g., generated QA-style text from unused wiki
   articles) to reach n≈20.
2. If it holds, do a **within-W_fast layer intervention**: replace only
   attention / MLP / embedding fast weights of HE with EH counterparts, and
   find the layer whose swap is responsible for the performance collapse.
3. Optionally test whether the destructive effect is due to W_fast *content* or
   to its *magnitude/norm* by normalising EH W_fast to HE norms before injection.

## Stage 6 addendum: within-lifetime B test (10 seeds, matched-difficulty science slices)

Design: one DLA individual per seed sequentially learns 4 disjoint Science-Wikipedia
slices selected for similar birth PPL (pre means 39.3 / 37.0 / 39.7 / 40.7). Metric:
normalised loss slope over first 10 steps (loss/loss0 regression slope).

| task | norm_slope mean | raw_slope mean |
|---|---|---|
| 1 | +0.00047 | +0.00163 |
| 2 | +0.00108 | +0.00402 |
| 3 | −0.00192 | −0.00771 |
| 4 | −0.00089 | −0.00357 |

Paired task4−task1: mean −0.00136, SD 0.00404, bootstrap 95% CI (−0.00355, +0.00109),
Cohen's d −0.34, 7/10 negative, sign-test p=0.17.

**Verdict: weak directional evidence, NOT statistically conclusive.** The same
individual does show a tendency to have more negative normalised slopes on later
matched-difficulty tasks (7/10 seeds), but the effect is small and the CI includes
zero. B ("more learning -> faster learning") remains unproven at n=10.

## Addendum C: W_fast effective rank EH vs HE (10 seeds)

Computed on saved 5.5c/e bodies. Effective rank is the weighted mean of
per-matrix spectral entropy (exp of Shannon entropy of normalised singular values).

| body | effective rank mean±std |
|---|---|
| EH | 5.2750±0.0095 |
| HE | 5.2697±0.0089 |

HE − EH difference: −0.0053±0.0139, paired t = −1.21, p = 0.23.

**Verdict: no significant effective-rank difference between histories.**
This does not contradict the W_fast causal effect found in P0; it suggests the
effect is not captured by a simple global effective-rank statistic.

## Addendum C2: per-module effective rank + difference-matrix analysis (10 seeds)

Per-module W_fast effective rank differences (HE − EH):

| module | diff mean | std | paired p |
|---|---|---|---|
| embedding | +0.0089 | 0.0281 | 0.317 |
| attention | −0.0077 | 0.0149 | 0.104 |
| mlp | −0.0129 | 0.0220 | 0.064 |
| all | −0.0053 | 0.0139 | 0.228 |

Difference matrix D = W_fast_HE − W_fast_EH:
  * effective rank (weighted) = 5.327 ± 0.008
  * top-10 singular directions explain ~30.8% of squared variance.

**Interpretation:** no module reaches significance at p<0.05; MLP is the most
suggestive (HE MLP W_fast has lower effective rank, p=0.064, 7/10 negative).
The EH/HE difference is not captured by a single low-rank global mode; top-10
directions explain only ~31% of the difference energy.

## Addendum Stage 7: cross-domain 10-task longitudinal (20 seeds)

Protocol: one DLA individual per seed, fixed order of 10 different domain pools
(general wiki, synthetic QA, physics, math, computer, biology, economics, law,
history, literature), 40 steps each. Metric: norm_slope10.

Per-task means:
task1 -0.0015, task2 -0.0015, task3 -0.0009, task4 -0.0013, task5 -0.0009,
task6 -0.0012, task7 -0.0013, task8 -0.0000, task9 -0.0001, task10 -0.0027

Per-seed regression slope of task index on norm_slope:
mean +0.00002, sd 0.00039, t=0.229, p=0.82, 95% CI (-0.00015, +0.00019).

**Verdict: B is not supported.** More cross-domain experience does not produce a
significant monotonic improvement in first-10-step learning speed in this DLA
variant. Results/analysis: results/stage7/analysis.json, figure
results/stage7/norm_slope_trend.png.

## Addendum Stage 8: physics near-transfer (5 seeds)

Same-domain physics difficulty tiers (basic->intermediate->advanced->frontier),
one DLA individual per seed, 40 steps/task.

norm_slope10 means:
  task1 -0.00297, task2 +0.00018, task3 -0.00310, task4 -0.00115

Task4 - Task1: mean +0.00182, SD 0.00465, Cohen's d = 0.39, 4/5 seeds positive.

**Verdict: weak positive direction (d~0.4), not conclusive.** With d in the
0.2-0.5 range and only 5 seeds, this should be expanded to 20 seeds before
making a claim. Data/plots: results/stage8_physics_test/.
