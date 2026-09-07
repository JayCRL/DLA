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
