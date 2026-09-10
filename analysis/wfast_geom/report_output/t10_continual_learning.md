# T10 — 10-task cross-domain continual learning (A2/B2), Chinese char-GPT, seeds 0–7

Sequential learning of 10 different-domain slices (wiki→qa→physics→…→literature),
40 steps each, DLA with sleep variants `direct` / `nocons` / `nosleep` (n=8 seeds).
Forward = norm_slope10 of the train-loss (more negative = faster first-10-step drop);
Retention = end-of-history ppl on the earliest tasks (wiki, qa) after all 10 tasks
(lower = better kept).

## Forward learning speed (per-seed mean over 10 tasks)

| variant | norm_slope10 mean±sd | mean gain |
|---|---|---|
| direct | −0.00110 ± 0.00125 | −0.031 |
| nocons | −0.00100 ± 0.00129 | −0.038 |
| nosleep | −0.00090 ± 0.00125 | −0.025 |

Paired: `direct vs nocons Δ=−0.00011±0.00018 t=−1.64` (ns trend) ·
`direct vs nosleep Δ=−0.00020±0.00014 t=−3.93` · `nocons vs nosleep t=−2.38`.

## Retention of earliest tasks (end ppl after all 10 tasks)

| variant | end ppl (t1,t2) mean±sd |
|---|---|
| direct | 38.27 ± 5.74 |
| nocons | 39.03 ± 6.01 |
| nosleep | 39.51 ± 6.07 |

Paired: `direct vs nocons Δ=−0.77±0.60 t=−3.60` · `direct vs nosleep Δ=−1.25±0.34 t=−10.27` ·
`nocons vs nosleep Δ=−0.48±0.38 t=−3.61`.

Forward trend across the 10 tasks is flat/negative for every variant (no forward
plasticity collapse over the sequence).

## Reading (B2/B5)

In a 10-task sequential curriculum the **direct (self-organized) consolidation**
delivers the best of both worlds: strongest retention of the earliest tasks
(paired t ≈ −3.6 vs no-write, −10 vs no-sleep) while forward learning speed is not
worse than no-consolidation and significantly better than no-sleep boundary
(t ≈ −1.6 and −3.9). This is a *continual-learning primitive obtained for free from an
unselective fast→slow write* — no replay buffer, no importance gate, no EWC-style
penalty; the write allocation is emergent from W_fast (previous batches: energy-matched
shuffle destroys the benefit, n=12). Sleep-boundary (decay+reset) itself protects memory
(nosleep worst retention), and the direct write adds the forward/developmental content.

Caveats: n=8, one backbone, no tuned baselines (AdamW/replay/EWC) in this 10-task run;
forward-speed differences are small; gains noisy across domains.
