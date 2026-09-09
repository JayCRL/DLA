# DLA W_fast Geometry — Interim Report (Stage 5.5e, n=12)

Date: 2026-09-09  ·  Data: 24 saved bodies (seeds 0–11 × 1 histories) + archived p0 JSON + deterministic replays
Replays completed: 48/48  ·  parity max |Δppl| vs archived curves: mean 0.2398 (n=48)

## 1. Phase A — ΔW_fast geometry (per seed, global + module groups)

| stat (12 seeds) | mean | sd |
|---|---|---|
| ‖ΔW_fast‖ = ‖W_fast(HE)−W_fast(EH)‖ | 3.3405 | 0.2425 |
| ‖Δ‖ / ‖W_fast(HE)‖ | 0.9365±0.0709 | — |
| cos(W_fast_EH, W_fast_HE) | 0.5789±0.0281 | — |
| Δ energy share · embedding | 0.4887 | 0.0446 |
| Δ energy share · attention | 0.1498 | 0.0130 |
| Δ energy share · MLP | 0.3615 | 0.0316 |

**Null control (same-history, cross-seed):**

- EH_i vs EH_j : ‖Δ‖ mean 3.2638 · cos mean 0.6018
- HE_i vs HE_j : ‖Δ‖ mean 3.2065 · cos mean 0.6071

**Verdict:** the observed history-contrast ‖Δ‖≈3.3405 is the same magnitude as seed-to-seed noise (≈3.2638–3.2065), and cos(EH,HE)≈0.5789 ≈ same-history cross-seed cos (≈0.6018–0.6071). At the **global geometry level, history is not separable from seed noise**: the behavioural W_fast effect must be carried by finer structure (lower-dimensional directions), not by whole-state norm/similarity. Δ energy is concentrated in embedding (≈49%) while causal module controls implicated MLP/attention — norm-heavy directions are not the causal driver.

## 2. Gradient trajectory (instrumented replay of the stored D-probe)

Alignment of the per-step future gradient g_f = softplus(P)·∇L with the same-seed ΔW_fast direction:

| arm | cos(g_f0, Δ̂) mean | mean_abs | over 40 steps | g_f norm t=0 |
|---|---|---|---|---|
| EH/EH | -0.0147 | 0.0051 | — | 0.4999 |
| HE/HE | 0.0195 | 0.0066 | — | 0.5002 |
| EH_body+HE_fast | 0.0115 | 0.0049 | — | 0.4999 |
| HE_body+EH_fast | -0.0075 | 0.0042 | — | 0.5001 |

## 3. P1 — correlations with adaptation outcome (per seed, n=12, exploratory)

| predictor → outcome | r | ρ | perm p (5000) |
|---|---|---|---|
| delta_norm → out_HE/HE_gain40 | -0.187 | -0.203 | 0.562 |
| delta_norm → out_HE/HE_LE_D | -0.345 | -0.270 | 0.271 |
| delta_norm → out_EH/EH_gain40 | -0.185 | -0.049 | 0.616 |
| rel_delta_HE → out_HE/HE_gain40 | -0.140 | -0.245 | 0.677 |
| cos_eh_he → out_HE/HE_gain40 | +0.165 | +0.056 | 0.589 |
| cos_eh_he → out_HE/HE_LE_D | +0.252 | +0.140 | 0.435 |
| share_mlp → out_HE/HE_gain40 | +0.434 | +0.482 | 0.161 |
| tr_HE/HE_cos0 → out_HE/HE_gain40 | +0.870 | +0.832 | 0.000 |
| tr_HE/HE_cos0 → out_HE/HE_LE_D | +0.673 | +0.425 | 0.015 |
| tr_HE/HE_cos_mean_abs → out_HE/HE_gain40 | -0.185 | +0.049 | 0.561 |
| tr_HE/HE_g0_norm_gf → out_HE/HE_gain40 | +0.615 | +0.629 | 0.032 |
| tr_HE/HE_g0_norm_gf → out_HE/HE_LE_D | +0.290 | +0.098 | 0.362 |
| tr_EH/EH_cos0 → out_EH/EH_gain40 | -0.508 | -0.615 | 0.122 |
| tr_EH/EH_cos0 → out_EH/EH_LE_D | -0.531 | -0.457 | 0.080 |
| delta_norm → contrast_HE_delta | -0.225 | -0.266 | 0.488 |
| tr_HE/HE_cos_mean_abs → contrast_HE_delta | -0.297 | -0.580 | 0.341 |

_Exploratory, n=12, many comparisons. Two HE-arm signals survive permutation (cos0→gain@40 r≈+0.87, p≈0.000; g0-norm→gain@40 r≈+0.62, p≈0.03); macro-geometry scalars (‖Δ‖, cos(EH,HE), module shares) and the EH-arm signals do not._

## 4. P1b — initial-gradient alignment nulls

- mean cos(g0(HE), Δ_true) = 0.0199 ; shuffle-null = 0.0001 (sd 0.0003); cross-seed-null = 0.0038
- r(cos_true, gain@40 HE) = 0.8574 (perm p = 0.0005); r(shuffle-null) = 0.3020; r(cross-null) = 0.6587

## 5. P2 — PCA over ΔW_fast & projection prediction

Variance ratios (centered ΔW_fast, 12×D via Gram): PC1 0.205, PC2 0.113, PC3 0.091, PC4 0.084, PC5 0.083

| gf0 arm projection → outcome | PC | r | perm p |
|---|---|---|---|
| HE → gain_HE | PC1 | -0.834 | 0.001 |
| HE → gain_HE | PC2 | -0.720 | 0.007 |
| HE → gain_HE | PC3 | -0.674 | 0.019 |
| HE → gain_HE | PC4 | -0.279 | 0.376 |
| HE → LE_HE | PC1 | -0.654 | 0.020 |
| HE → LE_HE | PC2 | -0.527 | 0.079 |
| HE → LE_HE | PC3 | -0.577 | 0.043 |
| HE → LE_HE | PC4 | -0.328 | 0.301 |
| EH → gain_EH | PC1 | -0.386 | 0.125 |
| EH → gain_EH | PC2 | -0.199 | 0.352 |
| EH → gain_EH | PC3 | -0.090 | 0.695 |
| EH → gain_EH | PC4 | +0.173 | 0.711 |
| EH → LE_EH | PC1 | -0.376 | 0.236 |
| EH → LE_EH | PC2 | -0.482 | 0.058 |
| EH → LE_EH | PC3 | -0.589 | 0.030 |
| EH → LE_EH | PC4 | -0.198 | 0.516 |
| delta → gain_HE-gain_EH | PC1 | -0.800 | 0.001 |
| delta → gain_HE-gain_EH | PC2 | +0.055 | 0.866 |
| delta → gain_HE-gain_EH | PC3 | +0.012 | 0.967 |
| delta → gain_HE-gain_EH | PC4 | +0.167 | 0.612 |

## 6. Interim interpretation (mechanism-neutral)

1. **Macroscopic geometry is not the carrier.** The history contrast ΔW_fast is indistinguishable in norm/cosine from same-history cross-seed noise (Phase A null) — whole-state size/similarity cannot explain the causal swap effect, and the norm-heavy embedding part of Δ (≈49%) is not the causal module (MLP/attention are).
2. **Per-seed alignment of the future gradient with its own history contrast predicts adaptation** (HE arm): cos(g_f0, Δ) → gain@40 r≈+0.87 (perm p≈0.000), → LE_D r≈+0.67 (p≈0.015); g0-norm → gain@40 r≈+0.62 (p≈0.03). |cos| magnitudes are small (≈0.02) but ~5–50× the shuffle/cross-seed nulls, and true pairing (r≈0.86) beats cross-seed pairing (r≈0.66) and shuffle (r≈0.30). The EH-arm relation is negative but not significant — an asymmetry consistent with the destructive P0 result.
3. **Shared directions exist but are diffuse.** PC1 explains ≈20% of Δ variance (top-3 ≈41%), yet HE-gradient projections onto PC1–PC3 predict HE outcome (r≈−0.83/−0.72/−0.67; perm p≈0.001/0.007/0.019) and the Δ-score on PC1 predicts the HE−EH gain contrast (r≈−0.80, p≈0.001). PCA sign is arbitrary; the predictive content is |r| + permutation p.
4. **P3 counterfactual is justified by these first-round findings** (per-seed Δ-direction alignment and PC1–3 structure both predict adaptation). Recommended design: per-seed W_fast interpolation W_fast' = W_fast(HE) + α·Δ̂_i for α∈{−1, 0, +0.5, +1} (mirror on EH bodies), probed on D. Prediction: HE gain decreases monotonically as α moves toward EH (α>0) and recovers on the negative side. ~60 probes ≈ 30–45 min on 5 workers — not run yet (pending user confirmation).

## 7. Caveats & robustness (P1c)

- n=12, exploratory.
- **Leave-one-seed-out:** the headline signal is stable — tr_HE/HE_cos0→gain@40 r∈[+0.81,+0.91] across all 12 exclusions; g0-norm→gain@40 r∈[+0.54,+0.71]; the EH-arm cos0 relations also keep sign (r≈−0.5) under LOO.
- **FDR (Benjamini-Hochberg over the 16-test table):** only cos0(HE)→gain@40 survives q<0.1 (q≈0.003). g0-norm→gain@40 and cos0→LE_D are suggestive (q≥0.1); all other rows are null.
- Replays reproduce archived curves to mean max |Δppl|≈0.24 (~0.6% relative) — near-identical, not bit-exact (threaded float reductions); trajectory gradients are used for geometry only.
- g0 norms are ≈0.5 (post-clip, gated) and near-constant; the cos0 predictor is a direction, not a magnitude, effect.
- Cross-setting replication and larger n remain open.

## 8. Files

- geometry: `geometry_summary.json` ; replays: `replays/seeds/*.json` ; p1: `p1_correlations.json` ; p1b: `p1b_grad0_null.json` ; p1c: `p1c_robustness.json` ; p2: `p2_pca.json`
- figures: `report/fig1_geometry_nulls.png`, `report/fig2_trajectory_alignment.png`, `report/fig3_pca.png`