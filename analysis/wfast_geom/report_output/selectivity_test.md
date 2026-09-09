# Budget-matched selectivity test (shufwrite) — does write *allocation* matter?

Question: after the minimal B3 showed `direct ≈ full` and `nocons/qonly` drop the HE
advantage, is the direct writeback's benefit due to **where it writes** (per-coordinate
allocation driven by W_fast content) or merely to adding energy anywhere?

Design (Mac, n=4 seeds × EH/HE, parity-validated harness):
- `shufwrite`: identical to `direct` (sleep writes γ·W_fast) except the γ·W_fast
  increment is **randomly permuted within each matrix** before being added to W_slow.
  Total write energy per matrix is exactly preserved; only the coordinate placement
  (and its sign alignment with W_fast/W_slow) is destroyed. Everything else identical.

## Results

| variant | HE gain40 mean±sd | paired Δ vs direct (dz) | HE−EH effect |
|---|---|---|---|
| full (archive) | +0.0239 ± 0.0081 | — | +0.0232 ± 0.0136 (dz 1.71) |
| direct | +0.0229 ± 0.0102 | — | +0.0230 ± 0.0137 (dz 1.68) |
| nocons | +0.0103 ± 0.0048 | −0.0126 ± 0.0061 (dz −2.06) | +0.0144 ± 0.0113 (dz 1.27) |
| qonly | +0.0094 ± 0.0022 | −0.0135 ± 0.0080 (dz −1.68) | +0.0155 ± 0.0067 (dz 2.31) |
| **shufwrite** | **+0.0091 ± 0.0024** | **−0.0138 ± 0.0094 (dz −1.47)** | +0.0151 ± 0.0033 (dz 4.51) |

EH arm stays ≈0/noise in every variant (as in minimal B3).

## Reading
- `shufwrite ≈ nocons ≈ qonly ≪ direct ≈ full` on the HE arm: **preserving the total
  consolidation energy but randomising which coordinates receive it removes the
  entire direct-write benefit** → the per-coordinate *allocation* of the write is
  causally necessary.
- Allocation is emergent: the increment at every coordinate is simply
  `γ · W_fast_i` (uniform scalar coefficient, zero explicit gating), so the "choice of
  where to write" comes entirely from the self-organized structure (magnitude + sign)
  of W_fast at sleep time. Destroying that structure with a same-norm shuffle kills
  the effect.
- This upgrades the earlier minimal-B3 reading: the coefficient is uniform, but the
  **allocation is not** — and the allocation is what matters. There is now causal
  evidence for **self-organized (emergent) selectivity of consolidation allocation**.

## What this does and does NOT show
- DOES: write-allocation selectivity is real and functionally necessary (energy-matched
  shuffle control); it emerges from W_fast content without any explicit selection
  objective or per-parameter gate.
- DOES NOT: success-gated selection (Q) — remains numerically and causally inert
  (qonly ≈ nocons); experience-level selection (choosing which tasks/experiences to
  rehearse) is absent; per-parameter meta-selection is absent. "Selective" here means
  coordinate-level allocation of the direct fast→slow write, not a gating mechanism.
- Caveats: n=4 seeds, single backbone/setting; dz large but noisy; HE-arm specific;
  retention (old-task forgetting) not tested (probes never sleep).

## Decision tree outcome
nocons ≠ direct (real) → qonly ran → qonly ≈ nocons → Q has no independent role.
shufwrite ≈ nocons, direct ≈ full ⇒ **active pathway = direct W_fast→W_slow write,
whose coordinate allocation (emergent from W_fast) is functionally required**.
