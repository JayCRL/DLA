# Selectivity-of-allocation test — n=12 confirmatory (seeds 0–11, Mac)

Methods identical to the n=4 run (parity-validated Mac harness). Extension seeds 4–11
added to the existing 0–3. `direct`/`nocons`/`shufwrite` × EH/HE = 96 bodies/probes total
(0–3 n=4 done earlier; 4–11 n=8 in this run). Full = server archives (same protocol).

HE arm gain@40 (n=12) and paired t vs full:

| contrast | mean Δ | sd(Δ) | t (paired, n=12) |
|---|---|---|---|
| direct − full | +0.0005 | 0.0028 | +0.59 |
| nocons − full | −0.0107 | 0.0067 | **−5.53** |
| shufwrite − full | −0.0101 | 0.0072 | **−4.84** |
| direct − nocons | +0.0112 | 0.0073 | **+5.30** |
| direct − shufwrite | +0.0106 | 0.0075 | **+4.86** |
| nocons − shufwrite | −0.0006 | 0.0051 | −0.38 |

Variant means on HE: direct +0.0148±0.0150 · nocons +0.0036±0.0096 · shufwrite +0.0042±0.0095 ·
full +0.0143 (archive).

## Verdicts (n=12)
1. **direct ≈ full** (t=0.59): the direct W_fast→W_slow write reproduces the full history effect.
2. **nocons ≪ direct** (t=5.30) and **shufwrite ≪ direct** (t=4.86): removing consolidation — or keeping
   its total energy but destroying which coordinates receive it (same-norm within-matrix shuffle) —
   removes the benefit. nocons ≈ shufwrite (t=−0.38).
3. ⇒ The **coordinate allocation of the direct write is functionally necessary**: the write is
   selective in *where* it lands, and that selection is **self-organized** — the increment is just
   `γ·W_fast` (uniform scalar coefficient), so the allocation is entirely determined by the
   structure (magnitude/sign) of W_fast at sleep time, with no explicit selection/gating objective.

Caveats: single backbone/setting; retention (old-task forgetting) not measured (probes never sleep);
raw EH/HE future-adaptation gap is itself noisy across seeds (seeds 4–6 even reverse), so the
robust object is the *within-seed, within-arm* contrast above, not the absolute HE level; qonly (n=4)
remains ≈ nocons, so the success-gated Q channel stays causally inert.
