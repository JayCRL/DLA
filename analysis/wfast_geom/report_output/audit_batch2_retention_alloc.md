# Unified batch results — retention (B1), nosleep (B4), structured allocation (A4)

Mac parity-validated harness · HE arm · n=12 seeds (0–11) · gain@40 on unseen D +
same-protocol retention (end-of-history ppl on the 3 seen curriculum domains vs birth).

## Adaptation (gain@40 on D)

| variant | mean±sd | vs direct (paired) | t |
|---|---|---|---|
| direct | +0.0140 ± 0.0148 | — | — |
| nocons | +0.0027 ± 0.0103 | −0.0113 ± 0.0053 | −7.36 |
| nosleep | +0.0303 ± 0.0215 | +0.0163 ± 0.0093 | +6.06 |
| uniformwrite | +0.0035 ± 0.0094 | −0.0105 ± 0.0071 | −5.16 |
| topwrite | +0.0051 ± 0.0092 | −0.0089 ± 0.0074 | −4.18 |
| shufwrite | +0.0025 ± 0.0086 | −0.0115 ± 0.0073 | −5.45 |

Read 1: sleep costs some raw future-adaptation (nosleep highest), but that gain is
trivial if old-task retention matters (below). Among *sleeping* variants, direct is
the only one that preserves the consolidation benefit; every structured alternative
(uniform spread, top-20% concentration, shuffled) falls to ≈ no-consolidation level.

Read 2 (A4 refinement): selectivity is **not** reducible to energy concentration or
top-magnitude selection — even `topwrite` (same total energy on the top 20% |W_fast|
coordinates) fails. What matters is the **coordinate-matched write** (magnitude *and*
sign matched to W_fast itself): any deviation removes the benefit. "Useful structure"
is therefore fine-grained and sign-sensitive, not a coarse top-k heuristic.

## Retention (rel. ppl change on seen domains vs birth; more negative = better kept)

| variant | rel_forget mean±sd | ret_ppl mean |
|---|---|---|
| direct | −0.0192 ± 0.0083 | 35.88 |
| nocons | −0.0152 ± 0.0111 | 35.98 |
| nosleep | −0.0049 ± 0.0079 | 36.42 |
| uniformwrite | −0.0153 ± 0.0108 | 35.98 |
| topwrite | −0.0153 ± 0.0107 | 35.98 |
| shufwrite | −0.0153 ± 0.0112 | 35.98 |

Paired retention differences: `nocons vs direct Δ=+0.0040±0.0068 t=+2.03` (trend:
direct ≥ nocons) · `nosleep vs direct Δ=+0.0143±0.0048 t=+10.26` (nosleep markedly
worse).

## Interpretation (B1 + B4)

1. **The sleep boundary itself protects old-task knowledge** (nosleep ≪ all sleeping
   variants on retention; t≈10 vs direct). Decay + moment-reset at the boundary, not
   the write, is what mainly keeps seen tasks readable.
2. **The direct write carries the developmental/forward benefit**: among sleeping
   variants only direct preserves future-adaptation advantage; its content-matched
   coordinate allocation is necessary (uniform/top/shuffle ≈ no-consolidation).
3. Combining: the architecture is a small 2×2 —
   boundary(decay+reset) ⇒ retention; content-matched write ⇒ forward adaptation with
   selective allocation; success-gated Q ⇒ nothing (previous results). This gives the
   paper a clean, mechanism-complete story for "self-organized selective consolidation
   = a working continual-learning primitive" (forward benefit + protected memory), with
   all claims paired/energy-matched at n=12.
