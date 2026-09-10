# A1 final — write-strength (γ) and sleep-decay (δ) sweeps for allocation selectivity

Completed run: `a1x` (Mac, M2, parity-validated harness), launched 09-10 09:55, finished 09-10 11:31
(`~/llm-lab/dla_audit/a1x_run.log`, last line `A1X_DONE`).

Supersedes `a1_gamma_prelim.md` (which reported γ ∈ {1.0, 0.5, 0.05} at **n=3**). This report expands
the γ sweep to **n=12** at γ ∈ {0.5, 1.0, 1.5}, adds a **sleep-decay** sweep (seeds 0–5), and anchors
every arm against the no-consolidation floor (`nocons`).

**Notation — γ here is a *scale*, not the coefficient.** All γ values in this report are the
`--gamma` multiplier passed to `audit_b3.py`, which sets
`gam = tv["consolidate_fast_direct"] * gamma_scale`. The underlying `consolidate_fast_direct` is a
**learned** parameter (`dla/transformer_dla.py`: config default `0.15`, applied through a sigmoid), so
the effective per-sleep write coefficient is `0.15·scale` at config init and seed-dependent after
training. "γ=0.5 / 1.0 / 1.5" therefore means **0.5× / 1× / 1.5× the learned coefficient**, and
"γ=1.0" is the default setting used everywhere else in the project — it is *not* a coefficient of 1.0.
(Notation for the manuscript: γ_scale.)

**Question.** `a1_gamma_prelim.md` found the direct-vs-shuffled gap shrinking at weak write strength and
flagged that n=3 could not distinguish a real dose-response from a knife-edge effect at the single
default γ=1.0. n=12 decides between those readings.

Reproduce all numbers with: `~/.venv/bin/python analysis/wfast_geom/a1_final.py`

---

## 1. Data provenance

| tag | arm | dir | seeds | notes |
|---|---|---|---|---|
| `""` | γ=1.0 (default) | `~/llm-lab/dla_audit/` | 0–11 | seeds 3–11 from the unified batch (03:57–05:22); **seeds 0–2 re-run 06:20–06:44** after an earlier overwrite |
| `g0.5` | γ=0.5 | `.../g0.5/` | 0–11 | seeds 0–2 earlier, 3–11 in `a1x` |
| `g1.5` | γ=1.5 | `.../g1.5/` | 0–11 | all in `a1x` |
| `g0.05` | γ=0.05 | `.../g0.05/` | 0–2 | **n=3 only**, retained as a weak-write tail point |
| `nocons` | no write | `.../nocons/` | 0–11 | reference floor (unified batch) |
| `d0.25` / `d0.75` | δ=0.25 / 0.75 | `.../d0.25`, `.../d0.75` | 0–5 | `direct` only, `a1x` |

Reported statistic throughout: paired `direct − shufwrite` (energy-matched within-matrix coordinate
shuffle of each sleep's `γ·W_fast` increment) on HE-arm `gain@40` of the unseen domain D. Paired t,
Cohen's dz, and two-sided 95% CI (t-table; scipy is not installed in `~/.venv`).

## 2. Per-seed gain@40 (direct vs shufwrite)

| seed | γ=0.05 d | γ=0.05 s | γ=0.5 d | γ=0.5 s | γ=1.0 d | γ=1.0 s | γ=1.5 d | γ=1.5 s |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.0127 | +0.0114 | +0.0126 | +0.0108 | +0.0352 | +0.0139 | +0.0475 | +0.0115 |
| 1 | +0.0101 | +0.0086 | +0.0103 | +0.0091 | +0.0217 | +0.0074 | +0.0347 | +0.0060 |
| 2 | +0.0128 | +0.0089 | +0.0142 | +0.0110 | +0.0245 | +0.0077 | +0.0435 | +0.0081 |
| 3 | — | — | +0.0101 | +0.0065 | +0.0180 | +0.0078 | +0.0186 | +0.0081 |
| 4 | — | — | −0.0038 | −0.0126 | −0.0016 | −0.0047 | −0.0064 | −0.0024 |
| 5 | — | — | −0.0138 | −0.0103 | −0.0125 | −0.0134 | −0.0061 | −0.0117 |
| 6 | — | — | −0.0069 | −0.0026 | −0.0025 | −0.0037 | +0.0055 | −0.0030 |
| 7 | — | — | +0.0044 | −0.0032 | +0.0116 | +0.0019 | +0.0167 | +0.0017 |
| 8 | — | — | −0.0036 | −0.0026 | +0.0126 | −0.0054 | +0.0302 | −0.0037 |
| 9 | — | — | +0.0301 | +0.0200 | +0.0382 | +0.0167 | +0.0623 | +0.0204 |
| 10 | — | — | +0.0118 | +0.0102 | +0.0180 | +0.0071 | +0.0207 | +0.0093 |
| 11 | — | — | +0.0056 | +0.0058 | +0.0077 | −0.0027 | +0.0224 | +0.0036 |

## 3. γ sweep (HE arm, paired)

| γ | n | direct | shufwrite | gap | sd(gap) | t | dz | 95% CI |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 3 | +0.0118 | +0.0096 | +0.0022 | 0.0014 | +2.64 | +1.52 | [−0.0014, +0.0058] |
| 0.5 | 12 | +0.0059 | +0.0035 | +0.0024 | 0.0046 | +1.82 | +0.52 | [−0.0005, +0.0053] |
| **1.0** | 12 | +0.0143 | +0.0027 | **+0.0115** | 0.0072 | **+5.54** | +1.60 | [+0.0069, +0.0161] |
| **1.5** | 12 | +0.0241 | +0.0040 | **+0.0201** | 0.0146 | **+4.78** | +1.38 | [+0.0109, +0.0294] |

**Dose-response (paired, same seeds, across γ):**

| contrast | mean Δ | sd | t | dz | n |
|---|---|---|---|---|---|
| gap(γ=1.0) − gap(γ=0.5) | +0.0091 | 0.0072 | **+4.42** | +1.28 | 12 |
| gap(γ=1.5) − gap(γ=1.0) | +0.0086 | 0.0084 | **+3.57** | +1.03 | 12 |
| gap(γ=1.5) − gap(γ=0.5) | +0.0177 | 0.0147 | **+4.17** | +1.20 | 12 |

Both *adjacent* contrasts are significant, so the gap is not merely monotone in point estimates: each
step in write strength buys a statistically reliable increment.

## 4. Anchoring against the no-consolidation floor (n=12)

`nocons` (no write at all) = **+0.0027 ± 0.0103**.

| arm | vs `nocons` | sd | t | dz | 95% CI |
|---|---|---|---|---|---|
| direct, γ=0.5 | +0.0032 | 0.0047 | +2.38 | +0.69 | [+0.0002, +0.0062] |
| **shufwrite, γ=0.5** | +0.0008 | 0.0035 | +0.83 | +0.24 | [−0.0014, +0.0031] |
| direct, γ=1.0 | +0.0116 | 0.0056 | **+7.20** | +2.08 | [+0.0081, +0.0151] |
| **shufwrite, γ=1.0** | +0.0001 | 0.0041 | +0.05 | +0.01 | [−0.0025, +0.0027] |
| direct, γ=1.5 | +0.0215 | 0.0113 | **+6.58** | +1.90 | [+0.0143, +0.0287] |
| **shufwrite, γ=1.5** | +0.0013 | 0.0044 | +1.05 | +0.30 | [−0.0015, +0.0042] |

Two readings, both load-bearing:

1. **Shuffled allocation sits on the no-consolidation floor at every write strength** (all three
   |dz| ≤ 0.30, all n.s.). Cranking the write up by 50% does not make a misallocated write useful —
   it is wasted energy, not delayed benefit. This is the sharpest form of the selectivity claim so far:
   the *placement* is what carries the effect, not the *amount*.
2. **The content-matched write ramps off that floor with γ**: direct − nocons ≈ +0.003 (γ=0.5, barely)
   → +0.012 (γ=1.0) → +0.022 (γ=1.5).

## 5. Is it linear in γ? (first-order account)

`docs/mechanism_chain.md` §3 predicts `ΔL_D ≈ ⟨∇_{W_slow} L_D, A⟩` with `A = γ·W_fast`, i.e. a
first-order effect **proportional to γ**.

| γ | observed gap | linear fit `k·γ` (k=0.0123) | deviation | gap/γ |
|---|---|---|---|---|
| 0.5 | +0.0024 | +0.0061 | −0.0037 | +0.0048 |
| 1.0 | +0.0115 | +0.0123 | −0.0007 | +0.0115 |
| 1.5 | +0.0201 | +0.0184 | +0.0017 | +0.0134 |

Same shape using `direct − nocons` (gap/γ = +0.0064 / +0.0116 / +0.0143): monotone, **approximately
linear for γ ≥ 1.0**, with the γ=0.5 point falling *below* the linear trend. So the first-order account
captures the regime the paper actually uses, and the deviation is in the safe direction (the account
does not over-predict at weak write); weak writes are somewhat *less* useful than proportionality
implies — a soft onset, not a threshold.

## 6. Sleep-decay (δ) sweep — direct only, seeds 0–5

δ is the per-sleep decay `W_fast ← δ·W_fast` (default 0.5). Lower δ ⇒ less fast-trace survives a
boundary and therefore smaller writes at later boundaries.

| δ | gain@40 | sd | vs δ=0.50 | sd(Δ) | t | dz |
|---|---|---|---|---|---|---|
| 0.25 | +0.0013 | 0.0098 | −0.0129 | 0.0091 | **−3.48** | −1.42 |
| 0.50 (default) | +0.0142 | 0.0178 | — | — | — | — |
| 0.75 | +0.0346 | 0.0327 | +0.0204 | 0.0176 | **+2.84** | +1.16 |

Per-seed, δ=0.75: `+0.0746 +0.0563 +0.0589 +0.0187 +0.0010 −0.0016`; δ=0.25:
`+0.0082 +0.0048 +0.0118 +0.0037 −0.0058 −0.0146`.

Monotone in the same direction as γ (more retained structure ⇒ larger benefit), but this sweep is
**n=6 and variance-dominated** — see caveats.

---

## 7. Verdicts

1. **The allocation-selectivity benefit is not a knife-edge at one hyperparameter.** The paired gap
   rises monotonically across a **3× range of γ_scale** (0.5 → 1.5) and each adjacent step is significant
   (t=+4.42, t=+3.57). The n=3 preliminary's "write-strength dependent" reading survives at n=12.
2. **Misallocation is not rescued by more energy.** `shufwrite` ≈ `nocons` at γ = 0.5, 1.0 *and* 1.5
   (all n.s., |dz| ≤ 0.30), while the matched write moves +0.003 → +0.012 → +0.022 off the same floor.
   The effect tracks *coordinate-matched placement*, not write magnitude.
3. **Approximately linear in γ for γ ≥ 1.0** (as the first-order account predicts), with a sub-linear
   soft onset at γ=0.5. Consistent with `ΔL_D ≈ ⟨∇L_D, A⟩`, `A = γW_fast`.
4. **Sleep decay behaves the same way** (δ=0.25 ≪ 0.50 < 0.75, t=−3.48 / +2.84), i.e. the benefit
   scales with how much structure survives to be written — but at n=6 this is indicative only.
5. **γ=0.5 weakens but does not eliminate the effect**: the γ=0.5 gap is itself not significant
   (t=1.82, CI includes 0) and its excess over `nocons` is marginal (+0.0032, t=2.38, CI lower bound
   +0.0002), yet it is *reliably smaller* than the γ=1.0 gap. The correct statement is
   "graded benefit", not "no consolidation below a threshold".

## 8. Caveats

- **Provenance**: the γ=1.0 (default) arm mixes seeds 3–11 from the unified batch with **re-runs of
  seeds 0–2**. Recomputing `direct` (n=12) from this directory gives **+0.0143**, while
  `docs/mechanism_chain.md` / the paper quote **+0.0140** from the pre-A1 unified-batch aggregate. The
  shift (0.0003) is ~2% of one sd (0.0148) and is attributable to harness non-determinism
  (eval-RNG/threading; the parity check itself carries mean |Δppl| ≈ 0.24). **The paper's +0.0140 is
  the legacy number and should be kept unless the whole arm is re-run under one provenance**; the
  γ-comparisons above are unaffected because γ=0.5 and γ=1.5 arms are internally consistent
  single-run sets.
  The same 0.0003 offset applies to the γ=1.0 arm means and therefore to its `nocons` anchoring:
  this report uses this directory (`direct` +0.0143, `shufwrite` +0.0027, so `shufwrite − nocons`
  = **+0.0001**), whereas the paper §4.4 quotes the unified-batch arm means (`direct` +0.0140,
  `shufwrite` +0.0025, hence `shufwrite − nocons` = **−0.0002**). The **paired gap is identical**
  (+0.0115) under both, which is why the dose-response figure plots the gap as its primary panel;
  only the separately-reported arm means move by 0.0003.
- **γ=0.05 is n=3** (`t=+2.64`, CI includes 0 once the correct df=2 critical value 4.303 is used).
  It is shown as a weak-write tail point only and supports no claim.
- **δ sweep is n=6**, and its Δ at δ=0.75 (sd 0.0176, dz +1.16) is driven by seeds 0–2
  (`+0.075 +0.056 +0.059`) — i.e. by the same seeds whose default-γ=1.0 files were re-run. Treat §6 as
  a hypothesis for an n=12 confirmation, not a result.
- Single backbone, single protocol, HE arm only; D-probes never sleep, so retention is not measured
  here (same limitation as `selectivity_test_n12.md`).
- Absolute per-seed levels are noisy (seeds 4–6 go negative); only within-seed, within-arm paired
  contrasts are used, as everywhere else in this project's audit line.
- **Not yet integrated into the paper.** This report is a prerequisite for the §4.4 / Discussion
  falsifier discussion of hyperparameter robustness; the "knife-edge" review objection is answered by
  §3+§4 but is not yet written into `paper/DLA_paper_draft.md`.
