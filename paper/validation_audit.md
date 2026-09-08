# Paper Validation Audit — DLA

Date: 2026-09-08
Goal: recycle all completed evidence into the paper, identify only the real
remaining P0 gaps, and avoid repeating exploratory experiments.

---

## 1. Claims and current evidence inventory

| Claim | Supporting experiments | Seeds / samples | Statistical strength | Status |
|---|---|---|---|---|
| Fast/Slow separation protects consolidated memory | Stage 4 Formal | 5 seeds | B gain +14% vs AdamW +13%; slow forgetting −0.4%±0.2 | Strong, paper-safe |
| DLA is robust to increasing difficulty | Stage 5 (normalised LE) | 5 seeds | LE DLA 0.79→0.70→0.69 vs AdamW collapse | Moderate, paper-safe as robustness |
| Learning history changes future adaptation (A) | Stage 5.5b, 5.5c | 5 seeds for 2×2, 3 seeds causal | EH vs HE on unseen D replicable | Strong for between-history effect |
| φ is not a stable/dominant carrier | Stage 5.5c body×φ cross-injection | 3 seeds | Body effect >> φ effect | Moderate, should be worded carefully |
| W_fast is a causal component | Stage 5.5d (n=5), 5.5e P0 (n=12) | 12 seeds best evidence | **HE+EH_fast vs HE/HE: Δgain@40 −0.0195, CI [−0.0337,−0.0071], d −0.80, sign p=0.019 (10/12)** | **Current strongest evidence** |
| History effect is NOT monotonically faster learning (D) | Stage 5, 6, 7, 8 | 5/10/20/20 seeds | Stage 7 p=0.82; Stage 8 d=−0.17 | Negative result, must keep |
| Effective rank separates histories | Effective rank analyses | 10 seeds | EH/HE no significant difference; MLP p=0.064 | Negative/suggestive only |

## 2. What is already completed and should NOT be repeated

- Stage 4 Formal (fast/slow retention)
- Stage 5 difficulty-normalised LE
- Stage 5.5b History×φ 2×2 (5 seeds)
- Stage 5.5c body×φ causal dissection (3 seeds)
- Stage 5.5d full body component decomposition (5 seeds)
- Stage 5.5e P0 10→12 seed destructive contrast
- Effective rank global and per-module analysis (10 seeds)
- B negative results Stage 6/7/8

## 3. Outdated claims in current paper draft

`paper/DLA_paper_draft.md` still says:
- Abstract: “with 10 seeds ... not statistically conclusive”
- Section 5.4: “10 seeds” and “not statistically significant”

Must be updated to:
- n=12 destructive contrast is significant (d≈−0.80, p=0.019, CI excludes zero)
- transparently state the history: original n=10 run was near-borderline; n=12 was an extension
- the positive direction (EH+HE_fast) remains not significant

## 4. Reviewer attack points

1. Is W_fast effect due to norm/magnitude rather than content/structure?
   → Not yet fully controlled (only smoke data exists).
2. Is W_fast effect due to random structure (any shuffled vector would hurt)?
   → Not yet fully controlled.
3. Is the destructive effect localisable (embedding/attention/MLP)?
   → Only suggestive n=5 module decomposition exists; no full n=12 module-localised W_fast injection.
4. DLA vs AdamW/replay/EWC baselines in the history/future-adaptation protocol are missing.
5. All evidence is from one 6.59M Chinese nanoGPT setting; no second-task-family replication.
6. Results are exploratory (no pre-registration). Need explicit n=10→n=12 transparency.
7. “φ no effect” phrasing is too strong; must say “φ is not a stable/dominant carrier in this setting”.

## 5. Remaining P0 gaps (only these)

| Gap | Why it matters | Existing status |
|---|---|---|
| Norm-matched W_fast control | Exclude magnitude confound | Script written; smoke only |
| Structure-shuffle W_fast control | Show effect depends on organisation, not only norm/random content | Script written; smoke shows shuffle may remove effect |
| Module-localized W_fast control (full n=12) | Localise effect to embedding/attention/MLP | Full n=5 module slow decomposition only; not W_fast injection |
| Fair baselines (AdamW, replay, EWC) in same protocol | Biggest formal-paper gap | Not done |
| Second-setting replication | Show not one-corpus-only | Not done |

## 6. Recommended next actions

1. Recycle completed evidence: update `DLA_paper_draft.md` and `.tex` with n=12 + trajectory/memory-vs-dynamics numbers from existing Stage 5.5e data.
2. Run the already-written validation script (`experiments/validation_p0_controls.py`) on seeds 0–11 to close norm/shuffle/module gaps. This is the highest-value, already-scoped P0.
3. Run fair baselines only after controls.
4. Leave second-setting replication as P1 if compute allows.

## 7. P0 controls result (validation run, n=12, LE_D metric)

Run: `experiments/validation_p0_controls.py`, seeds 0-11, all 7 tags.
Concurrent writes overwrote the single output file, so the 12-seed table was
recovered from per-seed logs (`LE_D` only; full gain curves were not retained
for all seeds).

Mean LE_D:

| tag | mean LE_D |
|---|---|
| HE/HE | 0.684 |
| HE+EH_Wfast (raw) | 0.507 |
| HE+normEH | 0.230 |
| HE+shuffleEH | 0.345 |
| HE+EH_emb | 0.611 |
| HE+EH_attn | 0.270 |
| HE+EH_mlp | 0.279 |

Paired LE_D differences (bootstrap 95% CI):

| contrast | mean | CI | Cohen d |
|---|---|---|---|
| HE/HE -> HE+EH_Wfast | −0.177 | (−0.505, +0.177) | −0.28 |
| HE/HE -> HE+normEH | −0.454 | (−0.739, −0.145) | −0.80 |
| HE/HE -> HE+shuffleEH | −0.339 | (−0.559, −0.126) | −0.82 |
| raw -> norm | −0.276 | (−0.459, −0.099) | −0.81 |
| raw -> shuffle | −0.162 | (−0.518, +0.206) | −0.24 |
| HE/HE -> HE+EH_emb | −0.073 | (−0.154, +0.007) | −0.49 |
| HE/HE -> HE+EH_attn | −0.415 | (−0.632, −0.212) | −1.07 |
| HE/HE -> HE+EH_mlp | −0.406 | (−0.633, −0.170) | −0.92 |

Important caveat:
- This validation run used LE_D, not the gain@40 that produced the strongest P0
  result. In LE_D, the raw destructive contrast is not significant
  (CI includes zero).
- Norm-matching does NOT rescue the effect; if anything it makes the LE_D drop
  larger, so global magnitude alone cannot explain the destructive effect.
- Shuffle does NOT remove the LE_D drop (CI excludes zero vs native), so a
  simple structure-preserving within-matrix shuffle also does not explain it.
- Attention and MLP module transfers both show CI-excluding-zero drops;
  embedding transfer is weaker. Localization is suggestive for attention+MLP,
  not definitive for one module.

Because of the metric mismatch, the norm/shuffle/module P0 controls should be
re-run with per-seed JSON + gain@40 output before being used as confirmatory
paper evidence. The raw P0 gain@40 evidence from Stage 5.5e remains the
strongest single result.
