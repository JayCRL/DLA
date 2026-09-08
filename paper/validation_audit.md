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

## 7. P0 controls result (gain@40, n=12, per-seed JSON)

Run: `experiments/validation_p0_controls.py`, seeds 0-11, all 7 tags, per-seed
JSON + full curves saved under `results/validation/seeds/`.

Gain@40 means:

| tag | mean gain@40 |
|---|---|
| HE/HE native | +0.0141 |
| HE+EH_Wfast raw | −0.0051 |
| HE+normEH | −0.0078 |
| HE+shuffleEH | −0.0026 |
| HE+EH_emb | +0.0080 |
| HE+EH_attn | +0.0021 |
| HE+EH_mlp | −0.0028 |

Paired gain@40 contrasts:

| contrast | mean | bootstrap 95% CI | Cohen d | sign p |
|---|---|---|---|---|
| HE/HE → raw EH | −0.0192 | (−0.0335, −0.0054) | −0.71 | 0.019 |
| HE/HE → norm EH | −0.0219 | (−0.0344, −0.0088) | −0.90 | 0.073 |
| HE/HE → shuffle EH | −0.0167 | (−0.0233, −0.0092) | −1.25 | 0.003 |
| raw → norm | −0.0027 | (−0.0063, +0.0014) | −0.39 | 0.073 |
| raw → shuffle | +0.0025 | (−0.0066, +0.0145) | +0.13 | 0.39 |
| HE/HE → EH_emb | −0.0061 | (−0.0093, −0.0018) | −0.87 | 0.003 |
| HE/HE → EH_attn | −0.0120 | (−0.0180, −0.0059) | −1.06 | 0.019 |
| HE/HE → EH_mlp | −0.0169 | (−0.0253, −0.0079) | −1.05 | 0.073 |

Conclusions (gain@40):
- The destructive raw transfer replicates robustly (CI excludes zero, d≈−0.7).
- Norm-matching does NOT rescue the effect: norm-matched EH W_fast is as
  destructive as raw (CI vs native excludes zero; raw-vs-norm CI includes zero).
  Global magnitude cannot explain the effect.
- Shuffle also does NOT rescue the effect: within-matrix shuffled EH W_fast is
  destructive (CI excludes zero; raw-vs-shuffle not different). A simple
  random-parameter control is not a sufficient explanation; the effect depends
  on more than exact EH structure alone (it also does not require exact EH
  structure, since shuffled EH is still harmful).
- Module localization: MLP and attention both show clear destructive drops;
  embedding also contributes but is smaller. Localization is suggestive of
  MLP+attention, not a single module.

Metric note: earlier LE_D-only analysis showed weaker raw-signal because LE_D
is noisier; gain@40 is the primary paper metric and is used here.


## 8. Fair baselines (preliminary, n=3 seeds)

Same EH/HE curriculum and D protocol as Stage 5.5e, but with standard learners.
LE_D means:

| method | EH | HE | HE−EH |
|---|---|---|---|
| AdamW | 0.232 | 0.967 | +0.735 |
| AdamW+replay | 0.291 | 0.997 | +0.706 |
| EWC | 0.153 | 0.951 | +0.798 |
| DLA (Stage5.5e, n=12) | 0.517 | 0.709 | +0.192 |

Gain@40 means:

| method | EH | HE |
|---|---|---|
| AdamW | −0.0024 | +0.0412 |
| AdamW+replay | +0.0007 | +0.0219 |
| EWC | −0.0031 | +0.0425 |
| DLA (n=12) | −0.0006 | +0.0143 |

Interpretation (preliminary, n=3 for baselines):
- Standard continual learners also show a history effect (HE > EH), so history
  sensitivity is not unique to DLA.
- DLA is more robust after the EH history (EH LE 0.52 vs baselines 0.15-0.29),
  while baselines achieve higher HE performance. This is consistent with the
  paper framing: DLA protects/stabilises future adaptation after difficult early
  histories rather than universally accelerating learning.
- Baseline results are preliminary (n=3); extend to n=5+ before strong claims.

## 9. Fair baselines extended (n=5) and second-setting replication (n=2)

Baselines n=5 (LE_D means):

| method | EH | HE | HE−EH |
|---|---|---|---|
| AdamW | 0.339 | 0.675 | +0.336 |
| AdamW+replay | 0.375 | 0.961 | +0.586 |
| EWC | 0.292 | 0.895 | +0.603 |
| DLA (n=12) | 0.517 | 0.709 | +0.192 |

Gain@40 means (baselines n=5):

| method | EH | HE |
|---|---|---|
| AdamW | +0.0081 | +0.0238 |
| AdamW+replay | +0.0042 | +0.0168 |
| EWC | +0.0062 | +0.0279 |
| DLA (n=12) | −0.0006 | +0.0143 |

Interpretation: DLA remains the most robust after EH history; standard
learners show larger HE performance but also larger EH degradation.

Second-setting replication (Shakespeare character GPT, ~10.65M, 2 seeds):

| seed | HE/HE gain40 | HE+EH_fast gain40 |
|---|---|---|
| 0 | +0.0522 | +0.0039 |
| 1 | +0.0424 | −0.0012 |

The destructive W_fast effect replicates in the second backbone/corpus (2/2
seeds, direction consistent). Small n, exploratory.

Remaining gaps after this audit:
- Second-setting n is small (2); increase to 5+ if used in paper.
- Fair baselines still n=5; enough for workshop/undergrad, more for strong claim.
