# Developmental Learning Architecture: Fast/Slow Memory, Difficulty Robustness, and the Search for Learning-Rule Development

**Draft v0.1 — for workshop / arXiv first posting**

---

## Abstract

We study a family of neural architectures in which the learning rule itself is a
developmental object. Our Developmental Learning Architecture (DLA) keeps a
standard Transformer/MLP skeleton unchanged but attaches per-parameter fast
weights, plasticity traces, and sleep consolidation, so that knowledge and the
ability to acquire knowledge are separated into slow and fast state. We report
four main findings. First, DLA's fast/slow separation reproduces the
adaptation quality of tuned AdamW fine-tuning while keeping the consolidated
slow memory essentially immune to forgetting in sequential language-domain
adaptation (5 seeds). Second, DLA is robust to a progressively harder
curriculum: its normalized learning efficiency does not collapse when task
difficulty increases, whereas static fine-tuning degrades. Third, a causal
dissection shows that learning history strongly changes future learning
performance on an unseen domain, but the 9-dimensional learning-rule
parameters (tempos) are *not* the carrier of this effect. Fourth, body-state
cross-injection localises the carrier to the fast weights: transplanting the
fast weights from a hard→easy learner into an easy→hard body changes the
future adaptation *trajectory* rather than the initial perplexity. However,
with 10 seeds this fast-weight transfer effect is directionally consistent but
not statistically conclusive (one near-significant contrast). We therefore
frame this paper as an honest mechanistic study: learning history shapes the
learner, the effect is carried by accumulated fast-weight state, and current
meta-updates of scalar learning-rule parameters do not yet produce stable
improvements in future learning.

---

## 1 Introduction

Large models are typically treated as fixed functions that are trained and then
deployed. This paper asks a different question: *can the way a model learns
change as a result of what it has learned?* Drawing on ideas of developmental
plasticity and fast/slow memory, we build a minimal architecture that has:

- **Slow weights** — consolidated long-term knowledge;
- **Fast weights** — a current learning trace;
- **Plasticity P** — a per-parameter state controlling how much each weight can
  change;
- **Consolidation Q** — an eligibility trace used during sleep.

We deliberately do *not* modify the Transformer skeleton. The developmental
state lives beside the standard weights and is updated by an online learning
rule, with periodic sleep consolidation.

The central empirical question of this project is whether "learning history"
can shape the *future learner itself*, not just the knowledge it stores. We
operationalise this by comparing learners who experienced the same three
domains in different orders (easy→hard vs hard→easy) and then measuring how
they adapt to an unseen fourth domain.

## 2 Related Work

- **Fast weights and memory**: recent and classic work on fast-weight
  mechanisms (e.g., Ba et al., Schmidhuber, Hinton & Plaut) motivates a
  separation between fast and slow stores.
- **Metaplasticity / continual learning**: studies of synaptic tagging,
  metaplasticity and stability-plasticity balance (Abraham & Robins;
  Zenke et al.) provide biological and algorithmic priors.
- **Learning to learn / meta-learning**: ANML (Beaulieu et al.) and related
  methods show that selective plasticity can be meta-learned.
- **Learned optimizers / learning rules**: learned optimizers and
  differentiable plasticity make the update rule itself a trainable object.
- **The distinction in this paper**: existing meta-learning asks "can a good
  rule be learned"; we ask whether, within a single lifetime, the learning
  rule (and its associated body state) continues to develop, and which state
  actually carries development forward.

## 3 Method

### 3.1 DLA on an MLP core (Stages 1–3)

For small-scale mechanism experiments we use a two-layer MLP. Every connection
has:

- slow weight `W_slow`
- fast weight `W_fast`
- plasticity `P`
- eligibility `Q`

Effective weight:

```
W_eff = W_slow + softplus(P) * W_fast
```

Experience update (wake) uses a per-connection Learning Rule Network `F_phi`
that receives local and cognitive signals and outputs coefficients for a
Hebbian basis and a teaching basis (`delta x^T`). Sleep consolidates Q into
W_slow and decays fast traces. Meta-training can unroll entire "mini-lifetimes"
of two tasks with a retention loss, so the stability-plasticity trade-off is
part of the objective.

### 3.2 DLA on a standard GPT (Stage 4 onward)

For the Transformer experiments we wrap each `nn.Linear` and embedding of a
standard nanoGPT (6 layers, 8 heads, 256 dim, 7280 Chinese characters) with
per-matrix states `W_fast, P, Q` and Adam moments. The backbone code is not
changed; each forward uses `W_eff` and each backward supplies the teaching
signal `g = dL/dW_eff`.

The nine scalar *tempos* `phi = {eta_fast, eta_plast, fast_decay, stability,
alpha_q, consolidation rates}` start as DNA priors. In a "meta" variant they
can be updated by one meta-gradient step on future-task losses.

### 3.3 Causal dissection

We decompose the developmental state ("body") into slow weights, fast weights,
plasticity P and eligibility Q. Cross-injection swaps one component between
two individuals with different histories while holding the rest fixed. We
then measure future adaptation on a held-out domain D under identical data,
update rule, budget and evaluation.

## 4 Experimental Setup

- Backbone for Transformer experiments: 6.59M Chinese character-level GPT,
  pretrained on ~9.3M tokens of simplified Chinese Wikipedia (val PPL ~34.5).
- Curriculum domains: general Wikipedia, SFT-style QA, science Wikipedia.
- Held-out domain D: a science-Wikipedia slice never used in the curriculum.
- Each individual trains each domain with the same budget (40–100 Adam-style
  steps of 32×128 tokens depending on stage).
- Metrics: raw relative gain, difficulty-normalized LE, T80 (steps to 80% of
  the individual's own best gain), backward transfer/forgetting on previous
  domains, and plasticity/state traces.

All scripts and per-seed JSONs are in the public repository:
`https://github.com/JayCRL/DLA` (see `experiments/` and `docs/experiments.md`).

## 5 Results

### 5.1 Fast/slow separation protects consolidated memory (Stage 4, 5 seeds)

| metric | AdamW (tuned) | DLA fast | DLA slow |
|---|---|---|---|
| B-domain adaptation gain | +13.0% ± 2.4 | +14.0% ± 2.5 | +0.1% |
| A forgetting after B | +10.1% ± 2.2 | +7.3% ± 1.2 | **−0.4% ± 0.2** |

DLA matches tuned AdamW on adaptation while the consolidated slow memory is
essentially stable. This is our cleanest positive result.

### 5.2 Difficulty robustness (Stage 5, 5 seeds)

With a fixed 4% relative-gain threshold, both AdamW and DLA appear to "get
worse" as tasks get harder, but this is a metric confound. Under normalized LE,

| arm | LE across curriculum | LE slope |
|---|---|---|
| AdamW | [0.63, 0.87, 0.25] | −0.19 |
| DLA fast | [0.79, 0.70, 0.69] | −0.05 |
| DLA slow | [0,0,0] | 0 |

DLA does not collapse when difficulty increases.

### 5.3 Scalar learning-rule parameters are not the carrier (5.5b/c, 5 seeds)

2×2 cross-injection (body × phi) on unseen D:

| Body / phi | LE_D mean |
|---|---|
| EH/EH | 0.522 |
| EH/HE | 0.148 |
| HE/EH | 0.898 |
| HE/HE | 0.900 |

The body (slow/fast/plasticity state) dominates; swapping the 9 tempos has
little and inconsistent effect.

### 5.4 Fast weights carry (part of) the history effect (5.5d/5.5e, 10 seeds)

Body decomposition (5.5d, 3 seeds):

- EH body + HE fast weights → LE_D jumps from 0.33 to 0.91;
- HE body + EH fast weights → LE_D drops from 0.96 to 0.00.

Extended P0 trajectory experiment (10 seeds):

| combo | gain@40 mean | sign |
|---|---|---|
| EH/EH | −0.001 | — |
| HE/HE | +0.014 | — |
| EH body + HE fast | +0.005 | +0.006 vs EH/EH (6/10 positive) |
| HE body + EH fast | −0.007 | −0.021 vs HE/HE (8/10 negative) |

Initial perplexity differences are small; the differences appear in the
adaptation *slope*. This supports the interpretation that fast weights change
future learning dynamics rather than simply encoding memory. However, with
10 seeds the positive transfer contrast is not statistically significant
(sign-test p≈0.38), while the negative "destructive" contrast is borderline
(p≈0.055). Fast-weight transfer is therefore an important but not self-sufficient
carrier; it interacts with the rest of the body.

## 6 Discussion

**What we now believe**
1. A developmental architecture can decouple fast adaptation from slow memory
   preservation.
2. A hard→easy curriculum produces a learner that adapts faster on an unseen
   domain than an easy→hard learner; this history effect is reproducible.
3. The effect is mediated by accumulated body state, especially fast weights,
   not by scalar meta-learned tempos.
4. Fast-weight transfer changes the *dynamics* of future adaptation, not merely
   the initial perplexity.

**What we do not yet claim**
- That meta-updating scalar learning rules yields stable lifelong improvements
  in transfer (5.5b/5.5c negative/nuanced).
- That fast-weight transfer alone is a sufficient causal carrier (5.5e
  underpowered).
- That this scales beyond a 6.59M model.

**Limitations**
- Single small backbone, single language, synthetic domain slices.
- No standard continual-learning baselines (EWC, replay, LoRA) yet.
- 5–10 seeds on noisy PPL metrics; several effects need more power.
- Sleep consolidation writes very little to slow memory in the current
  implementation (`gain_B_slow ≈ 0`); fast/slow separation is achieved mostly
  by isolation, not yet by strong consolidation.

## 7 Conclusion

We present an honest mechanistic account of a developmental learning
architecture. The positive contributions are (a) fast/slow memory separation,
(b) difficulty robustness, and (c) evidence that learning history shapes the
future learner and that this shaping is carried in part by fast-weight state.
The negative contributions are equally important: the scalar learning-rule
parameters we tested are not a stable carrier of development, and a clean
causal proof that fast weights change future learning dynamics still requires
more power and better baselines.

## Reproducibility

- Code: `https://github.com/JayCRL/DLA`
- Result JSONs and logs on the training server:
  `~/llm-lab/dla-v0.2/results/` and `~/llm-lab/dla-v0.2/logs/`
- Key scripts:
  - `experiments/stage4_formal.py`
  - `experiments/stage5_progressive_curriculum.py`
  - `experiments/stage5_analysis.py`
  - `experiments/stage55_learning_rule_development.py`
  - `experiments/stage55b_development_2x2.py`
  - `experiments/stage55c_causality.py`
  - `experiments/stage55d_body_decomposition.py`
  - `experiments/stage55e_wfast_p0.py`
- Full per-stage tables: `docs/experiments.md`

## References (to be completed)

- Abraham & Robins, "Memory retention: the synaptic stability versus plasticity dilemma", TINS 2005.
- Zenke, Poole, Ganguli, "Continual learning through synaptic intelligence", ICML 2017.
- Ba, Hinton, Mnih, Leibo, Ionescu, "Using fast weights to attend to the recent past", NeurIPS 2016.
- Beaulieu, Frantar, Mironov, Chen, Savarese, "Learning to continually learn", ECAI 2020.
- Miconi et al., "Differentiable plasticity", ICLR 2019.
- Hinton & Plaut, "Using fast weights to deblur old memories", CogSci 1987.
- Andrychowicz et al., "Learning to learn by gradient descent by gradient descent", NeurIPS 2016.
