# The mechanism chain, formalised

> **gradient leaky integration → W_fast spatial structure → selective write-back → future adaptation**

This note writes the chain as equations that match the code (`dla/transformer_dla.py`),
tags each arrow with its evidence level, and lists what would falsify it. All numbers
quoted are from the project's committed runs (n=12 Chinese char-GPT unless noted;
second backbone n=9).

---

## 0. Notation

| symbol | meaning |
|---|---|
| $W_{\mathrm{slow}} \in \mathbb{R}^d$ | slow weights (one wrapped matrix; $d$ coordinates; we write per-matrix and concatenate over the 26 wrapped matrices) |
| $W_{\mathrm{fast}} \in \mathbb{R}^d$ | fast trace (`store[key]["w_fast"]`) |
| $P \in \mathbb{R}^d$ | per-parameter plasticity (`store[key]["p"]`), gate $\varphi(P) = \mathrm{softplus}(P) > 0$ |
| $Q \in \mathbb{R}^d$ | success-gated eligibility trace |
| $m, v \in \mathbb{R}^d$ | Adam moments of the fast update |
| $g_t = \nabla_{W_{\mathrm{eff}}} L_t$ | teaching gradient of the current batch |
| $\lambda, \eta, \gamma, \beta, \delta, \rho$ | fast decay, fast step, direct-write coeff, Q-write coeff, sleep decay of $W_{\mathrm{fast}}$, sleep decay of $Q$ |
| $W_{\mathrm{eff}}$ | effective weight actually used by the forward pass: $W_{\mathrm{eff}} = W_{\mathrm{slow}} + \varphi(P) \odot W_{\mathrm{fast}}$ |
| $G(\Phi)$ | adaptation functional of a learner state $\Phi = (W_{\mathrm{slow}}, W_{\mathrm{fast}}, P, Q, m, v)$: $G = \text{gain@40} = (\mathrm{PPL}_{\mathrm{pre}} - \mathrm{PPL}_{40})/\mathrm{PPL}_{\mathrm{pre}}$ on the unseen domain $D$ |

Code constants: $\lambda = 0.02$, $\eta = 6\times10^{-4}$, $\beta \approx 1.0$, $\gamma \approx 0.15$, $\delta = 0.5$, $\rho = 0.7$,
Adam $\beta_1=0.9,\ \beta_2=0.95$, $\epsilon=10^{-8}$, $\alpha_q = 0.3$. Probes run with these defaults
(`load_body` rebuilds `tempos`), so all numbers below use them.

---

## 1. Link 1 — gradient leaky integration (exact identity, code-level)

**Wake step** (per batch, per coordinate $i$):

$$
\begin{aligned}
g_t &= \nabla_{W_{\mathrm{eff}}} L_t\\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t\\
\hat{a}_t &= \hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)\qquad(\hat{m},\hat{v}\ \text{bias-corrected})\\
W_{\mathrm{fast},t+1} &= (1-\lambda) W_{\mathrm{fast},t} - \eta\,\varphi(P_t)\odot \hat{a}_t
\end{aligned}
$$

**Closed form.** Unrolling over a lifetime of $T$ steps with start `0`:

$$
\begin{aligned}
W_{\mathrm{fast},T} &= -\eta \sum_{k=0}^{T-1}(1-\lambda)^{T-1-k}\,\varphi(P_k)\odot \hat{a}_k\\
&= -\eta \sum_{k} \kappa(T,k)\,\hat{a}^{\varphi}_k,\qquad \kappa(T,k) = (1-\lambda)^{T-1-k}
\end{aligned}
$$

So $W_{\mathrm{fast}}$ is the **gated-Adam gradient field integrated with an exponential kernel**;
the kernel's time constant is $1/\lambda = 50$ steps ≈ one 40-step curriculum stage
(hence "one-stage memory"). $\hat{a}$ is a *coordinate-normalised* EMA of $g$, so the integral
keeps the **sign pattern and relative-magnitude structure** of recent gradients while
being invariant to per-coordinate scale.

**Status:** exact (a re-derivation of the code, verified by deterministic replays of
the stored D-probes: 48/48 runs, mean max |Δppl| ≈ 0.24 vs archive).

---

## 2. Link 2 — W_fast spatial structure (measured, partly null)

**History contrast.** For the same seed, define

$$
\Delta_{\mathrm{hist}} = W_{\mathrm{fast}}^{\mathrm{HE}} - W_{\mathrm{fast}}^{\mathrm{EH}} = -\eta \sum_k \kappa(T,k)\left[\hat{a}^{\varphi}_k(\mathrm{HE}) - \hat{a}^{\varphi}_k(\mathrm{EH})\right]
$$

i.e. the history difference *is* a difference of two exponentially weighted gradient-trajectory
integrals — no explicit alignment term appears anywhere in the code (static audit: no
cosine/dot objective exists in the Transformer-DLA path).

**Measured properties (n=12):**

| quantity | value | reading |
|---|---|---|
| $\|\Delta_{\mathrm{hist}}\|$ | 3.34 ± 0.24 | — |
| same-history cross-seed $\|W_{\mathrm{fast},i} - W_{\mathrm{fast},j}\|$ | 3.21–3.26 | **history contrast ≈ seed noise in norm** |
| $\cos(W_{\mathrm{fast}}^{\mathrm{EH}}, W_{\mathrm{fast}}^{\mathrm{HE}})$ | 0.579 | same-history cross-seed cos 0.60–0.61 → **not separable globally** |
| PCA of {Δ_hist} | PC1 ≈ 20% var, top-3 ≈ 41% | no dominant shared mode |
| module energy share of $\Delta_{\mathrm{hist}}$ | emb 49% / attn 15% / mlp 36% | norm-heavy ≠ causal (controls: MLP+attn destructive) |

So "structure" must be read as **fine-grained, per-seed, allocation-level**, not as a
global direction or a large-norm difference.

**Directional character (correlational).** With $g_0$ the first D-probe gradient
(Adam-shaped, gated):

```
ρ_seed = cos(g_0, Δ_hist)
r(ρ_seed, G_he) = +0.87  (perm p ≈ 0.000; LOO [0.81, 0.91]; FDR q ≈ 0.003)
partial r after controlling ‖Δ‖, rel-norm, cos(EH,HE), ‖g_0‖ = +0.77
module split: MLP 0.87, attention 0.83, embedding 0.68
```

$|\rho|$ itself is tiny (≈0.02, vs shuffle-null 0.0003, cross-seed-null 0.004): the
*ordering* of seeds by $\rho$ predicts adaptation, but the alignment is not a large
geometric overlap. **Status: correlational only — no direction-manipulation experiment.**

---

## 3. Link 3 — selective write-back (causal)

**Sleep operator.** At each task boundary:

$$
W_{\mathrm{slow}} \leftarrow W_{\mathrm{slow}} + \beta Q + \gamma W_{\mathrm{fast}}\qquad\text{(Q-pathway} + \text{DIRECT pathway)}
$$

$$
W_{\mathrm{fast}} \leftarrow \delta\,W_{\mathrm{fast}};\qquad Q \leftarrow \rho\,Q;\qquad m,v \leftarrow 0
$$

Define the **allocation field** of a write: $A = \beta Q + \gamma W_{\mathrm{fast}}$. For the direct term,
coordinate-wise $A_i = \gamma \cdot W_{\mathrm{fast},i}$ with a *single scalar* $\gamma$: the rule is
unselective, so **allocation is entirely inherited from $W_{\mathrm{fast}}$**.

Two facts make Q inert in this setting: `success` is a global scalar ($q_{\mathrm{inc}} = dw\cdot\mathrm{success}$
applies the same coefficient to every coordinate), and the numbers are tiny —
$\|Q\| \approx 0.002$ vs $\|W_{\mathrm{fast}}\| \approx 3.6$; estimated Q-write ≈ 0.4% of the direct write per sleep.
Predictably, the Q-only individual behaves like no-consolidation (`qonly ≈ nocons`).

**Interventions on the state transition** ($\Phi \mapsto I\cdot\Phi$, all else fixed):

| intervention | operator | estimand τ = E[G(I·Φ)] − E[G(Φ_ref)] | measured |
|---|---|---|---|
| carrier swap | $W_{\mathrm{fast}} \leftarrow W_{\mathrm{fast}}^{\mathrm{EH}}$ in the HE body | τ_swap (HE ref) | −0.019, CI [−0.033,−0.005], d≈−0.7, 10/12 (n=12); second backbone −0.067, t=−7.13 (n=9) |
| no write | $A = 0$ | τ_nocons (direct ref) | −0.0113, t=−7.36 |
| Q only | $A = \beta Q$ | τ_qonly | ≈ τ_nocons (Q inert) |
| **energy-matched shuffle** | $A' = \Pi_{\pi} A$, so $\|A'\| = \|A\|$ exactly, pairing destroyed | τ_shuf (direct ref) | **−0.0115, t=−5.45** |
| uniform placement | $A'_i = \gamma\|W_{\mathrm{fast}}\|/\sqrt{d}$ | τ_unif | −0.0105, t=−5.16 |
| top-20% concentration | energy on the largest $|W_{\mathrm{fast},i}|$ | τ_top | −0.0089, t=−4.18 |
| boundary off (`nosleep`) | skip sleep entirely (no write, no decay, no reset) | τ_nosleep (direct ref) | **+0.0163, t=+6.06** on adaptation; retention worse (see §4) |

**Selectivity criterion.** A write rule is *allocation-selective* iff destroying the
coordinate pairing at fixed energy changes the outcome:

$$
\tau_{\mathrm{shuf}} < 0 \quad\text{with}\quad \|A'\| = \|A\|\quad\Rightarrow\quad \text{the coordinate placement of the write is functionally necessary}
$$

This holds (t=−5.45, n=12) and is *not* explained by energy concentration, since
uniform placement and top-20% concentration are both as harmful as shuffling. What
matters is the **coordinate-matched (magnitude *and* sign) write** $A_i = \gamma W_{\mathrm{fast},i}$.

**Why it works — first-order account.** For a small write $A$ onto $W_{\mathrm{slow}}$, the change
of the future-task loss at the start of adaptation is

$$
\Delta L_D \approx \langle \nabla_{W_{\mathrm{slow}}} L_D,\, A\rangle + O(\|A\|^2)
$$

so a write helps exactly to the extent that it is **aligned with the future-loss gradient
field in coordinate space**. Content-matched $A \propto W_{\mathrm{fast}}$ inherits partial alignment
through Link 1 ($W_{\mathrm{fast}}$ integrates the same gradient field, with gate $\varphi(P)$); an
energy-matched shuffle $\Pi A$ randomises that pairing, and $\mathbb{E}\langle \nabla L_D, \Pi A\rangle \approx 0$ unless the
field has a large constant component. This is the mathematical content of the shuffle
control, and it also explains the modest absolute size of the effect ($|\rho| \approx 0.02$
⇒ first-order term is small).

---

## 4. Link 4 — future adaptation and retention (causal, two-sided)

Same-protocol outcomes (n=12, HE arm; $G$ on unseen D and retention as end-of-history
ppl on the seen curriculum domains relative to birth):

| learner | forward $G$ (gain@40) | retention (rel. forgetting, negative = kept) |
|---|---|---|
| direct | +0.0140 ± 0.0148 | **−0.0192 ± 0.0083** |
| nocons | +0.0027 ± 0.0103 | −0.0152 ± 0.0111 |
| nosleep | **+0.0303 ± 0.0215** | −0.0049 ± 0.0079 (worst, t=+10.3 vs direct) |

Ten-task cross-domain sequence (n=8, relative forgetting of each task measured right
after learning it vs after all ten tasks): mean forgetting is ≈0 for every variant
(direct −0.96%, nocons −0.88%, nosleep +0.79%); the only robust separation is
**direct vs nosleep** (per-task Δ −0.6…−6.0%, t≈−3…−14), i.e. the **boundary**
(decay + moment reset), not the write, is what protects old tasks; the direct write's
measurable contribution is *forward* adaptation.

So the chain closes as two coupled halves:

```
boundary (δ, reset)          → retention of old tasks
content-matched write (γ W_fast) → forward adaptation (allocation-selective)
```

---

## 5. Evidence-level summary of the chain

| link | statement | type | status |
|---|---|---|---|
| 1 | $W_{\mathrm{fast}}$ = gated-Adam gradient field with exponential kernel (τ≈50) | identity (code) | exact |
| 2a | $\Delta_{\mathrm{hist}}$ is a difference of two such integrals; no explicit alignment objective exists | static audit | established |
| 2b | global geometry of $\Delta_{\mathrm{hist}}$ is not separable from seed noise | measurement + null | established (negative) |
| 2c | `cos(g_0, Δ_hist)` orders seeds by adaptation (r=0.87, FDR q=0.003) | correlational | supported, **not causal** |
| 3 | direct write carries the effect; Q/success-gating inert (global scalar, ‖Q‖≈0.002) | causal (ablation, n=12) | established |
| 3' | **allocation necessity**: energy-matched shuffle removes the effect | causal (n=12, t=−5.45) | **established (core)** |
| 4a | forward adaptation: direct ≈ full, nosleep highest (boundary costs forward) | causal (n=12) | established |
| 4b | retention: boundary protects old tasks; direct write adds little *relative* retention | causal (n=12) + 10-task (n=8) | established |

## 6. Falsifiers (what would break the chain)

1. An explicit alignment objective found in the codebase (breaks "emergent" in 2a).
2. Energy-matched shuffle losing significance at larger n (breaks 3').
3. $\tau_{\mathrm{shuf}} \approx \tau_{\mathrm{unif}} \approx \tau_{\mathrm{top}} \approx 0$ (would show allocation is irrelevant).
4. Direction manipulation (e.g. α·Δ̂ interpolation) not changing adaptation (would confine 2c to correlation).
5. Retention surviving `nosleep` (would move the boundary claim to the write).
