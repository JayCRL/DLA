# DLA Static Mechanism Audit (code-only) — Directional Alignment & Selective Consolidation

Date: 2026-09-09 · Method: pure static audit of current repo code (no runs).
Audited files:
- `dla/transformer_dla.py` — the Transformer-DLA used by the paper (stages 4–8).
- `experiments/stage55_learning_rule_development.py`, `experiments/stage55c_causality.py` — history runner & probe.

> Scope note: `dla/model.py:94-100` & `dla/config.py:77` contain per-connection
> `alignment` outputs of an F_phi rule network. That belongs to the **old MLP
> rule-net lineage (stages 1–3)**, NOT the Transformer DLA in the current paper,
> and is never executed by `stage4+`/`transformer_dla.py`. It is therefore out of
> scope for "the current mechanism" and must not be cited as evidence of an
> explicit alignment mechanism in DLA.

---

## A. Directional Alignment

### A.1 Every direction/history-related update in the code (wake step, `DLA_GPT.dla_step`, `transformer_dla.py:256-324`)
1. **Teaching gradient**: `loss.backward()` gives `g = dL/dW_eff` on every wrapped weight (`dla_step` L259-262, used via `mod.weight.grad`, L284).
2. **Adam moments**: `m ← b1·m+(1-b1)g`; `v ← b2·v+(1-b2)g⊙g`; `m_hat=m/(1-b1^t)`, `v_hat=v/(1-b2^t)`; `adam = m_hat/(√v_hat+1e-8)` (L290-294). `m_hat` is an EMA of recent gradients → a **short-term gradient-direction memory**; the `√v_hat` division is per-coordinate.
3. **W_fast recurrence**: `dw = -eta_fast · softplus(p) ⊙ adam - fast_decay · w_fast` (L296); `w_fast += dw` (L297). With `fast_decay=0.02` (cfg L55) the time constant is ≈1/0.02=50 steps ≈ one 40-step curriculum stage: **W_fast is a leaky integrator of the gated-Adam update direction over ~one stage**.
4. **P gate**: `gate=softplus(p)` positive per-coordinate scale (L287, applied in L296); `p` updated as `dp = eta_plast·progress·relevance - stability·(p-p0)`, `relevance=|g|/(rms(g))` (L302-304). `relevance` is magnitude-only; `progress` is a scalar. → P only rescales coordinates positively (no sign flips); it never carries a direction objective.
5. **Q (consolidation trace)**: `q_inc = dw·success`; `Q ← (1-α_q)Q + α_q·q_inc` (L307-308) — only feeds sleep (see B).
6. **No alignment operator exists**: no cosine / normalized-dot / projection term with any history reference anywhere in the wake path, sleep, or the runners (`grep align|cosine|dot` matches only comments & the MLP lineage above).

### A.2 Meta surrogate (note)
`meta_unroll_loss` (L344-402) computes φ-gradients on a **plain-gradient** clone of the fast dynamics (`dw = -eta_fast·gate·g - fast_decay·w_fast`, L387) — i.e., **without Adam**. φ is meta-tuned on dynamics that differ from the real wake step. Not an alignment mechanism either, but a real code-level inconsistency worth noting for φ results.

### A.3 Where any observed "directional signal" must come from (code-deduced)
1. Adam `m_hat` (EMA of recent g) — the only explicit "direction memory";
2. leaky recurrence of `W_fast` over the stage (≈50-step horizon, halved-not-reset at sleep, L336);
3. curriculum content/recency (last stage's domain) shaping `g` history;
4. positive coordinate rescaling by `softplus(p)`.
There is **no term that compares the future gradient to a history-induced direction**, and none that optimises such alignment.

**Verdict A**: explicit alignment operator/loss → **not implemented**. Directional content is an **emergent property** of (Adam + W_fast leaky recurrence + stage recency). Whether the emergent alignment observed in data (cos0→gain, r≈0.87) is mechanistic or confound is a data question, not a code question — but the code cannot implement "directional alignment as a mechanism", because no such operation exists.

---

## B. Selective Consolidation

### B.1 Exact trace success → Q → W_slow (with coefficients)
Wake (per step, all parameters identically):
```
success  = clamp( (loss_ema − loss) / (loss_ema + 1e-4), 0, 1 )      L280   ← GLOBAL SCALAR
q_inc    = dw · success                                              L307
Q        ← (1−α_q)·Q + α_q·q_inc      α_q = softplus(log_α_q) ≈ 0.30 cfg L56 / L308
```
Sleep (task boundary, `dla_sleep` L328-341):
```
W_slow   += β·Q + γ·W_fast                                            L335
            β = sigmoid(logit(consolidate_beta))  ≈ 0.9999 (cfg 1.0,  L57)
            γ = sigmoid(logit(consolidate_fast_direct)) ≈ 0.15       (L58)
W_fast   *= δ = consolidate_fast_decay ≈ 0.5                         L336
Q        *= ε = consolidate_q_decay ≈ 0.7                            L337
reset m,v                                                            L340
```
Effective coefficients come from `tempos.values()` (sigmoid/softplus spaces, L111-114). Note: probes **never restore the saved φ** — `load_body` constructs a fresh `TransformerDLAConfig(eta_fast=...)` and new `tempos` (`stage55c_causality.py:105-115`) — so every D-probe runs with **default** consolidation coefficients.

**Direct bypass exists**: the `γ·W_fast` term at L335 writes W_fast (unselected) into W_slow **in the same instruction** as `β·Q`.

### B.2 Selection level — precise
- Selection coefficient = `success`, one scalar from EMA losses (L280) → identical coefficient for **every parameter and every coordinate**.
- Per-parameter Q differences therefore only reflect per-parameter `dw` magnitudes, not any per-parameter decision; no subspace structure is ever computed.
- `relevance` (the only per-parameter signal) enters **P**, not Q (L302-304 vs L307).
- The meta-surrogate clones the same scalar-gated Q (L376, L392-396).

**Verdict B/C**: implemented = **step-level / global gating only**. Parameter-level and subspace-level selection = **not implemented**.

### B.3 Why Q cannot form a history signal (code-derived, matches measured ‖Q‖≈0.002)
1. **Short EMA window**: α_q≈0.3 → Q is dominated by the last few steps.
2. **Decayed at every sleep**: Q ← 0.7·Q (L337); after 2–3 sleeps ≈ 0.34–0.49× and then immediately overwritten by the next stage's recent steps → Q never integrates across the lifetime.
3. **Scalar success**: Q direction ≈ direction of local recent `dw`; history-specific content can only leak in through state-dependence of `dw`, which the short window + decay mostly discard.
4. **Q is inert during the future-task probe**: `run_transfer` (stage55 L264-284) calls only `dla_step`, never `dla_sleep` → on the unseen-domain adaptation measurement, **Q never acts**. It can influence that measurement only indirectly, through W_slow writes made earlier (β·Q≈0 numerically).

### B.4 Why W_fast is the dominant history carrier (code-derived)
- W_fast integrates the full last stage of gated-Adam directions (τ≈50 steps) and is only halved, never reset, at sleep (L297, L336) → it retains the freshest, highest-energy history trace.
- Per sleep, W_slow gains only γ·W_fast (≈0.15×) + β·Q (≈0); over 2–3 sleeps W_slow is a damped, mixed accumulator (measured total sleep write ≈0.45% of ‖W_slow‖ per boundary, Q contribution ≈0.4% of the write).
- On the D-probe (no sleep) the directly-probed history state with the largest, freshest directional content is W_fast; P is a magnitude field, W_slow is heavily damped.

**Verdict B**: a true parameter/subspace **selective consolidation is not implemented**; the implemented "selection" is a **global step-level scalar**, its numeric channel (Q) is ~3 orders of magnitude below the unselected direct bypass, and it is never invoked on the measured future-adaptation probe.

---

## C. Classification summary (实现 / 太弱 / 未实现)

| 项 | 代码 | 分类 |
|---|---|---|
| success-gated Q formation (step-level) | transformer_dla.py L280, L307-308 | ✅ 已实现（但=global scalar） |
| β·Q + γ·W_fast sleep write + decays + moment reset | L335-340 | ✅ 已实现（含 **direct bypass γ·W_fast**） |
| Adam + leaky W_fast recurrence + P gating | L290-304 | ✅ 已实现（涌现方向性的来源） |
| Q 通道数值强度 / 跨 lifetime 积分 | α=0.3 EMA + ×0.7/sleep → ‖Q‖≈0.002 | ⚠️ 存在但太弱（~0.4% of direct write） |
| "选择性"粒度 | success 单一标量 | ⚠️ 仅 step/global 级；parameter/subspace 级未实现 |
| Explicit directional-alignment operator/loss | 无 | ❌ 根本没有实现 |
| Parameter/subspace-level consolidation selection | 无 | ❌ 根本没有实现 |
| 睡眠/巩固影响 D-probe 未来适应 | run_transfer 从不调用 dla_sleep | ❌ 根本不在测量路径上（仅经 W_slow 间接） |

---

## D. Failure-mode localization & which B3 runs are needed

**Static analysis fully explains**: (1) why Q has no history signal (window/decay/scale), (2) why W_fast is the history carrier (leaky integrator, halved-not-reset, directly probed), (3) why "selectivity" is only a step-level scalar. Measured magnitudes already available from existing bodies (`audit_cheap.json` B2) → **B2 的逐 body 分解可以省掉**。

**Static cannot decide** one empirical question: whether the sleep-write pathway matters at all for the EH/HE future-adaptation effect (i.e. does the effect survive with consolidation disabled?). This depends on dynamics, not code → needs the ablation.

**Prediction from the audit** (so qonly can be deprioritised):
- `qonly` ≈ `nocons` (Q write ≈ 0),
- `nocons` ≈ `direct` only if γ·W_fast writes are also non-causal; if `nocons ≠ full`, `direct` isolates the (dominant) bypass.

**Minimal experiment set (if still wanted)**:
1. `nocons` on seeds {0,1,2,3} × {EH,HE} (8 jobs) — does the history effect need ANY consolidation?
2. `direct` on seeds {0,1,2,3} × {EH,HE} (8 jobs) — isolates the direct bypass.
3. `qonly`: run only if `direct` and `nocons` differ, on seeds {0,1} as tiebreak.
Baseline `full` = existing archived bodies/probes (no new compute).
Estimated cost on the Mac (parity-validated harness, ~2.5 min/job): 16 jobs ≈ 25–35 min at 4 workers.

**Not needed by this audit**: P3 interpolation; 72-job grid; qonly-at-scale; retention-side (old-task forgetting) probes (that is a separate Stage-4-style question about sleep consolidation's effect on memory, not on future adaptation).
