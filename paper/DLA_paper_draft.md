# Developmental Learning Architecture: Learning History, Fast-Weight Traces, and Future Adaptation

**First author: Yang Liu**

**Draft v0.2 (validation-stage) — workshop/arXiv**

---

## Abstract

A conventional neural learner is treated as a fixed function trained once by an external optimizer. This paper studies a more developmental question: does a learner's prior learning history change how it will learn in the future, and which internal state carries this effect? We propose a Developmental Learning Architecture (DLA) that keeps standard MLP/Transformer backbones unchanged but augments each weight with per-parameter fast weights, plasticity, and sleep consolidation. In a 6.59M Chinese character GPT we find that different curriculum histories lead to different future adaptation on an unseen domain, and that this effect is causally tied to the fast-weight state `W_fast`: replacing a hard→easy learner's fast weights with an easy→hard learner's significantly degrades future adaptation (n=12, gain@40 difference −0.019, bootstrap CI excludes zero, Cohen's d≈−0.7). Norm-matching and shuffling do not remove the effect, so it is not explained by weight magnitude or random parameter noise. The effect replicates on a second Shakespeare character-level backbone. We do not find evidence that more experience monotonically accelerates learning: multiple controlled longitudinal studies return null results. The conclusion is that learning history leaves a persistent, partially localizable trace in transient learner state, with `W_fast` providing a causal component of history-dependent future adaptation.

---

## 1. Introduction

### 1.1 What problem exists

Modern neural networks are trained and then frozen. When they learn a second task, they either overwrite the first (catastrophic forgetting) or treat the two tasks as independent (Kirkpatrick et al., 2017; Zenke et al., 2017; De Lange et al., 2021). More fundamentally, standard training does not allow the model itself to become a different kind of learner: the learning algorithm is fixed outside the model, whether it is an external optimizer or a meta-learned rule (Andrychowicz et al., 2016; Miconi et al., 2019; Beaulieu et al., 2020).

We ask a problem that sits before "continual learning" and "meta-learning" — yet is closely tied to a long-standing empirical question about task and curriculum ordering (Poirier & Silver, 2005; Bell & Lawrence, 2022; Li & Hiratani, 2025):

> Can what a model has experienced change how it will learn in the future, and which part of the model carries that developmental change?

### 1.2 Why it matters

If learning history only changes stored knowledge, the learner remains a static function with a growing database. If learning history also changes the learner state, then a small model could, in principle, become a better learner through experience—without changing its architecture. This developmental framing has a deep biological precedent: complementary learning systems (CLS) theory explains memory/learning trade-offs by the interaction of a fast hippocampal system and a slow neocortical system, consolidated over time (McClelland, McNaughton & O'Reilly, 1995; Kumaran, Hassabis & McClelland, 2016). Recent empirical results further show that plasticity itself is a fragile resource that ordinary training can consume (Dohare et al., 2024; Lyle, Rowland & Dabney, 2022; Nikishin et al., 2022), so "what kind of learner a model is" is not fixed by its architecture alone. This is the key motivation for "growing models" and for understanding whether development is a meaningful object in machine learning.

The practical significance is twofold. First, it tells us which parts of a model to preserve or transfer between learning phases. Second, it gives a falsifiable framework for claims such as "more experience makes models learn faster" — a claim we test and do not find support for.

---

## 2. How existing work addresses the problem

- **Catastrophic forgetting methods** (regularization, replay, EWC, pseudorehearsal) address memory, not the change of the learner itself (Kirkpatrick et al., 2017; Zenke et al., 2017; Robins, 1995; De Lange et al., 2021).
- **Fast-weight models** add a temporary memory but usually keep the learning rule external. The idea goes back to fast-weight memories (Hinton & Plaut, 1987; Ba et al., 2016) and has seen a modern revival: linearised attention is formally equivalent to fast-weight programming (Schlag, Irie & Schmidhuber, 2021), in-context learning can be understood as implicit gradient descent / fast-weight updates inside a Transformer (von Oswald et al., 2023), and recent sequence models introduce trainable "neural memory" modules with fast updates alongside slow base weights (Behrouz, Zhong & Mirrokni, 2025). Fast/slow dual-module designs also appear in RLHF pipelines ("fast-slow chasing" in online DPO; Qi et al., 2024). In most of these, the fast store is either a memory mechanism or an engineering device; DLA instead treats fast/slow as a *developmental state* whose history we causally dissect.
- **Complementary learning systems and consolidation.** CLS theory attributes the complementary benefits of fast, instance-based and slow, generalising stores to their interaction and to offline consolidation (McClelland et al., 1995; Kumaran et al., 2016). Machine-learning instantiations usually realise this as experience replay; DLA instantiates the fast/slow split as an internal, per-parameter weight state with sleep-boundary consolidation and no external replay buffer.
- **Meta-learning** optimizes a learning rule over many episodes; it asks whether a rule can be learned, but rarely whether one individual's rule develops within a single lifetime (Andrychowicz et al., 2016; Beaulieu et al., 2020).
- **Learned plasticity** (differentiable plasticity, ANML) shows that plasticity can be meta-learned, but does not separate *body state* from *learning-rule parameters* as causal carriers (Miconi et al., 2019; Beaulieu et al., 2020). Complementary findings that plasticity degrades over long training—capacity loss and primacy bias—motivate making plasticity itself a state to maintain (Dohare et al., 2024; Lyle et al., 2022; Nikishin et al., 2022).
- **Task and curriculum ordering.** The order in which tasks are learned measurably changes continual-learning outcomes and the amount of forgetting (Bell & Lawrence, 2022; Li & Hiratani, 2025), affects the consolidation of task knowledge (Poirier & Silver, 2005), and is the core object of curriculum learning (Wang, Chen & Zhu, 2022). This literature mostly measures order effects on *stored knowledge* about seen tasks. We ask a complementary question: does order change the *learner state* itself, such that future adaptation to a *never-seen* task changes—and which state component carries that effect?
- **We therefore design DLA** to ask a mechanistic question that is usually skipped: given the same task sequence, which state component carries history-dependent future adaptation?

---

## 3. Our solution: DLA

### 3.1 Design principle

We do not modify the Transformer skeleton. Instead, each weight matrix is accompanied by:

- `W_slow`: long-term consolidated knowledge
- `W_fast`: a fast learning trace
- `P`: per-parameter plasticity
- `Q`: sleep eligibility trace

Effective weights:

```
W_eff = W_slow + softplus(P) * W_fast
```

Structurally, this additive form resembles parameter-efficient transfer methods that freeze a base network and attach small trainable additive weights—adapters (Houlsby et al., 2019), LoRA (Hu et al., 2021), and the unified additive-PEFT view (He et al., 2022). DLA differs in three ways that matter for development: (i) the additive pathway is a *within-lifetime developmental state* updated online by the learner's own wake/sleep dynamics, not a separately fine-tuned module; (ii) it is shared across tasks and persists (subject to decay and consolidation) rather than being re-initialised per task; and (iii) its contribution is gated per parameter by a learned plasticity `softplus(P)`. The fast/slow decomposition itself follows complementary learning systems theory (McClelland et al., 1995; Kumaran et al., 2016).

### 3.2 Wake and sleep

During learning, `W_fast` is updated with Adam-style moments, gated by `softplus(P)`. At task boundaries (sleep), part of `W_fast` is consolidated into `W_slow` through `Q`, and `W_fast` is decayed. This is the fast/slow separation: it follows the CLS prescription that fast, recent traces be gradually transferred into a slow, generalising store (McClelland et al., 1995; Kumaran et al., 2016), but is implemented as internal weight state with no external replay buffer. Fast/slow weight co-design has engineering precedent in optimisation and RLHF (Qi et al., 2024) and in neural-memory architectures (Behrouz et al., 2025); here it is studied as a developmental mechanism whose causal carrier we test.
**Algorithm 1: DLA wake–sleep cycle**

```
1: for task t in curriculum do
2:   for step in 1..T do
3:     compute g = dL/dW_eff
4:     update W_fast with Adam(g), gated by softplus(P)
5:     update P from progress/relevance
6:   end for
7:   consolidate: W_slow += beta*Q + gamma*W_fast
8:   decay: W_fast *= c, Q *= d
9: end for
```



### 3.3 Experimental design to isolate development

To test whether history changes the future learner, we:

1. Build two histories with the same three domains but different order:
   - easy→hard (EH)
   - hard→easy (HE)
2. Probe each individual on a never-seen domain `D`.
3. Perform body/component cross-injection to see which state component transfers the history effect.
4. Run norm-matched, shuffled, and module-localized controls to rule out magnitude/randomness.
5. Compare with standard baselines (AdamW, replay, EWC).

---

## 4. Experimental setup

Primary backbone: 6.59M Chinese character GPT (6 layers, 8 heads, 256 dim, vocab 7280), pretrained on Chinese Wikipedia. Curriculum domains: Wikipedia, SFT-style QA, science Wikipedia; unseen domain D is a science slice. Second setting: ~10.65M Shakespeare character GPT (vocab 65). Metrics: gain@40, LE_D, T80, paired bootstrap CI, sign test.

**Table 1: Common experimental hyperparameters.**

| Hyperparameter | Value |
|---|---|
| Block size | 128 |
| Batch size | 32 |
| Steps per task | 40 |
| AdamW learning rate (baselines/DLA base) | 1e−4 |
| EWC lambda | 1e3 |
| Replay buffer capacity | 48 batches |
| Seeds (primary setting) | 12 |
| Seeds (baselines) | 5 |
| Seeds (second setting) | 2 |

---

## 5. Results: how well does it work

### 5.1 Fast/slow separation protects consolidated memory

DLA matches tuned AdamW on new-domain adaptation (gain +14.0% vs +13.0%) while slow-memory forgetting is ≈0 (−0.4%). This shows the mechanism does not sacrifice memory for plasticity.
![Figure 1a: Fast/slow separation and retention across domains.](figures/A_across_lifetime.png)

![Figure 1b: Adaptation curve on the new domain.](figures/B_adaptation_curve.png)

![Figure 1c: Relearning curve.](figures/A_relearning_curve.png)



### 5.2 Learning history changes future adaptation

Across histories, HE learners adapt better to unseen D than EH learners. That later adaptation depends on the *order* of earlier tasks mirrors continual-learning and curriculum results measured on seen tasks (Bell & Lawrence, 2022; Li & Hiratani, 2025; Poirier & Silver, 2005). The distinctive feature here is that D was never seen in either history, so the order effect must act through the learner's internal state rather than through stored content about D. This is not unique to DLA; but DLA is most robust after a difficult history.

![Figure 2: History effect on unseen D (2x2 development experiment).](figures/2x2_D.png)

### 5.3 The effect is not carried by the learning-rule parameters φ

Body×φ cross-injection shows that swapping φ between histories changes future adaptation much less than swapping body state. φ is not a stable/dominant carrier in this setting.

![Figure 3: Body × φ causal dissection.](figures/cross_2x2.png)

### 5.4 W_fast is a causal component

Core result (n=12):

| condition | mean gain@40 |
|---|---|
| HE/HE | +0.014 |
| HE + EH W_fast | −0.005 |

Paired HE→raw EH: mean −0.019, 95% CI [−0.033, −0.005], Cohen's d ≈ −0.71, sign p=0.019 (10/12 negative). Initial PPL differences are small; the effect appears in the adaptation trajectory.

![Figure 4: P0 future-adaptation trajectories.](figures/p0_trajectory.png)

### 5.5 Controls: magnitude, randomness, module

| control | mean Δ gain@40 | 95% CI | d |
|---|---|---|---|
| raw EH W_fast | −0.019 | [−0.034, −0.005] | −0.71 |
| norm-matched EH W_fast | −0.022 | [−0.034, −0.009] | −0.90 |
| shuffled EH W_fast | −0.017 | [−0.023, −0.009] | −1.25 |
| EH embedding W_fast | −0.006 | [−0.009, −0.002] | −0.87 |
| EH attention W_fast | −0.012 | [−0.018, −0.006] | −1.06 |
| EH MLP W_fast | −0.017 | [−0.025, −0.008] | −1.05 |

Norm-matching does not rescue the effect; shuffling also does not remove it. Module localization is suggestive of MLP and attention, with embedding weaker.

![Figure 5: W_fast cross-injection and full future-adaptation trajectories. A: norm/shuffle controls. B: module-localized transfer. C: paired gain@40 effects. D: example trajectories.](figures/fig7_multi_panel.png)

### 5.6 Fair baselines

| method | EH | HE | HE−EH |
|---|---|---|---|
| AdamW | 0.339 | 0.675 | +0.336 |
| AdamW+replay | 0.375 | 0.961 | +0.586 |
| EWC | 0.292 | 0.895 | +0.603 |
| DLA | 0.517 | 0.709 | +0.192 |

DLA is the most robust after an easy→hard history; standard learners show larger HE performance but larger EH degradation.

![Figure 6: Fair baselines — future adaptation after EH vs HE history.](figures/fig_baselines.png)

### 5.7 Second-setting replication

Shakespeare char GPT, 2 seeds: HE/HE gain@40 +0.052/+0.042; HE+EH fast +0.004/−0.001. The destructive effect replicates in direction.

### 5.8 Negative result: more experience does not necessarily make learning faster

Stage 5 (difficulty-normalized): flat LE. Stage 6 (matched difficulty): p=0.17. Stage 7 (cross-domain, 20 seeds): p=0.82. Stage 8 (physics near-transfer, 20 seeds): d=−0.17. We therefore do not claim "more experience → faster learning". This null is consistent with a growing plasticity literature: more training does not monotonically make a network more learnable, early experience can dominate later learning (primacy bias; Nikishin et al., 2022), and both capacity and plasticity can be consumed by ordinary updates (Lyle et al., 2022; Dohare et al., 2024).

![Figure 7a: Negative result — cross-domain longitudinal (Stage 7, 20 seeds).](figures/norm_slope_trend.png)

![Figure 7b: Negative result — physics near-transfer (Stage 8, 20 seeds).](figures/stage8_trend_20.png)

---

## 6. Limitations

- Second-setting n=2; baselines n=5.
- Single small backbones; no large-scale validation.
- Sleep consolidation writes only a small amount into `W_slow`; fast/slow separation is mostly isolation.
- No pre-registration; we report n=10→n=12 transparently.
- We cannot yet claim W_fast is the only carrier or that module localization is definitive.

---

## 7. Conclusion and significance

The learner is not fully characterized by what it currently knows. Learning history leaves a persistent, partially localizable trace in transient learner state, with `W_fast` providing a causal component of history-dependent future adaptation. This result gives a concrete mechanism for "development" in neural networks and clarifies a frequent but unsupported claim: developmental change does not equal universally faster learning.

---

## Reproducibility

Code: https://github.com/JayCRL/DLA
Experiment details: `docs/experiments.md`, `paper/overnight_report.md`, `paper/validation_audit.md`
Scripts: `stage4_formal.py`, `stage55e_wfast_p0.py`, `validation_p0_controls.py`, `validation_baselines.py`, `validation_second_setting.py`, `stage6_longitudinal.py`, `stage7_cross_domain.py`, `stage8_physics.py`.

---

## References

1. Andrychowicz, M., Denil, M., Gomez, S., Hoffman, M. W., Pfau, D., Schaul, T., Shillingford, B., & de Freitas, N. (2016). Learning to learn by gradient descent by gradient descent. *Advances in Neural Information Processing Systems (NeurIPS)*.
2. Ba, J., Hinton, G. E., Mnih, V., Leibo, J. Z., & Ionescu, C. (2016). Using fast weights to attend to the recent past. *Advances in Neural Information Processing Systems (NeurIPS)*.
3. Beaulieu, S., Frantar, L., Mironov, E., Chen, Y., & Savarese, S. (2020). Learning to continually learn. *European Conference on Artificial Intelligence (ECAI)*.
4. Behrouz, A., Zhong, P., & Mirrokni, V. (2025). Titans: Learning to memorize at test time. *arXiv:2501.00663*.
5. Bell, S. J., & Lawrence, N. D. (2022). The effect of task ordering in continual learning. *arXiv:2205.13323*.
6. De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., Slabaugh, G., & Tuytelaars, T. (2021). A continual learning survey: Defying forgetting in classification tasks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43(10), 3366–3385.
7. Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., & Sutton, R. S. (2024). Loss of plasticity in deep continual learning. *Nature*, 632(8026), 768–774. https://doi.org/10.1038/s41586-024-07711-7
8. He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T., & Neubig, G. (2022). Towards a unified view of parameter-efficient transfer learning. *International Conference on Learning Representations (ICLR)*.
9. Hinton, G. E., & Plaut, D. C. (1987). Using fast weights to deblur old memories. *Proceedings of the Ninth Annual Conference of the Cognitive Science Society*.
10. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. *International Conference on Machine Learning (ICML), PMLR 97*, 2790–2799.
11. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *arXiv:2106.09685* (ICLR 2022).
12. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526.
13. Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences*, 20(7), 512–534. https://doi.org/10.1016/j.tics.2016.05.004
14. Li, Z., & Hiratani, N. (2025). Optimal task order for continual learning of multiple tasks. *International Conference on Machine Learning (ICML)*. arXiv:2502.03350.
15. Lyle, C., Rowland, M., & Dabney, W. (2022). Understanding and preventing capacity loss in reinforcement learning. *International Conference on Machine Learning (ICML)*.
16. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review*, 102(3), 419–457. https://doi.org/10.1037/0033-295X.102.3.419
17. Miconi, T., Clune, J., & Stanley, K. O. (2019). Differentiable plasticity: Training plastic neural networks with backpropagation. *International Conference on Learning Representations (ICLR)*.
18. Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.-L., & Courville, A. (2022). The primacy bias in deep reinforcement learning. *International Conference on Machine Learning (ICML)*.
19. Poirier, R., & Silver, D. L. (2005). Effect of curriculum on the consolidation of neural network task knowledge. *Proceedings of the IEEE International Joint Conference on Neural Networks (IJCNN)*.
20. Qi, B., Li, P., Li, F., Gao, J., Zhang, K., & Zhou, B. (2024). Online DPO: Online direct preference optimization with fast-slow chasing. *arXiv:2406.05534*.
21. Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science*, 7(2), 123–146.
22. Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear transformers are secretly fast weight programmers. *International Conference on Machine Learning (ICML), PMLR 139*, 9355–9366.
23. von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., & Vladymyrov, M. (2023). Transformers learn in-context by gradient descent. *International Conference on Machine Learning (ICML), PMLR 202*, 35151–35174.
24. Wang, X., Chen, Y., & Zhu, W. (2022). A survey on curriculum learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(9), 4555–4576.
25. Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *International Conference on Machine Learning (ICML)*.
