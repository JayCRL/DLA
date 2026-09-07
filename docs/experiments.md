# DLA v0.2 实验结果（linghang1，CPU，2026-09-07）

所有实验代码见 `experiments/`，结果 JSON 与图保存在服务器
`~/llm-lab/dla-v0.2/results/`。除 Stage 4 外均使用自包含合成任务族
（Gaussian / XOR 持续学习流），核心配置 `6→32→2` MLP。

---

## Stage 1：可塑性是否有效（静态 MLP vs 自适应可塑性）

设置：100 个 meta epoch（按“两人生”展开 + 睡眠巩固 + 保持损失），3 seeds，8 个
未见过的评测任务。StaticMLP 的 SGD 学习率在两个基线上分别用 meta-train 流调参。

| 指标 | StaticMLP | StaticMLP+meta init | DLA（自适应可塑性） |
|---|---|---|---|
| post-task acc ↑ | 0.655 ± 0.002 | **0.884 ± 0.006** | 0.827 ± 0.053 |
| steps→85% ↓ | 7.75 | 5.08 | **4.50** |
| forgetting ↓ | 0.181 | 0.179 | **0.099** |
| final slow acc ↑ | 0.497 | 0.728 | 0.662 |

**结论**：DLA 用纯内部规则达到与“meta 初始化 + SGD”接近的峰值，且**遗忘只有其
约一半**；样本效率最好。StaticMLP 的高峰值来自对 W_slow 的直接覆写，代价是
灾难性遗忘。注意 3 seeds 下遗忘方差较大（DLA 0.099±0.106），需要更多 seed 的
正式版本。

## Stage 2：学习规则是否有效（固定 Hebbian vs 学到的 F_φ）

两臂同起点、同 meta 目标；固定 Hebbian 臂只优化 W_slow 初值（规则与 tempos 冻结）。

| 指标 | 固定 Hebbian | 学到的 F_φ |
|---|---|---|
| post-task acc ↑ | 0.729 ± 0.001 | **0.827 ± 0.053** |
| steps→85% ↓ | 5.50 | **4.50** |
| forgetting ↓ | **-0.002** | 0.099 |
| final slow acc ↑ | **0.731** | 0.662 |

**结论**：学到的规则显著提升学习速度与峰值性能；固定 Hebbian 几乎不改变慢记忆
（所以也不遗忘）。这直接支持“F_φ 是研究对象、规则应该被学出来”的论点。

## Stage 3：学习规则本身能否在生命周期中发育

同一个 meta 训练的 φ，frozen 臂终身固定；adaptive 臂用个体自己的经历回放
`H_t` 周期性做 `φ ← φ + G(H_t)`。

| 指标 | frozen φ | adaptive φ |
|---|---|---|
| avg post acc ↑ | 0.827 | **0.857** |
| per-seed post acc | 0.884 / 0.841 / 0.756 | 0.862 / 0.853 / 0.856 |
| per-seed forgetting ↓ | 0.021 / 0.026 / 0.248 | **0.023 / 0.028 / 0.003** |
| Λ_t 曲线斜率 | +0.014 / 任务 | +0.014 / 任务 |

**结论（诚实版）**：adaptive φ 在**鲁棒性/稳定性**上有正信号（尤其第三个 seed
的遗忘从 0.248 降到 0.003），但“Λ_t 随人生上升”的假设**没有被支持**——当前
任务流难度交替（Gaussian/XOR）把样本效率趋势掩盖了。下一版实验设计应使用
**难度递进的课程**，或固定难度下测 within-family 迁移，才能直接检验 Λ_t。

## DNA 实验：同一 F_φ、四种 DNA、同一段人生

| DNA | post acc ↑ | slow acc ↑ | forgetting ↓ | steps→85% ↓ |
|---|---|---|---|---|
| A 高可塑 | **0.638** | 0.494 | 0.0040 | 9.88 |
| B 高稳定 | 0.514 | **0.512** | **0.0024** | 10.00 |
| C 高新奇 | 0.580 | 0.504 | 0.0040 | 10.00 |
| D 保守 | 0.532 | 0.496 | 0.0026 | **9.54** |

**结论**：同样的学习规则 + 不同的 DNA 先验，在同一段人生中产生可分辨的发育
轨迹：A 学新最快、B 记忆最稳、D 样本效率最好。这正是
`DNA + Experience → 不同发展轨迹` 的最小实现。

## Stage 4 pilot：DLA 机制迁移到真实 Transformer

- 骨架：nanoGPT 6.59M（6 层 / 8 头 / 256 维 / 词表 7280 / block 256），
  `~/llm-lab/nanoGPT/out-chinese/ckpt.pt`，**未改动骨架**
- 一生：维基(A, 40M 字符处取 25 万) → SFT 问答(B, 25 万) → 维基重学(A)
- 预算：每阶段 100 步，batch 32 × block 128（每阶段约 41 万 token）
- 两臂：AdamW 6e-4 直接微调 W_slow vs DLA（per-parameter P 门控快权重 +
  Adam 力矩 + 睡眠巩固；eta_fast=1.2e-3，出生时有效步长≈6e-4）

| 指标（相对个体自身起点） | AdamW 微调 | DLA-Transformer |
|---|---|---|
| B 域适应 gain（SFT PPL 相对下降） | **-12.1%**（变差） | **+13.5%**（27.47→23.76） |
| A 遗忘（学完 B 后维基 PPL 相对上升） | +12.3% | 快记忆 +7.9%，**慢记忆 ≈ 0** |
| A 重学恢复 | 几乎不恢复（42.2） | 恢复 75% 的损失（30.3） |
| A 慢记忆 PPL（终） | 42.2 | **29.075（与出生时一致）** |

**结论（pilot 级）**：同样的适应预算下，普通微调在这台小模型上直接学崩并灾难性
遗忘；DLA 的 fast/slow 分离让个体**学到 B 的同时，慢记忆层对 A 零遗忘**。
本 pilot 的局限：基线学习率未做完整 sweep（6e-4 可能偏高）；Q 资格迹在本次
运行中接近 0，睡眠巩固几乎没有实际写入，慢记忆稳定主要来自“快权重隔离”而非
“巩固”。正式版需要：基线 lr sweep、更大适应预算、以及把 Q 更新改为更可靠的
学习成功信号。

## Stage 4 FORMAL（5 seeds + AdamW lr sweep + 修好的 Q）

设置：100 步/阶段（sweep 60 步），每 seed 用不同的维基/SFT 切片，固定评测批次；
评分 `score = gain_B − 2·forgetting_A`。sweep 结果：

* AdamW：lr 1e-4 最优（+0.027），lr 越大越差（6e-4 为 −0.299）
* DLA：eta_fast 6e-4 最优（+0.133），且三种 eta 全部为正

正式结果（5 seeds，均值 ± std）：

| 指标 | AdamW (lr=1e-4) | DLA (eta=6e-4) |
|---|---|---|
| B 域适应 gain | +13.0% ± 2.4 | **+14.0% ± 2.5** |
| A 遗忘（学完 B） | +10.1% ± 2.2 | 快记忆 **+7.3% ± 1.2**；慢记忆 **−0.4% ± 0.2** |
| A 重学恢复 | +7.1% | +6.9% |
| A 慢记忆 PPL 全程 | 35.5 → 40.2 → 37.3 | 35.5 → 35.4 → 35.4 |

**判定：机制成立（Stage 4 通过）**。在最佳调参下，DLA 与 AdamW 适应能力持平
（+14.0% vs +13.0%），但全权重重遗忘降低 28%，且 **W_slow 层对旧域零遗忘**。
局限：B 的知识目前主要由 W_fast 持有（`gain_B_slow ≈ 0`），睡眠写入慢记忆的
量还太小——这是 Stage 4.5 / 后续版本要修的“巩固深度”，但不影响 fast/slow
分离机制本身的结论。

## Stage 5：递进课程 + Λ_t / Learning Efficiency（5 seeds）

课程按出生模型在该 seed 的 PPL 自动从易到难排序（如 sft→wiki→science）。
原始指标（固定 4% 目标）：

| arm | Λ_t 三个课程阶段 | 趋势 |
|---|---|---|
| AdamW | [0.150, 0.175, 0.050] | 随难度下降 |
| DLA fast | [0.150, 0.050, 0.050] | 随难度下降 |
| DLA slow | [0, 0, 0] | 0 |

固定阈值把 Learning Capacity 和 Task Difficulty 混淆了（正是运行后发现的
confound），因此增加难度归一化指标：

    G_max(t) = 该个体在该阶段 80 步内的最佳增益
    LE_t     = G(final) / G_max
    T80_t    = min{k : G(k) >= 0.8 G_max}

| arm | LE_t 三阶段 | T80_t | LE 斜率 |
|---|---|---|---|
| AdamW | [0.634, 0.865, 0.253] | [44.8, 14.4, 20.0] | −0.19 |
| DLA fast | [0.794, 0.702, 0.693] | [44.0, 32.8, 38.4] | −0.05 |
| DLA slow | 0（快权重承担全部学习） | 80/80/80 | 0 |

**判定：核心发育假设（`LE_DLA(t+1) > LE_DLA(t)`）未被支持。** DLA 的归一化
效率几乎平坦（0.79→0.70→0.69），没有随发育上升；但它不随难度崩坏（AdamW
在第三阶段掉到 0.25，DLA 保持 0.69）。同时 DLA slow 增益仍为 0：睡眠巩固
依然没有把新知识写进慢记忆。

这是有价值的 negative result：当前 DLA 能做到**适应 + 抗遗忘 + 效率稳定**，
但还没有证据表明它能**改变自身的学习效率**。下一步方向因此明确：
（a）让 φ/学习规则参数本身在生命周期内可更新（而不是只有 P 在动）；
（b）把巩固做深，让 slow 真正获得新知识；
（c）如果仍是 flat，论文叙事应改成“发育式记忆分离”，而非“学习能力增长”。

## Stage 5.5：让 learning rule（φ）发育（3 seeds 正序 + 2 seeds 反向课程）

三臂：AdamW / DLA-static（φ 冻结）/ DLA-meta（每进入新阶段前，φ 对 **future
stage 的损失**做一步 meta-gradient，`φ ← φ − β ∇_φ L_future`）。φ 是 9 个
tempo 参数（η_fast, η_plast, fast_decay, stability, α_q, 三个 consolidate 率）。

| arm | LE_t 三阶段 | LE 斜率 | 未见域 D 的 LE_D |
|---|---|---|---|
| AdamW | [0.994, 0.960, 0.443] | −0.276 | 0.282 |
| DLA-static | [0.999, 0.964, 0.243] | −0.378 | **0.474** |
| DLA-meta | [0.997, 0.941, **0.539**] | **−0.229** | 0.230 |

Δφ 记录：DLA-meta 每次 meta 更新的 φ 变化范数 ≈ 0.0028（非零、可测）。

反向课程（2 seeds）：

| 历史 | DLA-static T80_D | DLA-meta T80_D |
|---|---|---|
| easy→hard | 36 / 56 | 36 / 56 |
| hard→easy | 40 / 32 | **24 / 12** |

**判定：部分正信号，但不构成“LE 随发育单调上升”的证据。**
DLA-meta 在正序课程的最难阶段 LE 明显高于 static（0.54 vs 0.24），且 LE 衰减
斜率最缓；反向课程中 hard→easy 的 meta 个体在未见域 D 上 T80 更快（24/12 vs
40/32）。即：**learning history 确实改变了 φ 和未来学习表现**，但该优势没有
稳定迁移到 LE_D（forward 序下 meta 反而低于 static，seed 方差很大）。

结论：φ 能发育、发育有方向性影响，但**尚无稳定证据证明发育让未来学习更好**。
这是一个可写进论文的 nuanced result，不是干净的成功，也不是干净的失败。

## Stage 5.5b：2×2 Development 验证（5 seeds）

设计：`curriculum history (easy→hard / hard→easy) × learner (DLA-static / DLA-meta)`，
5 seeds × 4 conditions，每个个体最后在从未见过的 science 域 D 上测 LE_D / T80_D。

| condition | LE_D（5 seeds） | mean | T80_D mean |
|---|---|---|---|
| easy→hard / static | [1.00, 0.17, 0.17, 0.79, 0.41] | 0.506 | 40.0 |
| easy→hard / meta | [1.00, 0.00, 0.62, 0.91, 0.23] | 0.551 | 44.0 |
| hard→easy / static | [1.00, 0.88, 0.86, 1.00, 0.00] | 0.748 | 34.4 |
| hard→easy / meta | [1.00, 0.84, 0.91, 0.91, 0.47] | 0.824 | 37.6 |

Development Gain（meta − static）：

| metric | easy→hard | hard→easy |
|---|---|---|
| DG (LE_D) | +0.045（weak +） | +0.076（weak +） |
| DG_T (static T80 − meta T80) | −4.0 | −3.2 |
| interaction LE | +0.031 | |
| interaction T | +0.8 | |

φ 距离：`||φ_final(EH) − φ_final(HE)||` = 0.00321（non-zero，历史确实改变 φ）。

**判定：历史效应大而稳定，development gain 小而不稳定。**
* **History × D 主效应强**：hard→easy 的两类 learner 在 D 上都明显优于 easy→hard
  （LE_D 0.75–0.82 vs 0.51–0.55）——经历顺序塑造未来学习者是可重复的。
* **Meta vs static 的差距弱**：meta 平均略高于 static（DG 正），但每个 seed 有正
  有负；T80 上 meta 没有稳定优势（DG_T 平均为负）。5 seeds 尚不能拒绝
  “meta 并不比 static 更利于迁移”的原假设。
* **Development 核心假设（LE_D_meta > LE_D_static）未通过严格检验。**

这仍是有价值的科学结论：**Development（经历顺序）确实 shaping learner；但当前
的 meta 学习规则更新方式还不足以稳定地把这种 shaping 变成迁移收益。** 下一步应
该改进 meta 更新（更多在线步、更强信号、或让 φ 维度更大），而不是扩大跑量。

## Stage 5.5c：机制与因果拆解（3 seeds）

在 5.5b 基础上新增 state snapshots、2×2 cross-injection、φ sensitivity。

Cross-injection 2×2（Body=W+P+W_fast+Q，φ=9 个 tempo；统一未见域 D）：

| Body / φ | LE_D（3 seeds） | mean | T80 mean |
|---|---|---|---|
| EH / EH | [1.00, 0.00, 0.57] | 0.522 | 34.7 |
| EH / HE | [0.00, 0.00, 0.44] | 0.148 | 37.3 |
| HE / EH | [0.87, 0.82, 1.00] | 0.898 | 16.0 |
| HE / HE | [0.99, 0.75, 0.96] | 0.900 | 24.0 |

φ sensitivity（固定 EH body，沿 `d=(φ_HE−φ_EH)/||…||`）：

| α | −1 | −0.5 | 0 | 0.5 | 1 | 1.5 |
|---|---|---|---|---|---|---|
| LE_D mean | 0.998 | 1.000 | 0.358 | 0.000 | 0.000 | 0.000 |

H→State 末态：P_gate 两种历史都≈0.50，差异极小；W_slow norm 只差 ~0.05；
W_fast norm 在 EH 下波动大（6.2–7.9），HE 下更集中（~7.4）。

**判定：Body 是 history effect 的主要中介，9 维 tempo φ 不是稳定的 development
carrier。**
* 换 body 的效应远大于换 φ（HE body mean LE_D 0.90 vs EH body 0.52/0.15）；
* 同一个 body 上换 φ 变化很小且方向不一致；
* φ sensitivity 非单调、方向与“HE 更好”相反——EH→HE 的 φ 方向本身不携带
  “更好的未来学习者”。

因此：**停止追“更强 meta φ”。** 论文主线应采用结局 B：
Learning History → Learner State（Body：W/P/fast-slow）→ Future Adaptation，
Development 主要由学习积累出来的 body 状态介导，而不是由 9 个 tempo 介导。
下一步可做 body 状态的具体分解（哪一层、哪个矩阵、哪个状态分量中介）。

## 下一步

1. Stage 5.5b：把 meta 更新从“阶段边界一步”改成终身在线（每阶段内多次、用
   重放缓冲），并加 5 seeds；看 φ 发育优势能否稳定迁移到 LE_D。
2. Stage 4.5：增大慢记忆巩固，让 B 真正进入 W_slow（与 5.5b 并行或随后）。
3. 若仍无稳定优势，论文叙事采用结局 B：
   “Adaptive plasticity + fast/slow memory + difficulty robustness”，
   不声称 learning ability grows。
