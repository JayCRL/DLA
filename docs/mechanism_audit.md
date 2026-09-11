# DLA 机制审计 —— 声称的 mechanism vs 实际实现的 mechanism

审计范围：`34cf348` → `2fa1d2a`（45 commits）
审计原则：只报告 diff 字面上做了什么；**既不替旧结论辩护，也不夸大缺陷**。

> ## ⚠️ 更正声明（2026-09-11）
>
> 本文件的首版（commit `2fa1d2a`）标题写着 **"the central 'emergent selective learning'
> claim is not supported"**。**那个结论是错的，现予撤回。**
>
> 错因：审计时读的是摘要 + §4.4 + 部分 §5，**没有读 §3.2「Defining 'selectivity'
> precisely (three levels)」**。那一节明确定义了论文的主张层级：
>
> | level | definition | by design? |
> |---|---|---|
> | step/global | 一个标量决定整次更新的保留/丢弃 | yes (`success`) |
> | parameter | 每个参数不同系数，基于逐参数信号 | **no** |
> | subspace | 沿学到的方向做选择 | **no** |
> | **allocation** | 系数无选择性，但**写入的 magnitude/sign 逐坐标不同** | **yes（涌现）** |
>
> **论文主张的是 allocation 层——「写落在哪些坐标」，不是「W_fast 稀疏/集中」。**
> 我却用"稀疏/集中"这个自带的定义去反驳，**打了一个论文没提出的靶**。
> 首版里"W_fast 不集中所以 selective 不成立"、"selective 被定义成对照所以不可证伪"、
> "一阶账户是对齐账户所以不是选择性"这三条**全部撤回**——它们要么打错靶，
> 要么把论文自己的机制读成了对论文的反驳。
>
> 下面保留的是经复核**确实成立**的部分。

---

## 0. 结论摘要

**论文的中心声明成立，且证据链完好：**

> 一个**标量均匀、无选择目标**的巩固规则（`W_slow += γ·W_fast`），
> 其效果**完全取决于写落在哪些坐标**——保持总能量与规则不变、
> 只打乱坐标分配，效应即被完全消除。

**审计发现的三个问题都是可修的，且都不动摇中心声明**（见 §3）。

---

## 1. 逐 commit 审计表

| commit | 实际实现的 mechanism | 原本声称的结论 | 现在是否成立 | 需要如何重写 |
|---|---|---|---|---|
| `2fa1d2a` | 无机制改动（审计文档） | — | ❌ 首版结论错误 | **本文件即更正** |
| `dc83280` | 无机制改动（收录队列脚本） | — | ✅ | — |
| `03c72a4` | **新增显式 selector**：`W_slow += γ·ŝ(\|W_fast\|)⊙Q` | "把 W_fast 当筛选器" | ⚠️ 属 **hypothesis line** | 移出主结果，单独成节 |
| `697452d` | `softplus(P)⊙W_fast` 写入 | "门控写回" | ⚠️ **仍把 W_fast 当内容写** | 标注为 dead end |
| `6a76455` | 无机制改动（.gitignore + 归档） | — | ✅ | — |
| `d77842d` | 无机制改动（报告/图脚本） | — | ✅ | — |
| `cabb18c` | 拆 `fast_decay`(时间尺度) 与 `λ`(写回增益) | "分离涌现与表达" | ✅ 工具成立 | 结论待二维数据 |
| `4ab3127` | 基线自身参数扫描 + Pareto | — | ✅ | — |
| `ad97334` | **修 EWC/SI 惩罚项脱离计算图** | — | ✅ | **修前的 EWC/SI 结果作废** |
| `bb5460a` | 三指标评分 | — | ✅ | 见 §3.1 |
| `07e6e6d` | P1 `audit_alloc.py` + P2 `cl_baselines.py` | 分配选择性因果检验 | ✅ **设计正确** | — |
| `d1afaf2` | 修 vocab zero-padding | — | ✅ | — |
| `ccbe75b` | 大 backbone 探针可测 | — | ✅ | — |
| `c36d57a` | 无机制改动（LaTeX 投稿包） | — | — | — |
| `4e005fe` | 规模化探针 GPT-2 355M/774M/1.5B | 规模复现 | ✅ | — |
| `38dfa66`/`4803502` | 无机制改动（排版） | — | — | — |
| `cac3b23` | A1x 写入强度(γ)/sleep-decay 扫描 n=12 | 写入强度稳健 | ✅ | — |
| `828c49c`/`a218fcc` | 机制链形式化 + 图 | — | ✅ | — |
| `814009a` | T10 逐任务遗忘审计 | — | ✅ | 缺中间检查点，已声明 |
| `a5af287` | 第二 backbone n=9 | swap 复现 | ⚠️ 见 §3.3 | 措辞需限定 |
| `43b0bf1` | 无机制改动（论文 v0.5） | — | ✅ | — |
| `10685ac` | T10 十任务 CL（种子 0-7） | 直接写回作为 CL 原语 | ✅ | — |
| `7f7ae7c` | A1 γ 初扫 n=3 | — | ✅ | — |
| `a365bcf` | **统一批次 n=12**：`uniformwrite`/`topwrite`/`shufwrite` | 坐标匹配分配必要 | ✅ 核心成立；`topwrite` 有缺陷 | 见 §3.2 |
| `4c5b141` | 汇总脚本 | — | ✅ | — |
| `3d25068`/`559f0b9` | 无机制改动 | — | — | — |
| `161da2b` | 论文 v0.4 转向核心声明 | — | ✅ §3.2 定义严谨 | — |
| `f041568`/`e61dc35` | README/论文 v0.3 按证据分级 | — | ✅ | — |
| `90f8af4` | n=12 选择性确认 | 坐标分配必要 | ✅ | — |
| `89b5dc1` | 预算匹配 shuffle 检验 | 自组织分配选择性 | ✅ **对照正确** | — |
| `16ec9a9` | 诊断脚本（无实验） | — | ✅ | — |
| `57a1b84` | B3 最小化：`direct`/`nocons`/`qonly` | Q 通道因果惰性 | ✅ 成立 | — |
| `723f656` | 静态机制审计（纯代码） | 无对齐算符 | ✅ 成立 | — |
| `946024b`/`46071b3`/`22ba244` | 几何分析、反事实插值 | 方向字符相关 | ✅ 相关级 | — |
| `acd2089`≈`34cf348` | 无机制改动（论文/图/格式） | — | — | — |

---

## 2. 仍然成立的核心证据链

### 2.1 分配必要性（**中心声明**）

能量匹配 shuffle（逐矩阵置换，总能量/规则/系数完全相同）：

```
docs/p1_allocation_124M.md，n=12

metric        effect      t       wins  pre imb  corr(pre,eff)
final_ppl    +1.0457    +5.94     11/12   +0.326      +0.158
```

**`final_ppl` 不含 `pre`，`corr(pre, effect)` 只有 +0.158 —— 干净的因果证据。**

两次独立运行一致：`t=+4.86` 与 `t=+5.45`（§3.3 的 aggregation provenance）。

### 2.2 写入强度稳健（**最强的一条**）

```
γ_scale    direct    shufwrite   paired gap      t
0.5        +0.0059   +0.0035     +0.0024      +1.82 (n.s.)
1.0        +0.0140   +0.0025     +0.0115      +5.45
1.5        +0.0241   +0.0040     +0.0201      +4.78
```

**并且 shufwrite 相对 nocons 地板：+0.0008 / −0.0002 / +0.0013（全部 n.s.），
而 direct 爬升 +0.0032 / +0.0113 / +0.0215（t=+2.38/+7.36/+6.58）。**

> **给错配的写加 50% 能量纯属浪费——起作用的是"写在哪里"，不是"写了多少"。**

这一条同时排除了"刀刃效应"和"写多了就行"两种替代解释。

### 2.3 设计的"选择"机制是惰性的（阴性结论）

`‖Q‖≈0.002` vs `‖W_fast‖≈3.6` —— Q 写入仅占 0.4%。
`qonly ≈ nocons`。**这是真阴性，也是论文的张力所在。**

### 2.4 载体（GPT-2 三规模，n=12×3）

```
swap 效应（HE → 注入 EH 的 W_fast）

model          n |  Δpre      t   |  Δfinal    t      变差
gpt2          12 |  −0.0055  −4.03 |  +0.7752  +9.33  12/12
gpt2-medium   12 |  −0.0086 −11.46 |  +0.4421  +4.14  10/12
gpt2-large    12 |  −0.0125  −6.21 |  +0.9054  +5.91  12/12
```

**`Δpre` 可忽略（−0.006 ~ −0.013），`Δfinal` 明确变差，三个指标方向一致。**

### 2.5 边界保护旧任务

```
nosleep vs direct（retention）:  t=+10.26   ← 不睡显著更差
T10（10 任务）: direct vs nosleep Δ=−1.25, t=−10.27
```

### 2.6 T10 持续学习

```
retention（最早任务 end ppl，越低越好）
  direct 38.27  <  nocons 39.03  <  nosleep 39.51
  direct vs nocons  t=−3.60
  direct vs nosleep t=−10.27
forward（norm_slope10，越负越快）
  direct vs nosleep t=−3.93    （direct 不更慢）
```

---

## 3. 三个需要修正的问题（都不动摇中心声明）

### 3.1 主指标应改用 `final_ppl`

`gain@40` 是**比值** `(pre−final)/pre`，起点差的 arm 机械上占优：

```
direct vs shufwrite（n=12）
  final_ppl  corr(pre, effect) = +0.158   ← 干净
  gain40     corr(pre, effect) = +0.916   ← 结构性混淆
```

**结论不受影响**（两指标同向且都显著；等 pre 校正后 gain40 仍 +0.0184, t=+6.82），
**但报告应把 `final_ppl` 作为主指标。**

### 3.2 `topwrite` 丢符号（实现缺陷，确实存在）

`audit_b3.py:96`、`audit_alloc.py:119`（所有版本）：

```python
c = gam * wf.norm() / math.sqrt(keep)          # 正常数
add = torch.where(mask, torch.full_like(wf, c), torch.zeros_like(wf))
                                ↑ sign(W_fast) 被丢弃
```

合成验证（n=10⁶）与梯度的对齐度：

| 变体 | cos(g, A) |
|---|---|
| `direct` | **−0.769** |
| `shufwrite` | +0.001 |
| `uniformwrite` | −0.002 |
| **`topwrite`（实际）** | **−0.002** |
| `topk_signed`（从未跑过） | **−0.619** |

**`topwrite` 的对齐度归零，所以它失败的原因与 `shufwrite` 相同——不是"集中化没用"。**

**影响范围（精确）**：论文 §4.4 用它支持 "not reducible to energy concentration"
这一句，以及 falsifier (iii) `τ_top ≈ 0`。
**不影响中心声明**——`direct` vs `shufwrite` 是完全独立的对照。

**缺失的对照**：`topk_signed`（保留符号的 top-k% 坐标，能量重标定）。从未跑过。

### 3.3 shuffle 粒度过粗

```python
perm = torch.randperm(flat.numel(), ...)   # 整个展平矩阵上全局置换
```

全局置换**同时破坏了**：(a) 与未来梯度的逐坐标对齐；(b) 矩阵本身的行/列结构。

**更细的渐进控制能把它分开**：

| 控制 | 保留 | 破坏 |
|---|---|---|
| 行内 shuffle | 每个输出神经元的能量 | 输入通道配对 |
| 部分符号翻转 | 全部 magnitude | 符号对齐 |
| 部分 shuffle（10/50/90%）| 大部分配对 | 渐进破坏对齐 |
| `topk_signed` | 符号 + 能量 | 稠密部分 |

**A5 第二 backbone 的内容特异性问题（§4.8）**：

```
HE + EH W_fast（真实）        vs HE :  −0.0673   t=−7.13
HE + shuffled EH W_fast（打乱）vs HE :  −0.0721   t=−7.60   ← 同等甚至更大
HE + EH  vs  HE + shuffled    :  +0.0048   ← 内容特异部分仅 7.1%
```

**建议**：§4.8 的措辞限定为"W_fast 因果地影响状态（任何注入都显著），
历史特定性约 7%"，而不是作为"载体承载历史"的独立复现。
**注意 §4.1（GPT-2 三规模）不受此影响**——那里的 `Δpre≈0`、`Δfinal` 明确变差。

---

## 4. 两代实验的严格区分

### Discovery line（原生 DLA，不得新增 selector）

`full` / `direct` / `nocons` / `qonly` / `shufwrite` / `uniformwrite` / `topwrite`⚠️

**目标命题**：收益与写回的 **coordinate identity** 有关，而非单纯写入能量。

### Mechanism-hypothesis line（必须单独标记）

`selwrite` / `shufsel` / `unifq` / `write_gate` / 任何"由 W_fast 算 mask 再决定写哪里"的方法。

**只能回答**："如果把某个候选机制显式实现出来，会发生什么？"
**不能证明** spontaneous emergence。

---

*本审计由 `git show` 逐 commit 提取实际 diff；数值来自仓库归档结果或可复现的本地 null 计算。*
