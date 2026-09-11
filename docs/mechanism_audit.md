# DLA 机制审计 —— 声称的 mechanism vs 实际实现的 mechanism

审计范围：`34cf348` → `dc83280`（45 commits）
审计原则：只报告 diff 字面上做了什么；**不为让论文故事完整而替旧结论辩护**。

---

## 0. 结论摘要（先说最重要的一条）

**论文的中心声明 "emergent selective learning" 目前的实验不支持。**
现有证据支持的是一个更弱、也更平庸的命题：**"写回必须保持与梯度的逐坐标对齐"**。

三处独立证据都指向同一结论：

| 证据 | 结果 | 含义 |
|---|---|---|
| W_fast 的集中度 vs 高斯 null | 只高 **1.2 倍**（gini 0.507 vs 0.414） | W_fast 在幅度上**几乎不选择性** |
| `topwrite`（本该检验集中化） | 实际实现**丢掉了符号**，对齐度归零 | 该对照**无法**检验集中化 |
| 论文 §4.4 的一阶解释 | $\Delta L_D \approx \langle \nabla L_D, A\rangle$ | 这是**对齐**账户，不是**选择性**账户 |

而且论文 §4.4 把 "selective" **定义成**了 shuffle 对照本身
（`selective ⟺ τ_shuf < 0`）——这是**定义性**动作，不是经验发现。

---

## 1. 逐 commit 审计表

| commit | 实际实现的 mechanism | 原本声称的结论 | 现在是否成立 | 需要如何重写 |
|---|---|---|---|---|
| `dc83280` | 无机制改动（收录队列脚本） | — | — | — |
| `03c72a4` | **新增显式 selector**：`W_slow += γ·ŝ(\|W_fast\|)⊙Q` | "把 W_fast 当筛选器" | ⚠️ 属 **hypothesis line** | 不得作为自发涌现证据；单独章节 |
| `697452d` | `softplus(P)⊙W_fast` 写入 | "门控写回" | ❌ **方向仍是错的**：还是把 W_fast 当**内容**写 | 标注为 dead end |
| `6a76455` | 无机制改动（.gitignore + 归档） | — | ✅ | — |
| `d77842d` | 无机制改动（报告/图脚本） | — | ✅ | — |
| `cabb18c` | 拆 `fast_decay`(时间尺度) 与 `λ`(写回增益) | "分离涌现与表达" | ✅ 工具成立 | 结论待二维数据 |
| `4ab3127` | 基线自身参数扫描 + Pareto | "比较前沿而非单点" | ✅ | — |
| `ad97334` | **修 EWC/SI 惩罚项脱离计算图** | — | ✅ 修复正确 | 修前的 EWC/SI 结果作废 |
| `bb5460a` | 三指标评分 | — | ✅ | — |
| `07e6e6d` | P1 `audit_alloc.py` + P2 `cl_baselines.py` | 分配选择性因果检验 | ✅ 设计正确 | 见 §2 |
| `d1afaf2` | 修 vocab zero-padding | — | ✅ | — |
| `ccbe75b` | 大 backbone 探针可测 | — | ✅ | — |
| `c36d57a` | 无机制改动（LaTeX 投稿包） | — | — | — |
| `4e005fe` | 规模化探针 GPT-2 355M/774M/1.5B | 规模复现 | ✅ | — |
| `38dfa66`/`4803502` | 无机制改动（排版） | — | — | — |
| `cac3b23` | A1x 写入强度(γ)/sleep-decay 扫描 n=12 | "写入强度稳健" | ⚠️ 见 §2.3 | gap 随 γ 单调 → 也是对齐账户的预测 |
| `828c49c`/`a218fcc` | 机制链形式化 + 图 | "leaky 恒等式、分配场、shuffle 判据" | ⚠️ 形式化里已含对齐账户 | 第 4 环应改名"对齐"而非"选择性" |
| `814009a` | T10 逐任务遗忘审计 | 遗忘指标 | ✅ | 已知缺中间检查点，已声明 |
| `a5af287` | 第二 backbone n=9 | swap 复现 | ✅ | — |
| `43b0bf1` | 无机制改动（论文 v0.5） | — | ⚠️ 集成时把"选择性"写进标题 | 需降级措辞 |
| `10685ac` | T10 十任务 CL（种子 0-7） | "涌现选择性巩固作为 CL 原语" | ⚠️ CL 现象成立，"选择性"标签不成立 | 改为"直接写回作为 CL 原语" |
| `7f7ae7c` | A1 γ 初扫 n=3 | 收益依赖写入强度 | ✅ 现象 | — |
| `a365bcf` | **统一批次 n=12**：含 `uniformwrite`/`topwrite`/`shufwrite` | "坐标匹配分配必要，非 top-k 或能量" | ❌ **`topwrite` 实现有缺陷**，见 §3.1 | **关键对照必须重跑** |
| `4c5b141` | 汇总脚本 | — | ✅ | — |
| `3d25068`/`559f0b9` | 无机制改动（相关工作/框架） | — | — | — |
| `161da2b` | 论文 v0.4 转向 "emergent selective learning" | 核心声明确立 | ❌ **过度声称** | 退回"coordinate-aligned consolidation" |
| `f041568`/`e61dc35` | README/论文 v0.3 按证据分级重写 | 载体强/方向相关/分配必要 | ✅ 分级诚实 | 分级里"分配"应改为"对齐" |
| `90f8af4` | n=12 选择性确认：`direct≈full`, `nocons`&`shufwrite` ≪ direct | 坐标分配必要 | ✅ `direct>shufwrite` 成立 | 结论措辞 |
| `89b5dc1` | 预算匹配的 shuffle 检验 | "自组织分配选择性" | ⚠️ 对照正确，**解释过度** | 见 §2.1 |
| `16ec9a9` | 诊断脚本（无实验） | — | ✅ | — |
| `57a1b84` | B3 最小化：`direct`/`nocons`/`qonly` | Q 通道因果惰性 | ✅ 成立 | — |
| `723f656` | 静态机制审计（纯代码） | 无对齐算符；success 只做步级门控 | ✅ 成立 | — |
| `946024b`/`46071b3`/`22ba244` | 几何分析工具 P1/P2/P4、反事实插值 P3 | 方向字符相关 | ✅ 相关级 | 论文已标 correlational |
| `acd2089`/`c696953`/`81a7bd1`/`80a70c3`/`33d79a3`/`34cf348` | 无机制改动（论文/图/格式） | — | — | — |

---

## 2. 三条仍然成立的核心结论（保留）

### 2.1 `direct > shufwrite`（能量匹配）—— 成立，但含义要改

P1 报告（`docs/p1_allocation_124M.md`，n=12）：

```
metric        effect      t       wins  pre imb  corr(pre,eff)
final_ppl    +1.0457    +5.94     11/12   +0.326      +0.158
```

**pre 不平衡只有 +0.326、与 pre 的相关只有 +0.158** —— 这条不受"治疗后变量"混淆影响，是干净的。

**但它的含义**：direct 写出的是 `γ·W_fast`，而 `W_fast` 是
`-Σ η·softplus(P)·adam` 的泄漏累积——**即一个下降方向**。shuffle 之后方向被随机化，
`⟨∇L_D, ΠA⟩ ≈ 0`。

> ✅ 成立：**写回必须保持逐坐标的值↔坐标对应关系**
> ❌ 不成立：**这证明了"选择性"** —— 它同样（更简约地）由"对齐 vs 不对齐"解释

### 2.2 Q 通道（success-gated eligibility trace）因果惰性 —— 成立

`qonly ≈ nocons`，Q 写入约占总能量 0.4%。这是**阴性结论**，论文已如实报告。保留。

### 2.3 载体 / 历史效应由 W_fast 承担 —— 成立

swap 因果（n=12）、第二 backbone 复现（n=9）。论文已标 causal。保留。

---

## 3. 三处必须修正的缺陷

### 3.1 `topwrite` 丢掉了符号 —— 关键对照失效 ⚠️ 最严重

`analysis/wfast_geom/audit_b3.py:87-96`：

```python
n_keep = max(1, int(round(0.2 * n)))      # ← 算了但从未使用
k = int(round(0.8 * n))                   # 阈值取 top-80%
thr = flat_abs.topk(k, largest=True).values.min()
mask = wf.abs() >= thr
c = gam * wf.norm() / math.sqrt(keep)     # 正常数
add = torch.where(mask, torch.full_like(wf, c), torch.zeros_like(wf))
```

写入的是**正常数 `c`**，`sign(W_fast)` 被完全丢弃。

数值验证（n=10⁶ 合成）：

| 变体 | cos(g, A) |
|---|---|
| `direct` | **−0.769** |
| `shufwrite` | +0.001 |
| `uniformwrite` | −0.002 |
| **`topwrite`（实际）** | **−0.002** ← 与 shuffle 同级 |
| `topk_signed`（从未跑过） | **−0.619** |

**`topwrite` 的对齐度归零，所以它失败的原因与 `shufwrite` 完全相同**——
不是"集中化没用"，而是"与梯度不对齐"。

**因此论文 §4.4 的这句话无实验支持：**
> "selectivity is not reducible to energy concentration (`uniformwrite`, `topwrite` both ≈ no-consolidation)"

值得注意的是，同一份报告自己写着正确判据：
> "What matters is the **coordinate-matched write (magnitude *and* sign matched to W_fast itself)**"

即报告知道符号关键，**而用来"排除集中化"的实验恰恰抹掉了符号**。

### 3.2 W_fast 在幅度上几乎不集中

| | cv | gini | top1% | top10% | nent |
|---|---|---|---|---|---|
| 标准高斯 N(0,1) | 0.756 | 0.414 | 0.036 | 0.259 | 0.982 |
| 均匀分布 | 0.577 | 0.333 | 0.020 | 0.190 | 0.988 |
| 稀疏(95%零+5%高斯) | 5.513 | 0.971 | 0.440 | 1.000 | 0.788 |
| **实测 W_fast** | **1.019** | **0.507** | **0.047** | **0.352** | **0.977** |

**只比高斯集中 1.2 倍，离稀疏结构差两个数量级。**
"selective" 这个词若指幅度集中，**数据不支持**。

### 3.3 "selective" 被定义成对照本身

论文 §4.4：

```
selective ⟺ τ_shuf := E[G(Φ_shuf)] − E[G(Φ_direct)] < 0   （固定写入能量下）
```

这使中心声明**按定义成立**，但也使它**不再是可证伪的经验命题**。
把它改名为 `coordinate-alignment effect` 才能恢复可证伪性。

---

## 4. 两代实验的严格区分

### Discovery line（原生 DLA，不得新增 selector）

| 变体 | 写入内容 | 能量 | 状态 |
|---|---|---|---|
| `full` | `β·Q + γ·W_fast`（原生规则） | 原生 | ✅ 保留 |
| `direct` | `γ·W_fast` | 原生 | ✅ 保留 |
| `nocons` | 不写 | 0 | ✅ 保留 |
| `qonly` | `β·Q` | 设计通道 | ✅ 保留 |
| `shufwrite` | `γ·perm(W_fast)` | 匹配 | ✅ 保留 |
| `uniformwrite` | 常数铺满 | 匹配 | ✅ 保留 |
| **`topwrite`** | 常数 `+c` 于 top-80% | 匹配 | ⚠️ **有符号缺陷，需重跑** |
| **`topk_signed`** | `γ·W_fast` 仅保留 top-k% 坐标 | 匹配 | ❌ **从未跑过——真正缺失的对照** |

**发现线要证明的命题**：收益与写回的 **coordinate identity / structure** 有关，
而不是单纯写入能量。

**当前状态**：`direct > shufwrite` 与 `direct > nocons` 成立；
但"structure"目前只能被证明到 **"逐坐标对齐"** 这一层，
**"选择性子集"这一层尚未被检验**。

### Mechanism-hypothesis line（必须单独标记）

| 变体 | 性质 |
|---|---|
| `selwrite` | **显式实现** `ŝ(\|W_fast\|)⊙Q` 作为筛选器 |
| `shufsel` | 同上的打乱对照 |
| `unifq` | 无筛选对照 |
| `write_gate` | `softplus(P)⊙W_fast`（仍是内容写，dead end） |
| 任何"由 W_fast 算 mask/gate 再决定写哪里"的方法 | 同属此类 |

**这些只能回答**："如果把某个候选机制显式实现出来，会发生什么？"
**不能用来证明 spontaneous emergence。**

---

## 5. 恢复后的研究主线

**核心问题**：原生 leaky $W_{\mathrm{fast}}$ 为何会 → structured writeback → selective consolidation？

按本审计，这个问题必须**先拆成两个可分别检验的层次**：

```
层次 1（已成立）：  写回必须逐坐标对齐  ⟨∇L_D, γ W_fast⟩ > 0
                    —— 证据：direct > shufwrite（n=12, t=5.94, 干净）
                    —— 机制：W_fast = 泄漏累积的 Adam 下降方向

层次 2（未检验）：  收益是否集中在 W_fast 的稀疏子集？
                    —— 唯一能判定的对照：topk_signed（保留符号的 top-k）
                    —— 现有 topwrite 因丢符号而无效
                    —— 幅度集中度指标（gini 0.507 vs 高斯 0.414）预示"否"
```

**如果层次 2 为否**（预期），则论文的真实贡献应改写为：

> 一个无选择目标的标量 fast→slow 写回，其有效性来自**逐坐标的梯度对齐**；
> 该对齐由 `W_fast` 对自身梯度历史的泄漏累积**自组织**产生，无需任何显式选择机制。
> 同时系统内存在一个**显式设计的选择机制（Q）却是惰性的**。

这仍然是一个**干净、可证伪、且有趣**的结论（"设计出来的选择机制无效，未设计的对齐有效"），
但它**不是** "emergent selectivity"。

---

## 6. 需要立刻做的三件事（按优先级）

1. **重跑 `topwrite`（修符号）+ 新增 `topk_signed`** —— 判定层次 2 的唯一方式。
   这属于发现线的对照补全，不是新增机制。
2. **把 `selwrite`/`shufsel`/`write_gate` 全部移出主结果**，单独成节并显式标注
   "explicitly implemented hypothesis"。
3. **论文全文把 "selective" 降级为 "coordinate-aligned"**，
   除非第 1 项给出支持集中化的结果。

---

*本审计由 `git show` 逐 commit 提取实际 diff 得出；所有数值均来自仓库内归档结果或可复现的本地 null 计算。*
