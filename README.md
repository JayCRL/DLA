# DLA v0.2 — Developmental Learning Architecture

> 研究对象不是“再加一个 fast weight”，而是：**学习历史在哪里留下痕迹、这个痕迹由什么构成、又由什么不构成。**
> 论文主线：**Emergent Selective Learning in Fast/Slow Learners**（Draft v0.4）——没有显式选择目标时，分配级选择性如何涌现且因果必要。

---

## 一、这是什么

DLA 是一套“不修改 Transformer 骨架”的发育式学习研究装置。它在标准权重旁挂上逐参数的发育状态：

| 状态 | 含义 |
|---|---|
| `W_slow` | 长期知识（慢权重） |
| `W_fast` | 当前学习轨迹（快权重，对 Adam 整形后梯度的**泄漏积分**，≈1 个 stage 的记忆窗） |
| `P` | 逐参数可塑性（softplus 正缩放，只调幅度不改符号） |
| `Q` | 睡眠资格迹（`EMA(dw·success)`，success 为**全局标量**） |
| `m/v` | 快权重 Adam 力矩 |

有效权重：`W_eff = W_slow + softplus(P)·W_fast`。睡眠（任务边界）有**两条写回通路**：

```
W_slow += beta*Q + gamma*W_fast      # Q 通路（success 门控） + DIRECT 通路（无选择直写）
W_fast *= 0.5 ; Q *= 0.7 ; reset m/v
```

研究主线：`Learning History → Learner State（Body） → Future Adaptation`，最新论文以 **“涌现选择性学习（allocation-level selectivity）”为核心论点**；每条主张标注证据级别（causal / controlled / correlational / negative），论文 §5 给出 Statement×Evidence 总表。

---

## 二、最重要的实验结果（按证据级别）

### ✅ 已证实 / 强证据（n=12 配对 + 对照）
1. **历史效应因果定位到 `W_fast`（carrier localization）**：HE body + EH W_fast 显著破坏未见域 D 的未来适应（gain@40 Δ≈−0.019，CI 不含 0，d≈−0.7，10/12 负）；norm/shuffle/module 对照不 rescue；第二骨干方向复现。
2. **巩固经由 DIRECT 直写（γ·W_fast→W_slow）起作用**：消融 n=12——`direct≈full`（t=0.6），`nocons≪direct`（t≈5.3）。
3. **直写的“坐标分配”是必要的（自组织选择性分配）**：能量匹配的坐标打乱对照 `shufwrite≪direct`（n=12，配对 t≈4.9），`shufwrite≈nocons` → 系统在没有显式选择目标的条件下，自发地“选择往哪些坐标写”。

### 🔶 相关级 / 方向性特征（不是因果）
4. **涌现方向性**：逐 seed 的 D-初始梯度与其自身 ΔW_fast 的对齐预测 HE 适应（r=0.87，FDR q=0.003，LOO 稳；控制 norm confound 后 partial≈0.77）；模块级收敛到 MLP/attention（与因果模块一致）。**仅为相关**（无方向操纵实验）。
5. **宏观几何不是载体**：‖ΔW‖/cos(EH,HE) 与同历史跨 seed 噪声不可区分；PCA 无主导共享方向（PC1≈20%）。

### ⚠️ 不成立 / 负结果
6. **“选择性巩固（success 门控 Q）”在当前实现里不成立**：`success` 是全局标量（只有 step 级 gating，无 parameter/subspace 级选择）；‖Q‖≈0.002（direct 写回的 ~0.4%）；`qonly≈nocons`（因果惰性）。
7. **“经验越多 → 学得越快”（B 命题）不成立**：Stage 5–8 多协议阴性（p=0.17 / 0.82 / d=−0.17）。

### 一句话核心结论
> 没有任何显式选择目标时，历史通过 `W_fast` 留下涌现方向性痕迹，任务边界用**标量均匀规则**把它写回慢权——但这个无选择规则的**坐标分配是因果必要的**（能量匹配 shuffle，n=12，t≈4.9）= **涌现的分配级选择性学习**；代码里唯一的显式“选择”（success 门控 Q）数值与因果双重惰性。

---

## 三、实验阶段速查

| 脚本 | 实验 | 关键结论 |
|---|---|---|
| `stage1–3` | MLP 规则网络 / φ 发育（旧 MLP 支线，非论文主线） | — |
| `stage4_formal.py` | Transformer + AdamW LR sweep | 慢记忆零遗忘 |
| `stage5/55*` | 递进课程 / φ / body / W_fast 因果拆解 | W_fast 是主要载体 |
| `stage6/7/8*.py` | 同难度 / 跨域 10 任务 / 物理近迁移 | B 不成立 |
| `validation_*.py` | norm/shuffle/module 对照、fair baselines、第二骨干 | 破坏效应稳健、非幅度/随机 |
| `analysis/wfast_geom/geom.py` | ΔW_fast 几何 + 同历史跨 seed null | 宏观几何非载体 |
| `analysis/wfast_geom/replay.py` | 存档 D-probe 的确定性仪器化重放（48/48） | 轨迹梯度（仅摘要） |
| `analysis/wfast_geom/p1*.py` | alignment 相关 + permutation + LOO + FDR | cos0→gain r=0.87（唯一 FDR 存活） |
| `analysis/wfast_geom/p2.py` | ΔW PCA + 投影预测 | PC1 仅 20%，方向弥散 |
| `analysis/wfast_geom/p4_*.py` | Q/W_slow 历史信号与对齐 | Q 无信号、W_slow 弱 |
| `analysis/wfast_geom/audit_b3.py` | 巩固通路消融 + 能量匹配 shuffle（Mac，n=12） | direct 载效应、Q 惰性、坐标分配必要 |
| `analysis/wfast_geom/audit_cheap.py` | A3 confound 偏相关、B2 写入量级、分阶段 trace | Q≈0.002、写入比 ~0.4% |

---

## 四、仓库结构

```
dla/               DLA 实现（transformer_dla.py 为论文主实现）
experiments/       stage1..8 / validation 实验脚本
analysis/wfast_geom/  机制审计工具 + report_output/（中间报告、审计报告、selectivity n=12）
paper/
  DLA_paper_draft.md   论文 v0.3（mechanism-audit）
  DLA_paper_draft.pdf  由 md2pdf（pandoc + headless Chrome）生成
  figures/              图（含 Figure 8: consolidation ablation & allocation shuffle）
docs/experiments.md    分阶段详细结果与判定
README.md
```

审计/结果报告：`analysis/wfast_geom/report_output/`
- `interim_report.md`（P1/P2 几何 + 轨迹 + PCA + 鲁棒性）
- `mechanism_static_audit.md`（纯代码审计：无显式对齐、success 全局标量）
- `min_b3_results.md`、`selectivity_test.md`、`selectivity_test_n12.md`（消融 + 能量匹配 shuffle）

---

## 五、运行

- 服务器（CPU，旧数据/存档）：`wust_1@192.168.2.2`，`~/llm-lab/dla-v0.2`，`~/llm-lab/venv`
- Mac（审计/消融等重活，M2）：仓库 `~/Desktop/dla-v0.2`，数据镜像 `~/llm-lab/{nanoGPT,corpus,datasets}`，`~/.venv`（torch 2.10，parity 已验证：与存档 ppl 偏差 ~0.24）
- 协议统一：block 128 / batch 32 / 40 步 / eval every 2；审计用同协议 + per-seed 配对统计

```bash
# 服务器：测试 / stage7 / stage8 见 README 历史版本与 docs/
# 审计（Mac，示例）：
TORCH_THREADS=3 ~/.venv/bin/python analysis/wfast_geom/audit_b3.py --seed 0 --order HE --variant direct --out ~/llm-lab/dla_audit
# PDF 论文：
python3 paper/md2pdf.py paper/DLA_paper_draft.md paper/DLA_paper_draft.pdf
```

---

## 六、论文与复现

- 论文：`paper/DLA_paper_draft.md` / `.pdf`（v0.3，claims×证据级别对齐）
- 审计证据链：上述 `report_output/` 各报告；逐 seed 数据、脚本与 commit 均在仓库
- GitHub：https://github.com/JayCRL/DLA

---

## 七、诚实声明与局限

- 因果/消融主张来自**单一 6.59M 中文 char-GPT、n=12、同 seed 配对**；原始 EH/HE 差距跨 seed 噪声大。
- “方向性对齐”仅为**相关级**（无方向操纵）；retention（旧任务记忆）未在同一协议下测量（probe 不 sleep）。
- Q/success 门控的“选择性巩固”**在当前实现中被证伪**（数值+因果双重惰性）——不要把它当作已实现机制写入结论。
- 无 pre-registration；n=10→n=12、所有阴性结果均透明记录；不挑 seed、不调参 rescue（审计规则）。
