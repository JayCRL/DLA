# DLA v0.2 — Developmental Learning Architecture

> 研究对象不是“再加一个 fast weight”，而是：**学习历史在哪里留下痕迹、这个痕迹由什么构成、又由什么不构成。**
> 论文主线：**Emergent Selective Learning in Fast/Slow Learners**（Draft v0.8）——没有显式选择目标时，分配级选择性如何涌现且因果必要。

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

### ✅ 已证实 / 强证据（n=12 配对 + 对照；数字统一取自 unified batch）
1. **历史效应因果定位到 `W_fast`（carrier localization）**：HE body + EH W_fast 显著破坏未见域 D 的未来适应（gain@40 Δ≈−0.019，CI 不含 0，d≈−0.7，10/12 负）；norm/shuffle/module 对照不 rescue；第二骨干方向复现（n=9，t=−7.13）。
2. **巩固经由 DIRECT 直写（γ·W_fast→W_slow）起作用**：消融 n=12——`direct≈full`（归档比较 t=0.59），`nocons≪direct`（t=+7.36）。
3. **直写的“坐标分配”是必要的（自组织选择性分配）**：能量匹配的坐标打乱对照 `shufwrite≪direct`（n=12，配对 t=+5.45），`shufwrite≈nocons`（Δ=−0.0002，t=−0.17）→ 系统在没有显式选择目标的条件下，自发地“选择往哪些坐标写”。
4. **写入强度稳健性（非刀刃效应）**：把写入强度扫 3× 范围（γ_scale 0.5/1.0/1.5，各 n=12），gap 单调上升（+0.0024 / +0.0115 / +0.0201），且相邻两步各自显著（t=+4.42 / +3.57）；更关键的是**打乱的写入在任何写入强度下都趴在 nocons 地板上**（Δ=+0.0008 / −0.0002 / +0.0013，均 n.s.），而匹配写入随强度抬离地板（+0.0032 / +0.0113 / +0.0215，t=+2.38 / +7.36 / +6.58）——即“加能量救不回错误落点”。

### 🔶 相关级 / 方向性特征（不是因果）
5. **涌现方向性**：逐 seed 的 D-初始梯度与其自身 ΔW_fast 的对齐预测 HE 适应（r=0.87，FDR q=0.003，LOO 稳；控制 norm confound 后 partial≈0.77）；模块级收敛到 MLP/attention（与因果模块一致）。**仅为相关**（无方向操纵实验）。
6. **宏观几何不是载体**：‖ΔW‖/cos(EH,HE) 与同历史跨 seed 噪声不可区分；PCA 无主导共享方向（PC1≈20%）。

### ⚠️ 不成立 / 负结果
7. **“选择性巩固（success 门控 Q）”在当前实现里不成立**：`success` 是全局标量（只有 step 级 gating，无 parameter/subspace 级选择）；‖Q‖≈0.002（direct 写回的 ~0.4%）；`qonly≈nocons`（因果惰性）。
8. **“经验越多 → 学得越快”（B 命题）不成立**：Stage 5–8 多协议阴性（p=0.17 / 0.82 / d=−0.17）。

### 一句话核心结论
> 没有任何显式选择目标时，历史通过 `W_fast` 留下涌现方向性痕迹，任务边界用**标量均匀规则**把它写回慢权——但这个无选择规则的**坐标分配是因果必要的**（能量匹配 shuffle，n=12，t=5.45；且在 3× 写入强度范围内稳健，打乱写入永远趴在无巩固地板上）= **涌现的分配级选择性学习**；代码里唯一的显式“选择”（success 门控 Q）数值与因果双重惰性。

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
| `analysis/wfast_geom/unified_summary.py` | unified batch（12 seed × 6 variant，含 retention）汇总 | 边界管 retention、匹配写入管前进 |
| `analysis/wfast_geom/audit_t10.py` | 10 任务跨域持续学习（n=8）+ 逐任务相对遗忘 | direct 对最早任务保留最好 |
| `analysis/wfast_geom/audit_second.py` | 第二骨干（Shakespeare，n=9）+ 打乱注入对照 | 载体结果跨骨干复现（t=−7.13） |
| `analysis/wfast_geom/a1_final.py` | 写入强度 γ_scale 扫描 + 睡眠衰减扫描分析 | gap 单调、打乱写入恒在地板 |
| `analysis/wfast_geom/a1_dose_response_fig.py` | 剂量-反应图（Figure 11） | 同上的可视化 |

---

## 四、仓库结构

```
dla/               DLA 实现（transformer_dla.py 为论文主实现）
experiments/       stage1..8 / validation 实验脚本
analysis/wfast_geom/  机制审计工具 + report_output/（中间报告、审计报告、selectivity n=12）
paper/
  DLA_paper_draft.md   论文 v0.8（emergent selective learning + 机制链形式化 + 写入强度稳健性）
  DLA_paper_draft.pdf  由 md2pdf（pandoc + headless Chrome）生成
  figures/              图（Figure 8 消融&分配 shuffle · Figure 9/10 机制链 · Figure 11 写入强度剂量-反应）
docs/
  experiments.md       分阶段详细结果与判定
  mechanism_chain.md   机制链形式化（恒等式 / 中间量 / 分配场 / 一阶解释 / falcifier）
  assets/              **展示材料，非科研材料**（含 DLA_cover.png 宣传图，版头为虚构，见该目录 README）
results/
  mac_audit/           **Mac 侧审计运行的逐 seed 原始数据**（消融 / shuffle / γ·δ 扫描 / 第二骨干 / T10）
                       路径镜像报告里引用的 ~/llm-lab/{dla_audit,dla_audit_second,dla_audit_t10}，
                       见 results/mac_audit/README.md（含臂→报告对应表与复现说明）
  server_archive/      **服务器（linghang1，CPU 期）results + logs 的逐 seed 原始数据**
                       含论文 `full` 基线（results/stage55e/seeds/，n=12，HE/HE ≡ +0.0143），
                       见 results/server_archive/README.md（排除 3.9 GB *.pt 权重）
  cloud/ · alloc_cloud/ · scale_cloud/   云端（GPU）运行归档
README.md
```

审计/结果报告：`analysis/wfast_geom/report_output/`
- `interim_report.md`（P1/P2 几何 + 轨迹 + PCA + 鲁棒性）
- `mechanism_static_audit.md`（纯代码审计：无显式对齐、success 全局标量）
- `min_b3_results.md`、`selectivity_test.md`、`selectivity_test_n12.md`（消融 + 能量匹配 shuffle；**n=4+8 独立系列**，见下）
- `audit_batch2_retention_alloc.md`（**unified batch，论文数字的权威来源**：前进 + retention）
- `t10_continual_learning.md`、`t10_forgetting/`（10 任务持续学习 + 逐任务遗忘）
- `a5_second_backbone.md`（第二骨干 n=9）
- `a1_gamma_prelim.md`（n=3 预实验）、`a1_gamma_final.md`（**n=12 写入强度/睡眠衰减扫描**）
- `mechanism_chain/`（Figure 9/10 的生成数据）

> 上列报告的**逐 seed 原始数据**（报告正文引用的 `~/llm-lab/dla_audit*` 路径）已归档到
> `results/mac_audit/`，臂→报告对应表与复现命令见 `results/mac_audit/README.md`。

---

## 五、运行

- 服务器（CPU，旧数据/存档）：`wust_1@192.168.2.2`，`~/llm-lab/dla-v0.2`，`~/llm-lab/venv`
- Mac（审计/消融等重活，M2）：仓库 `~/Desktop/dla-v0.2`，数据镜像 `~/llm-lab/{nanoGPT,corpus,datasets}`，`~/.venv`（torch 2.10，parity 已验证：与存档 ppl 偏差 ~0.24）
- 协议统一：block 128 / batch 32 / 40 步 / eval every 2；审计用同协议 + per-seed 配对统计

```bash
# 服务器：测试 / stage7 / stage8 见 README 历史版本与 docs/
# 审计（Mac，示例）：
TORCH_THREADS=3 ~/.venv/bin/python analysis/wfast_geom/audit_b3.py --seed 0 --order HE --variant direct --out ~/llm-lab/dla_audit
# 写入强度/睡眠衰减扫描分析 + 剂量-反应图：
~/.venv/bin/python analysis/wfast_geom/a1_final.py
~/.venv/bin/python analysis/wfast_geom/a1_dose_response_fig.py
# PDF 论文：
python3 paper/md2pdf.py paper/DLA_paper_draft.md paper/DLA_paper_draft.pdf
```

> **术语**：扫描里的 `gamma`（γ_scale）是 `audit_b3.py --gamma` 的**缩放系数**，乘在**可学习的** `consolidate_fast_direct` 系数上（`dla/transformer_dla.py` 配置默认 0.15，经 sigmoid）。所以 γ_scale=1.0 是项目默认设置，**不是系数等于 1.0**。

---

## 六、论文与复现

- 论文：`paper/DLA_paper_draft.md` / `.pdf`（v0.8，claims×证据级别对齐）
- **聚合口径（重要）**：论文全部数字取自 **unified batch**（`audit_batch2_retention_alloc.md`）。此前的 n=4+8 系列（`selectivity_test_n12.md`）**保留作为独立复现**——两者在核心对比上一致（配对 t=+4.86 与 +5.45），即核心分配结论已独立复现；仅与归档服务器 `full` 的比较在正文中另行标注。
- 审计证据链：上述 `report_output/` 各报告；**逐 seed 数据在 `results/mac_audit/`**（Mac 侧审计原始 JSON，路径镜像报告引用的 `~/llm-lab/dla_audit*`；臂→报告对应表见该目录 README），脚本在 `analysis/wfast_geom/` 与其 `queue/` 驱动，commit 为运行窗口 `a365bcf`→`828c49c`
- 服务器期（CPU）逐 seed 数据在 `results/server_archive/`，**论文 `full` 基线即 `results/stage55e/seeds/` 的 `HE/HE` 条件**（n=12，均值 +0.0143，与 `selectivity_test_n12.md` 所引数字一致；Mac parity 重放为 +0.0138）。排除项：3.9 GB `*.pt` 权重
- 写入强度扫描中默认臂 seed 0–2 为**重跑**：重算 direct(n=12)=+0.0143 vs 扫描前 unified 的 +0.0140（Δ=0.0003，≈单 sd 的 2%，源于 harness 非确定性；配对 **gap 两者完全相同 = +0.0115**）
- **定位（2026-09-11 定）**：本文**不是**提出高性能持续学习算法，而是一套**上下文校准实验**，用于剖析一种涌现式可塑性机制；性能只作为"现象存在"的佐证，不与 SOTA 方法比拼整体效果。论证骨架与证据分级见 `docs/论文思路_中文版.md`（§4.2 为定位与基线口径）。
- **基线对比口径（重要）**：`paper/validation_audit.md` §8/§9 的 **EWC/SI 数字作废**——早于 `ad97334`（EWC/SI 惩罚项曾脱离计算图）；且其中常被引用的 "DLA 0.517 vs EWC 0.895" 是**跨列比较**（0.517 是 DLA 的 EH 列，0.895 是 EWC 的 HE 列）。同列读法见该文件顶部的 VOID 说明；修后 n=10 参考 `results/cloud/dla_cl/gpt2/`。**在 EH/HE 协议上重跑基线之前，不得引用该表。** `paper/DLA_paper.tex` 是 2026-09-08 的旧产物（已被 md 稿与 ICLR 包取代），其表注已标 VOID。
- GitHub：https://github.com/JayCRL/DLA

---

## 七、诚实声明与局限

- 因果/消融主张来自**单一 6.59M 中文 char-GPT、n=12、同 seed 配对**；原始 EH/HE 差距跨 seed 噪声大。
- “方向性对齐”仅为**相关级**（无方向操纵实验）。
- retention（旧任务记忆）**已在同一协议下测量**（unified batch 的 end-of-history ppl + 10 任务序列）：**边界（decay+reset）管旧任务保留，匹配写入管前进适应**——两者是分开的两半。
- Q/success 门控的“选择性巩固”**在当前实现中被证伪**（数值+因果惰性）——不要把它当作已实现机制写入结论。
- 写入强度扫描只动了 direct 写入系数；**睡眠衰减轴仅 n=6**（指示性，不构成结果）。
- 无 pre-registration；n=10→n=12、所有阴性结果均透明记录；不挑 seed、不调参 rescue（审计规则）。
