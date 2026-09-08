# DLA v0.2 — Developmental Learning Architecture

![DLA v0.2 architecture](docs/architecture.png)

把研究对象从“加 fast weight 的 MLP”推进为：

> **Learning itself can be a developmental state.（学习本身是一种发育状态）**

四层定义与代码对应关系：

| 理论层 | 代码对象 | 位置 |
|---|---|---|
| DNA φ | Learning Rule Network `F_phi` 的权重 + 学习规则参数（η、衰减、巩固率，`TempoParams`）+ 固定的信号先验（可塑性先验、novelty/reward 权重） | `dla/model.py` |
| Learning Rules | 每个连接上的 `F_phi(x_i, h_j, δ_j, w_ij, p_ij, 全局/认知信号) -> (a_heb, a_del, m, pd, qs)` | `dla/model.py::step` |
| Cognitive State | EMA 状态向量 `[knowledge, confidence, uncertainty, skill, fatigue, progress]` | `dla/model.py::_compute_signals` |
| Knowledge / Skill | `W_slow`（长期知识）+ `W_fast`（当前技能痕迹） | `dla/model.py` |

## 三级参数

```
W_slow  长期知识（出生初始化 -> 睡眠巩固 -> 元学习优化）
W_fast  快速权重（每次经验更新，当前学习）
P       逐连接可塑性（meta-plasticity，本身随时间发育）
Q       慢权重资格迹（哪些突触值得在睡眠中巩固）
```

一次经验更新（两个学习基）：

```
W_eff = W_slow + softplus(P) * W_fast

basis  = a_heb * (h x^T)/sqrt(n_in) + a_del * (delta x^T)/sqrt(n_in)
dW_fast = eta_fast * softplus(P) * m * basis - lambda_f * W_fast
dP      = eta_plast * pd * (|h x^T| + |delta x^T|)/sqrt(n_in) - kappa * (P - P0)
dQ      = alpha_q * qs * softplus(P) * m * basis * success
```

其中 `delta_j = dL/d(post_j)` 是局部教学信号（对输出层就是 `p_j - y_j`，隐藏层按
当前有效权重回传）。`F_phi` 逐连接输出 `[a_heb, a_del, m, pd, qs]`：`a_heb` 决定
对 Hebbian 基的信任度（可正可负 = 反 Hebbian），`a_del` 决定对教学基的信任度，
`m` 是幅度门，`pd` 是可塑性变化方向，`qs` 是巩固资格门。因此规则空间同时包含
纯 Hebbian、delta rule 和两者的混合——到底信哪个、每个连接信多少，是**学出来的**，
而不是手工写死的。

`F_phi` 的输入是

```
[x_i, h_j, delta_j, w_fast_ij, p_ij, w_slow_ij]  （局部）
+ [error, novelty, uncertainty, reward, confidence, skill, fatigue, progress, knowledge]（全局/认知）
```

`rule_mode="hebbian"` 时退化为固定规则 `a_heb=1, a_del=0, m=1, pd=0, qs=0.5`，即
固定 Hebbian 对照。`F_phi` 的末层偏置初始化在 Hebbian 角落附近，因此 Stage 2
是公平的“同起点、一个学规则、一个不学”。

睡眠（任务边界）：

```
W_slow += consolidate_beta * Q
W_fast *= consolidate_fast_decay
Q      *= consolidate_q_decay
```

## 双层学习（learning to learn）

- **内层**：`DevelopmentalNet.step` —— `X_{t+1} = L_φ(X_t, E_t)`，全函数式、可微分，
  可整段展开做元梯度。
- **外层（按“人生”训练，不按单任务训练）**：`meta_train(lifetime_len=2,
  retain_weight=1.0)` 把任务成对串成微型人生：学任务 A → **睡眠巩固** → 学任务
  B → **睡眠巩固** → 用巩固后的 `W_slow` 重测 A 和 B。元损失同时包含在线学习损失
  与保持损失，所以 **stability–plasticity 权衡本身成为元训练目标**；元梯度穿过
  睡眠巩固进入 φ（规则网络 + 巩固参数）与 `W_slow` 初值。
- **生命周期内 φ 自适应**：`LifetimeAdapter` 维护个体自己的经历回放 `H_t`，
  周期性执行 `φ_{t+1} = φ_t + G(H_t)` —— 这就是 Stage 3 的 adaptive 臂，
  让“学习规则”成为随时间发育的状态。

`Λ_t`（Learning Capacity）的实验操作化：

```
Λ_t = 1 / (steps_to_reach_85%_test_accuracy_on_task_t + 1)
```

## 目录

```
dla/
  config.py          四层定义、学习规则参数、DNA 信号先验
  model.py           DevelopmentalNet（W_slow/W_fast/P/Q、F_phi、认知状态、睡眠巩固）
  transformer_dla.py Stage 4：把 W_fast/P/Q + Adam 力矩 + 睡眠巩固挂到标准 GPT
                     （不改骨架，每个权重矩阵上做 per-parameter 可塑性）
  meta.py            元训练 + 生命周期内 φ 自适应（LifetimeAdapter）
  tasks.py           自包含任务族：Gaussian / XOR 持续学习流
  metrics.py         终身评估：学习曲线、遗忘矩阵、steps-to-threshold、可塑性轨迹
  baselines.py       StaticMLP（优化器在模型外）与固定 Hebbian 配置
  dna.py             DNA-A/B/C/D 四种“学习倾向”个体
  plotting.py        图表输出（Agg，服务器无显示器可用）
experiments/
  stage1_adaptive_plasticity.py       Stage 1：静态 MLP vs 自适应可塑性
  stage2_learned_rule.py              Stage 2：固定 Hebbian vs 学到的规则 F_phi
  stage3_learning_rule_development.py Stage 3：φ 固定 vs φ 在生命周期内发育
  dna_lifetimes.py                    DNA-A/B/C/D 同一段人生
  stage4_formal.py                    Stage 4 Formal：AdamW LR sweep + 5 seeds
  stage5_progressive_curriculum.py    Stage 5：递进课程 + Λ_t
  stage5_analysis.py                  Stage 5：难度归一化 LE 分析
  stage55_learning_rule_development.py     Stage 5.5：φ 发育对照
  stage55b_development_2x2.py              Stage 5.5b：History × Learner 2×2
  stage55c_causality.py                    Stage 5.5c：因果拆解（body/φ cross）
  stage55d_body_decomposition.py           Stage 5.5d：body 组件分解
  stage55e_wfast_p0.py                     Stage 5.5e P0：W_fast trajectory
  stage6_longitudinal.py                   Stage 6：同难度纵向
  stage7_cross_domain.py                   Stage 7：跨领域 10 任务纵向
  stage7_analyze.py                        Stage 7 分析
  stage8_physics.py                        Stage 8：物理近迁移
  run_all_smoke.py                        全流程冒烟测试
analysis/
  compute_wfast_effective_rank.py          EH/HE 有效秩比较
  compute_wfast_decompose_rank.py          per-module + 差矩阵有效秩
tests/test_core.py                    核心机制 sanity check
```

## 在 linghang1 上运行

```bash
# 1) 上传（在 Mac 上）
scp -r dla-v0.2 wust_1@192.168.2.2:~/llm-lab/

# 2) 服务器：安装绘图依赖（清华 PyPI）
ssh wust_1@192.168.2.2
~/llm-lab/venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib

# 3) 测试
cd ~/llm-lab/dla-v0.2
~/llm-lab/venv/bin/python tests/test_core.py

# 4) 冒烟
~/llm-lab/venv/bin/python experiments/run_all_smoke.py

# 5) 正式实验（按需跑；多数已有结果在服务器 results/ 与 docs/experiments.md）
~/llm-lab/venv/bin/python experiments/stage4_formal.py --seeds 0,1,2 --epochs 100
~/llm-lab/venv/bin/python experiments/stage5_progressive_curriculum.py --seeds 0,1,2 --epochs 100
~/llm-lab/venv/bin/python experiments/stage55e_wfast_p0.py --seeds 0..11 --d-steps 40
~/llm-lab/venv/bin/python experiments/stage6_longitudinal.py --seeds 0..9
~/llm-lab/venv/bin/python experiments/stage7_cross_domain.py --seed 0  # 单个 seed
bash experiments/stage7_run_all.sh                                       # 20 seeds
~/llm-lab/venv/bin/python experiments/stage8_physics.py --seeds 0..19 --train-chars 100000 --val-chars 10000

# 结果在 results/<stage>/ 下：results.json / seed JSON + PNG
```

## 实验结果（截至 2026-09-08）

论文初稿见 [`paper/DLA_paper_draft.md`](paper/DLA_paper_draft.md)；完整数字与解释见 [`docs/experiments.md`](docs/experiments.md) 与 [`paper/overnight_report.md`](paper/overnight_report.md)。

| 阶段 | 结论 |
|---|---|
| Stage 1 | DLA 遗忘约减半（0.099 vs 静态 0.181），样本效率最好 |
| Stage 2 | 学到的规则 F_phi 显著优于固定 Hebbian（post acc 0.827 vs 0.729） |
| Stage 3 | adaptive φ 提高稳定性，但“Λ_t 随人生上升”未获支持 |
| Stage 4 Formal | DLA 匹配调好参的 AdamW（B 域 gain +14% vs +13%），**慢记忆层零遗忘**（forgetting −0.4%） |
| Stage 5 | 难度归一化后 DLA 对递进课程不崩坏（LE 0.79→0.70→0.69） |
| Stage 5.5 | 9 维 tempo φ 不是 history effect 的 carrier |
| Stage 5.5b | History effect 可重复；meta φ 的发展收益不显著 |
| Stage 5.5c/d | Body 分解：W_fast 是主要 carrier；φ/P/Q/W_slow 贡献小 |
| Stage 5.5e P0 | **n=12：HE body 换入 EH W_fast 会显著破坏未来学习**（d≈−0.80，p=0.019） |
| Stage 6 | 同难度 4 任务纵向：弱方向（7/10），不显著 |
| Stage 7 | 跨领域 10 任务、20 seeds：**B 不成立**（回归斜率≈0，p=0.82） |
| Stage 8 | 物理近迁移 20 seeds：**无正向效应**（d=−0.17） |
| Effective Rank | EH/HE 的 W_fast 有效秩无显著差异；MLP 层最接近显著（p=0.064） |

## 实验假设（与三阶段对应）

1. **Stage 1**：可塑性有效 —— 同样的一生，DLA（自适应可塑性）比静态 MLP（SGD
   在模型外部）学得更快、忘得更少；`StaticMLP(meta init)` 消融把“好的初始权重”
   与“发育式可塑性”分开。
2. **Stage 2**：学习规则有效 —— 固定 Hebbian（同样做元训练，只优化 `W_slow` 初值）
   vs 学到的 `F_phi`。
3. **Stage 3**：学习能力可发育 —— φ 冻结 vs φ 在生命周期内由自身经历继续更新，
   比较 `Λ_t` 随任务的斜率。
4. **DNA 实验**：同一 `F_phi`，不同 DNA 先验（A 高可塑 / B 高稳定 / C 高新奇 /
   D 保守）在同一段人生中走出不同的可塑性、知识积累与遗忘轨迹。

## Stage 4：Backbone + Developmental Learning（Transformer 迁移）

原则：**不改 nanoGPT 骨架**。每个 `nn.Linear` / `nn.Embedding` 仍执行原计算，
只是有效权重变为

```
W_eff = W_slow + softplus(P) * W_fast
```

每个权重矩阵维护 `W_fast / P / Q / m / v`（Adam 力矩）。一次醒态更新：

```
g      = dL/dW_eff                    （标准反传教学基，全局 clip）
adam   = Adam(g; m, v)
dW_fast = -eta_fast * softplus(P) * adam - fast_decay * W_fast
dP      = eta_plast * progress * relevance(g) - stability * (P - P0)
dQ      = alpha_q * dW_fast * max(progress, 0)
```

睡眠时 `W_slow += beta * Q`、衰减 `W_fast/Q`、清空力矩——知识与技能在同一个
Transformer 权重上分居“慢/快”两层。实验脚本 `stage4_transformer_pilot.py` 用
6.59M 中文 nanoGPT 跑一生：**维基(A) → SFT 问答(B) → 维基重学(A)**，对比
普通 AdamW 微调，指标全部相对个体自身起点：

* adaptation gain：当前域 PPL 相对下降
* forgetting：学完 B 后 A 的 PPL 相对上升
* relearning：重学 A 后恢复的比例
* plasticity trajectory：各权重矩阵 softplus(P) 的发育轨迹

> Stage 4 用 per-parameter 规则而非 per-connection F_phi：对 6.59M 个连接逐一
> 跑规则网络在 CPU 上不可行，且 Stage 4 要验证的是“发育动力学”在真实
> Transformer 上的迁移，不是规则网络本身。

## 边界（诚实声明）

- 小规模机制研究：Neural Core 是 6.59M 中文 nanoGPT 或 MLP；不是大模型。
- **A 命题**（Learning history shapes the future learner）已有较强证据，
  特别是 W_fast 的“破坏性因果效应”（n=12，p=0.019）。
- **B 命题**（越学越快 / 学习效率随经历上升）目前**未获支持**：
  Stage 5 递进课程、Stage 6 同难度纵向、Stage 7 跨领域 10 任务、Stage 8
  物理近迁移均未出现显著正向趋势。
- φ（9 个 tempo）不是 history effect 的稳定 carrier；有效秩统计也不能解释
  W_fast 效应。
- 睡眠巩固写入 W_slow 的量仍很小；Fast/Slow 分离目前主要靠隔离而非强巩固。
- `Λ_t` 是行为操作化（样本效率倒数），不是独立理论定义。

## 与已有工作的关系

可学习突触可塑性规则、neuromodulation、ANML 式选择性可塑性和
stability–plasticity 都是已有方向；本项目的增量问题不是“再做一个 fast weight”，
而是：

> **学到的学习规则能否在个体的持续生命周期中继续发育？**

Stage 3 的 frozen vs adaptive φ 对比就是对这个问题的第一个可证伪实验。
