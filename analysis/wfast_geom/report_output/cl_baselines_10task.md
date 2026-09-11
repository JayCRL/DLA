# 多算法抗遗忘对比（10 任务 CL 基准，n=10）

> **为什么会有这个文件**：`results/cloud/dla_cl/gpt2/`（10 seeds × 10 methods）是仓库里**唯一**一组
> "DLA 与多个持续学习算法同协议对比抗遗忘"的数据，但**此前没有任何报告记录它**——数字只躺在 JSON 里，
> 论文正文也完全没有引用。本文件把它落成可引用的结果，并把解读上必须说清的前提写明。

---

## 1. 运行与有效性

| | |
|---|---|
| 数据 | `results/cloud/dla_cl/gpt2/cl_gpt2_s{0..9}.json` |
| 规模 | 10 seeds × 10 methods |
| 骨干 | GPT-2 124M |
| 基准 | 10 任务序列（60 steps/task） |
| 方法 | `dla`、`replay`、`ewc@λ∈{1e4,1e5,1e6}`、`si@λ∈{1e4,1e6}`、`adamw@lr∈{3e-5,1e-4,3e-4}` |
| 脚本 | `analysis/wfast_geom/cl_baselines.py`（启动记录见 `analysis/wfast_geom/queue/run_job.sh` 的 `cl)` 分支） |

**有效性检查（这一批是修后数据）**：EWC/SI 的惩罚项必须真的在计算图里。`cl_baselines.py` 把
"惩罚项惰性"当作**硬失败**来检（`penalty_ratio_max` 低于阈值即报 `** INERT **`）。实测各 EWC/SI 臂的
penalty/loss 最大值分别为 3.86e-1（si@1e4）、3.06e+1（si@1e6）、2.93e-1（ewc@1e6）、3.08e-2（ewc@1e4）、
1.67e-1（ewc@1e5）——**全部远高于惰性阈值，惩罚项确实生效**。

这也正是 `ad97334` 修的那个 bug：修前 EWC/SI 的惩罚项脱离计算图，退化成普通 AdamW 的比特级副本，
所以**修前的一切 EWC/SI 数字都作废**（见 `paper/validation_audit.md` 的 VOID 块）。

指标定义（`cl_baselines.py`）：

```
forward_mean     : 各任务上学习该任务的相对增益（越高越会学）
retention_mean   : mean(after / end)，即"刚学完时"与"整个序列结束时"的困惑度比（越高越不遗忘）
forgetting_mean  : mean((end − after) / after)（越低越好）
```

---

## 2. 结果（n=10，按 retention 降序）

| method | forward ↑ | retention ↑ | forgetting ↓ |
|---|---|---|---|
| si@λ1e4 | +0.0840 ± 0.0056 | **1.0435 ± 0.007** | −0.0382 ± 0.0056 |
| si@λ1e6 | +0.0407 ± 0.0057 | 0.9937 ± 0.003 | +0.0073 ± 0.0036 |
| ewc@λ1e6 | +0.1588 ± 0.0119 | 0.9866 ± 0.035 | +0.0199 ± 0.0346 |
| adamw@lr3e-5 | +0.1616 ± 0.0065 | 0.9362 ± 0.006 | +0.0716 ± 0.0071 |
| adamw@lr1e-4 | +0.1967 ± 0.0131 | 0.9089 ± 0.022 | +0.1070 ± 0.0284 |
| ewc@λ1e4 | +0.1976 ± 0.0150 | 0.9045 ± 0.017 | +0.1115 ± 0.0227 |
| ewc@λ1e5 | +0.1860 ± 0.0125 | 0.8971 ± 0.023 | +0.1199 ± 0.0301 |
| **dla** | **+0.1768 ± 0.0109** | **0.8966 ± 0.019** | **+0.1219 ± 0.0248** |
| adamw@lr3e-4 | +0.1830 ± 0.0210 | 0.8632 ± 0.038 | +0.1756 ± 0.0536 |
| replay | +0.2257 ± 0.0188 | 0.8193 ± 0.077 | +0.2513 ± 0.1247 |

---

## 3. 怎么读（以及不能怎么读）

### 3.1 这是一条 stability–plasticity 前沿，不是一张排行榜

表里存在明显的单调权衡：**SI@λ1e4 拿到最高 retention（1.0435）却是最差的 forward（+0.0840，
几乎不学）；replay 拿到最高 forward（+0.2257）却是最差 retention（0.8193）**。

`cl_baselines.py` 自己就写明了对标方式（第 160–162 行）：

> "Fine-tuning baselines need their own lr sweep: their Pareto point is set by lr, and comparing
> DLA's single operating point against one arbitrary lr would be an unfair baseline.
> **The comparison that matters is FRONTS.**"

**DLA 落在前沿中部**：retention 0.8966（10 个里的第 8），forward +0.1768（第 5）。
与 `ewc@λ1e4`（forward +0.1976，retention 0.9045）最接近——forward 与 retention **各略低一点**。
它比 `replay` 与 `adamw@3e-4` 更不遗忘，比 `ewc@1e6` 与两个 SI 更会学。

### 3.2 ⚠️ 最重要的前提：**DLA 冻结骨干，基线是全量微调**

这一点决定了这张表**不是同预算的方法对比**：

```python
# cl_baselines.py
# DLA needs backbone grads ... but must NOT step the backbone itself.
self.state = model.make_state(device) if method == "dla" else None
self.opt = (torch.optim.AdamW(self.params, ...) if method != "dla" else None)
#                              ^ self.params = 全部非 tempos 参数 = 整个骨干
```

- `dla` 臂：**骨干冻结**，只更新 `W_fast` 与 `P`（外加 tempos）；
- `adamw` / `ewc` / `si` / `replay` 臂：**全量微调**整个骨干。

所以"DLA 的 retention 排第 8"**不能**读成"DLA 作为持续学习算法弱于这些方法"——它们是不同的
方法类别（参数高效式 vs 全量微调）。可以读成：**在"不微调骨干"这个约束下，DLA 的抗遗忘
落在这条前沿的什么位置**；而要把它说成性能对标，必须先做**容量匹配**的对照
（例如把基线也限制到同等可训练参数量）。

### 3.3 DLA 自己的旋钮也已被扫过，但只有 n=2

`results/cloud/dla_cl_gain/`（2 seeds，**初步**）扫了 DLA 自身的 leak/gain：

| arm | forward | retention |
|---|---|---|
| `dla@g1.1,fd0.02` | +0.1946 | 0.9035 |
| `dla@g1.2,fd0.02` | +0.2007 | 0.8836 |
| `dla@g1,fd0.001` | +0.1808 | 0.8957 |
| `dla@g1,fd0.005` | +0.1801 | 0.8944 |
| `dla@g1,fd0.02` | +0.1704 | 0.8806 |
| `dla@g0.75,fd0.02` | +0.1870 | 0.8708 |
| `adamw@lr1e-5` | +0.1246 | 1.0057 |

即 **DLA 也有自己的前沿**：提高 gain 买到 forward（+0.20）、付出 retention（0.884）。
在 forward 匹配到 `ewc@λ1e4`（+0.1976）的位置，DLA 的 retention 为 0.9035–0.9045——**与它持平**。

**但 n=2，不能作为结论**；它只是说明"DLA 单点 vs 基线扫描"这一对比不对称，需要扩到 n≥10
才能把 DLA 也画成一条前沿。

---

## 4. 给论文的用法（定位约束）

这组数据适合写进**"定位与边界"**一节，而不适合写成"我们与 SOTA 对比"：

- ✅ 可写：在同协议、同 backbone、多算法（3 个 EWC 强度、2 个 SI 强度、3 个 lr、1 个 replay）的
  对比中，DLA 位于 stability–plasticity 前沿内部，而非落后一个身位。
- ✅ 可写：显式惩罚/回放方法的抗遗忘点是**由其超参选出来的**（SI@1e4 几乎不学才换来 1.04 的 retention），
  因此单点比较没有意义——这也是本项目主张"看前沿"的理由。
- ⚠️ 必须写：DLA 在此表中**冻结骨干**，基线为**全量微调**；非同预算。
- ⚠️ 必须写：DLA 在该表中是**单一默认配置**，其自身旋钮的扫描只有 n=2（初步）。
- ❌ 不要写：DLA 优于/劣于 EWC、SI、replay 这类整体判断。
- ❌ 不要与 `paper/validation_audit.md` §8/§9 的 EH/HE 协议数字混列——**不是同一协议**，
  且那张表已作废。

---

## 5. 复算

```bash
cd ~/Desktop/dla-v0.2
python - <<'PY'
import json, glob, statistics as st
agg = {}
for f in sorted(glob.glob("results/cloud/dla_cl/gpt2/cl_gpt2_s*.json")):
    for m, v in json.load(open(f))["methods"].items():
        agg.setdefault(m, []).append(v)
for m in sorted(agg, key=lambda k: -st.mean(x["retention_mean"] for x in agg[k])):
    a = agg[m]
    print(f"{m:16s} forward={st.mean(x['forward_mean'] for x in a):+.4f} "
          f"retention={st.mean(x['retention_mean'] for x in a):.4f} "
          f"forgetting={st.mean(x['forgetting_mean'] for x in a):+.4f}")
PY
```
