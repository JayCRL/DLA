# Scale probe — post-fix re-run (124M / 355M / 774M)

> **为什么会有这个文件**：`docs/scale_results.md` 顶部那张"INVALID — SUPERSEDED"横幅写着
> "Re-run after commit `d1afaf2` is in progress; see `docs/scale_results_fixed.md`"——
> 但那个文件**一直没有被写出来**。数据其实早就跑完了（`results/cloud/dla_scale/fixed/`），
> 只是没人把数字落成报告。本文件补上这一环。

---

## 1. 坏在哪里，以及"修好了"是怎么验的

**故障**：`GPTConfig` 默认 `vocab_size=50304`，loader 把 50257..50303 行补零；由于 `lm_head`
与 `wte` 权重共享，这些零行对应的 47 个 token 的 logit 恰好为 0，而 GPT-2 的真实 logit 带有
很大的**负偏置**（本语料上最大约 −77）。于是补零 token 在排序上压过一切真实 token，模型在每个
位置都预测 index 50257。

**修复**：`d1afaf2` 起使用真实词表 `50257`；`analysis/wfast_geom/audit_scale.py` 现在会**拒绝**
补零并直接报错（见该文件 `refusing to zero-pad the vocabulary ...`）。

**修复的验证（用数据本身，不是靠声明）**：

| | birth_ppl（出生困惑度） |
|---|---|
| 坏 build（`results/scale_cloud/capmatch/`） | 124M: 6.6e5–1.3e6 ｜ 355M: **8.0e32** ｜ 774M: 5.4e3–1.0e4 ｜ 1.5B: 1.0e3–1.4e3 |
| 修后（`results/cloud/dla_scale/fixed/`） | 124M: 81–153 ｜ 355M: 51–103 ｜ 774M: **48–107** |

**权重加载从来不是问题**：148/148 个 key 与 HF 完全一致。之前的 Mac-CPU / 云-GPU parity 检验
在小数点后五位一致，是因为**两边跑的是同一个坏模型**——parity 证明的是可复现性，**永远不是正确性**。
真正抓到它的是 HF 与加载后模型的困惑度对比。

---

## 2. 修后结果

对比量定义（`analysis/wfast_geom/audit_scale.py`）：

```
delta        = gain40(EH) − gain40(HE)                    # 负 = HE 适应更好
delta_wfast  = 同一对比，但从只携带该历史 W_fast 的全新状态出发探测
```

| backbone | params | n | Δ(EH−HE) | 负号 | t | Δwfast | t |
|---|---|---|---|---|---|---|---|
| gpt2 | 124M | 12 | **−0.0390 ± 0.0240** | 12/12 | −5.64 | −0.0437 | −5.94 |
| gpt2-medium | 355M | 12 | **−0.0437 ± 0.0338** | 11/12 | −4.48 | −0.0466 | −4.64 |
| gpt2-large | **774M** | 12 | **−0.0331 ± 0.0417** | 9/12 | −2.75 | −0.0371 | −2.64 |

**读法**：

1. **方向在三个规模上都复现**（HE 历史优于 EH 历史），每个规模 $n{=}12$，各自显著。
2. **幅值在 6.2 倍参数范围内没有系统性趋势**（−0.039 / −0.044 / −0.033）——不是"越大越强"，
   也不是"越大越弱"，而是**跨规模稳定存在**。
3. **反号少数派随规模增多**（0/12 → 1/12 → 3/12），即效应真实但 774M 上更噪。
   报数时逐 seed 一起给，不要只给 p 值。

需要时按下面命令复算（数据已在仓库里，无需重跑）：

```bash
python - <<'PY'
import json, glob, statistics as st
for m in ["gpt2", "gpt2-medium", "gpt2-large"]:
    d = [json.load(open(f)) for f in sorted(glob.glob(f"results/cloud/dla_scale/fixed/{m}/*.json"))]
    ds = [x["delta"] for x in d]
    print(m, len(ds), f"{st.mean(ds):+.4f}", f"{st.stdev(ds):.4f}",
          f"t={st.mean(ds)/(st.stdev(ds)/len(ds)**.5):+.2f}")
PY
```

---

## 3. 坏 build 会让我们多报多少

被作废的 `capmatch` 表声称：

| backbone | 坏 build Δ | 修后 Δ | 夸大倍数 |
|---|---|---|---|
| 124M | −0.1411 (t=−13.37) | −0.0390 (t=−5.64) | **3.6×** |
| 355M | −0.1182 (t=−8.59) | −0.0437 (t=−4.48) | **2.7×** |
| 774M | −0.1662 (t=−5.37) | −0.0331 (t=−2.75) | **5.0×** |

引用旧表会把跨规模效应**夸大 3–5 倍**。这就是它必须留在 `scale_results.md` 里被标 INVALID、
而不能"顺手"流进正文的原因。

---

## 4. 1.5B **没有**重跑（这是当前规模的硬边界）

- `results/cloud/dla_scale/fixed/gpt2-xl/` 目录**存在但是空的** → 修后的 1.5B 运行没有产出任何结果。
- 仓库里唯一的 1.5B 数字（Δ = −0.1155 ± 0.0484，$n{=}3$，3/3 为负）来自**坏 build 的 capmatch 运行**，
  属于第 3 节那张作废表 → **不可引用**。
- 因此当前可主张的规模上界是 **774M**。

> 重跑 1.5B 的工程要点已记录在 `docs/scale_results.md` §Engineering notes：1.5B 的 DLA 状态
> 14.5 GB + backbone 5.8 GB，`set_dla_state` 会保持参数存活，先分配第二个历史状态会瞬时冲到
> 46.4 GB 并打爆 48 GB 卡；在每次 `make_state` 前释放模型的状态引用可把峰值压到 33 GB 左右。

---

## 5. 两套修后探测**不可混用**

仓库里有两个**都属修后**、但**设计不同**的规模探测：

| | 脚本 | 臂 | 布局 | 对比 |
|---|---|---|---|---|
| 本文件 | `audit_scale.py` | `EH` / `HE` / `EH_wfast` / `HE_wfast` | 各臂有自己的收尾块 | `delta = gain40(EH) − gain40(HE)` |
| v2 | `audit_scale_v2.py` | `ABD` / `BAD`（+`AAD`/`BBD` 对照） | `common-final-chunk`（所有臂收在同一块上，消除起点混淆） | `delta_gain = gain40(BAD) − gain40(ABD)` |

`results/cloud/dla_scale_v2/` 给出的 `delta_gain` 为 **+0.0081 / +0.0029 / +0.0162**
（124M / 355M / 774M，$t{=}+1.84 / +0.70 / +2.04$）。

**注意**：这是**另一个对比**（前缀顺序 A→B→D vs B→A→D，且收尾块相同），
**不能**与第 2 节的 `delta` 放进同一张表做趋势比较；而且它自己的效应量偏小、$t$ 偏弱。
要写进论文就必须单独说明其设计，或干脆不写。

---

## 6. 溯源性缺口（如实标注）

- **本组运行的确切启动命令没有被版本化**。`analysis/wfast_geom/queue/run_job.sh` 里只有 `v2` 的
  启动记录，`fixed/` 那次没有对应脚本进仓库。
- 每 seed 的 JSON **不记录 `chunk`**（只记 `model`/`params`/`state_dtype`/`easy`/`arms`/`delta`/`delta_wfast`），
  所以这次运行用的 chunk 大小**无法从仓库恢复**；`audit_scale.py` 的默认值是
  `block 128 / batch 4 / history 40 / probe 40 / chunk 4000`，而部署指南的示例用的是 4000 或 40000。
- 结论：**第 2 节的数字可用，但"用哪个 chunk 跑的"这一条目前不可考证**。补齐方式是把启动命令
  补进 `queue/`（与 v2 同等对待），或在下一次运行时把协议字段写进 JSON。
