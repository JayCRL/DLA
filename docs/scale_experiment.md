# DLA 规模检验 — AutoDL 部署指南

在 AutoDL 租一张 RTX 4090，把 DLA 的 HE/EH 历史对照实验跑在 GPT-2 774M backbone 上。

## 0. 为什么是 774M 而不是 1.5B

DLA 的状态（`w_fast`/`p`/`q`/`m`/`v`，每个都与 backbone 权重同形状）
在 fp32 下是 backbone 显存开销的 5 倍。实测预算（含 backbone 梯度与激活）：

| backbone | fp32 状态 | bf16 状态 | 4090 (24G) |
|---|---|---|---|
| gpt2 (124M) | 3.3 G | 2.0 G | ✅ 很宽裕 |
| gpt2-medium (355M) | 8.4 G | 7.1 G | ✅ |
| **gpt2-large (774M)** | **15.5 G** | **12.6 G** | ✅ **推荐** |
| gpt2-xl (1.5B) | 28.6 G | 22.8 G | ⚠️ 顶格，建议 A100 40G |

`--state-dtype bf16` 是上表右列的前提，务必带上。

## 1. 本地准备（Mac，不花钱）

```bash
# 权重（走 hf-mirror，实测比代理快 12 倍）
mkdir -p ~/llm-lab/hf_gpt2/gpt2-large
cd ~/llm-lab/hf_gpt2/gpt2-large
for f in config.json model.safetensors; do
  curl -sSL -O "https://hf-mirror.com/openai-community/gpt2-large/resolve/main/$f"
done
```

BPE 语料（tiktoken 编码，不能用字符级 .txt）：

```bash
python - <<'PY'
import os, numpy as np
os.environ['TIKTOKEN_CACHE_DIR'] = os.path.expanduser('~/llm-lab/tiktoken_cache')
import tiktoken
enc = tiktoken.get_encoding("gpt2")
text = open(os.path.expanduser('~/llm-lab/nanoGPT/data/shakespeare_char/input.txt'),
            encoding='utf-8').read()
np.save(os.path.expanduser('~/llm-lab/hf_gpt2/shakespeare_bpe.npy'),
        np.array(enc.encode(text), dtype=np.uint16))
PY
```

## 2. 上传到 AutoDL

### 先看这个：权重不用上传

AutoDL 提供[学术资源加速](https://www.autodl.com/docs/network_turbo/)，
实例内可以直接高速访问 huggingface.co，**3GB 权重无需从 Mac 上传**
（上传要 30-40 分钟且全程计费；云上下载通常几分钟）。

实例内执行：

```bash
source /etc/network_turbo

export HF_HOME=/root/autodl-tmp/cache/
python - <<'PY'
import os
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
from transformers import GPT2LMHeadModel
GPT2LMHeadModel.from_pretrained('openai-community/gpt2-large')   # 下载并缓存
print("gpt2-large ready")
PY
```

若加速后出现 SSL 证书错误，改用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from transformers import GPT2LMHeadModel
GPT2LMHeadModel.from_pretrained('openai-community/gpt2-large')"
```

BPE 语料同理（tiktoken 会自行缓存词表）：

```bash
python - <<'PY'
import os, numpy as np
os.environ['TIKTOKEN_CACHE_DIR'] = '/root/autodl-tmp/tiktoken_cache'
import tiktoken, urllib.request
enc = tiktoken.get_encoding("gpt2")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
txt = urllib.request.urlopen(url, timeout=60).read().decode('utf-8')
np.save('/root/hf_gpt2/shakespeare_bpe.npy', np.array(enc.encode(txt), dtype=np.uint16))
print("corpus ready:", len(enc.encode(txt)), "tokens")
PY
```

### 代码仍需要上传（很小，几 MB）

```bash
# 在 Mac 上，从 AutoDL 实例页复制 ssh 地址与端口
rsync -avP --exclude 'results*' --exclude '.git' --exclude 'data/domains' \
  ~/Desktop/dla-v0.2/ root@<host>:-p<port> /root/dla-v0.2/
rsync -avP ~/llm-lab/nanoGPT/model.py root@<host>:-p<port> /root/nanoGPT/model.py
```

（若 rsync 不便，用 AutoDL 的 JupyterLab 拖拽上传也行，只有几 MB。）

### 注意路径

`audit_scale.py` 默认从 `~/llm-lab/nanoGPT` 和 `~/llm-lab/hf_gpt2` 找文件。
云端目录不同时，用参数覆盖：

```bash
python analysis/wfast_geom/audit_scale.py \
  --model gpt2-large --seed 0 \
  --corpus /root/hf_gpt2/shakespeare_bpe.npy \
  --state-dtype bf16 --device cuda --out /root/dla_scale
```

并确保 `~/llm-lab/nanoGPT/model.py` 存在（或把 `NANO` 路径改成实际位置）。

## 3. 实例环境

选镜像：**PyTorch 2.x + CUDA 12.x**。然后：

```bash
pip install transformers tiktoken safetensors
```

## 4. 冒烟测试（约 3 分钟，先确认再烧钱）

```bash
cd /root/dla-v0.2
python analysis/wfast_geom/audit_scale.py \
  --model gpt2-large --seed 0 --smoke \
  --state-dtype bf16 --device cuda \
  --out /root/dla_scale/smoke
```

看到 `Delta(EH-HE)` 是一个**非零**数就说明链路正常。
（若 Δ 恰好为 0，说明评估时状态被摘掉了 —— 那是已知 bug 的征兆。）

## 4.5 测速（关键：先量再跑，别赌）

这一步花约 ¥0.1，却能避免跑一半才发现预算失控。

```bash
python analysis/wfast_geom/bench_scale.py \
  --model gpt2-large --state-dtype bf16 --device cuda \
  --seeds 4 --rate 1.88
```

输出示例：

```
  wake step: median 850.0 ms  (min 840.1, max 902.3)
  eval pass (2 batches): median 210.0 ms
  peak VRAM: 12.60G  (reserved 13.10G)

  per seed :   22.4 min
  4 seeds  :   89.6 min  = 1.49 h
  at ¥1.88/h : ¥2.81
  + setup/upload/debug margin (×1.8): ¥5.05
```

**拿到 `per seed` 的真实数字后再决定跑几个 seed**，并把 `--seeds` 改成目标值重估。
注意区分「CPU 测出的速度」和「GPU 测出的速度」——只有后者可用于算钱。

## 5. 正式实验

单个 seed：

```bash
python analysis/wfast_geom/audit_scale.py \
  --model gpt2-large --seed 0 \
  --state-dtype bf16 --device cuda \
  --history-steps 40 --probe-steps 40 \
  --block 128 --batch 4 --chunk 4000 \
  --out /root/dla_scale
```

多 seed（4 张卡并行或串行；单卡建议串行避免争抢）：

```bash
for s in 0 1 2 3; do
  python analysis/wfast_geom/audit_scale.py \
    --model gpt2-large --seed $s \
    --state-dtype bf16 --device cuda \
    --out /root/dla_scale \
    2>&1 | tee /root/dla_scale/log_s$s.txt
done

# 汇总
python - <<'PY'
import json, glob, statistics as st
rows = [json.load(open(p)) for p in sorted(glob.glob('/root/dla_scale/scale_gpt2-large_s*.json'))]
d = [r['delta'] for r in rows]
print(f"n={len(d)}  mean Δ(EH-HE) = {st.mean(d):+.4f}  sd = {st.stdev(d):.4f}" if len(d)>1
      else f"n=1  Δ = {d[0]:+.4f}")
for r in rows:
    print(f"  seed {r['seed']}: Δ={r['delta']:+.4f}  "
          f"EH gain={r['arms']['EH']['gain40']:+.4f}  HE gain={r['arms']['HE']['gain40']:+.4f}")
PY
```

## 6. 成本

AutoDL 4090 约 **¥1.88/小时**，按秒计费。

| 阶段 | 时长 | 费用 |
|---|---|---|
| 环境 + 冒烟 | 0.5 h | ¥0.94 |
| 4 seed 正式跑 | 2–4 h | ¥4–8 |
| 8 seed 扩展 | 4–8 h | ¥8–15 |
| **合计（含调试余量）** | ~6 h | **约 ¥12** |

**跑完立刻关机**，AutoDL 关机后只收存储费（几毛钱/天）。

## 7. 已知坑（都已在代码里修掉，列出来便于排查）

| 症状 | 原因 | 处理 |
|---|---|---|
| `size mismatch ... (3072,768) vs (768,3072)` | HF Conv1D 权重是 (in,out)，nanoGPT Linear 是 (out,in) | 脚本自动转置 |
| `does not require grad` | HF 权重默认 frozen | 脚本 `requires_grad_(True)` |
| `does not require grad`（探针步） | 训练步被 `@torch.no_grad()` 包住 | 探针循环不加装饰器 |
| **两个 arm 结果完全相同** | 评估时 `set_dla_state(None)` 摘掉了 DLA | 评估必须保持状态附着 |
| PPL 上百万 | 用字符级 `.txt` 喂 BPE 模型 | 用 `.npy` token 语料 |
| `hash(tag)` 跨进程不稳定 | Python hash 加盐 | 改用固定 `chunk_seed` |

## 8. 结果怎么进论文

`scale_gpt2-large_s*.json` 里的 `delta` 就是跨规模的载体效应估计，
与 6.59M / 10.65M 的 `Δ gain@40` 直接可比（`audit_scale.py` 沿用
`audit_second.run_d_probe` 的**相对增益**定义 `(pre − final)/pre`）。

报 `mean ± sd`、逐 seed、以及 n；不要只报 p 值。
