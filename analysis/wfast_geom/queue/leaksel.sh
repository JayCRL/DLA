#!/bin/bash
# 决定性实验：W_fast 当 sleep 筛选器，λ 能否放大它？
#
#   selwrite : W_slow += γ · ŝ(|W_fast|) ⊙ Q     ← W_fast 的涌现结构当筛子
#   unifq    : W_slow += γ · Q                   ← 无筛选（能量匹配）
#   shufsel  : W_slow += γ · ŝ(perm|W_fast|) ⊙ Q ← 筛子打乱（能量匹配）
#   direct   : W_slow += γ · W_fast              ← 原规则（W_fast 当内容）
#
# 三者能量严格归一化到同一 norm，任何差异只能来自"往哪些坐标写"。
# 主对照 B = ppl(unifq) − ppl(selwrite)：筛选带来的因果优势。
# 放大判据：B 是否随 λ 上升。
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
cd /root/dla-v0.2
OUT=/root/autodl-tmp/dla_leaksel
mkdir -p $OUT
for SEED in 0 1 2 3 4 5; do
  f=$OUT/leaksel_gpt2_s$SEED.json
  [ -f "$f" ] && { echo "SKIP sel s$SEED"; continue; }
  echo "RUN  sel s$SEED $(date +%H:%M:%S)"
  $PY analysis/wfast_geom/leak_gain.py --model gpt2 --seed $SEED \
    --variants selwrite,unifq,shufsel,direct --stages 3 --chunk 60000 \
    --d-train 60000 --d-eval 20000 --history-steps 40 --probe-steps 40 \
    --eval-batches 8 --snap-every 10 --state-dtype bf16 \
    --fast-decays 0.001,0.005,0.01,0.02 --lambdas 0.25,0.5,1.0,2.0 \
    --out $OUT 2>&1 | grep --line-buffered -E "^\[lg\]|Error|Traceback|out of memory"
  [ -f $OUT/leakgain_gpt2_s$SEED.json ] && mv $OUT/leakgain_gpt2_s$SEED.json $f
  [ -f "$f" ] && echo "DONE sel s$SEED $(date +%H:%M:%S)" || echo "** FAILED sel s$SEED"
done
echo LEAKSEL_DONE
