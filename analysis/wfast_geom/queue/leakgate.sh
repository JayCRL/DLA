#!/bin/bash
# 决定性实验：门控写回能不能让 λ 放大选择性写回？
#
# 现在 sleep 写的是裸 W_fast，而前向用的是 softplus(P)*W_fast。λ 放大写回 = 放大
# "模型没有在表达的那一份"。本队列在同一配置下对比 门控/非门控 两条 B(λ) 曲线。
# λ 扩到 2.0：如果门控假设对，λ>1 段应该出现 B 上升而不是塌陷。
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
cd /root/dla-v0.2
OUT=/root/autodl-tmp/dla_leakgate
mkdir -p $OUT
LAMS=0.25,0.5,0.75,1.0,1.5,2.0
for SEED in 0 1 2 3; do
  for G in "" "--write-gate"; do
    tag=$([ -n "$G" ] && echo gate || echo nogate)
    f=$OUT/leakgate_${tag}_gpt2_s$SEED.json
    [ -f "$f" ] && { echo "SKIP $tag s$SEED"; continue; }
    echo "RUN  $tag s$SEED $(date +%H:%M:%S)"
    $PY analysis/wfast_geom/leak_gain.py --model gpt2 --seed $SEED $G \
      --variants direct,shufwrite,nocons --stages 3 --chunk 60000 \
      --d-train 60000 --d-eval 20000 --history-steps 40 --probe-steps 40 \
      --eval-batches 8 --snap-every 10 --state-dtype bf16 \
      --fast-decays 0.02 --lambdas $LAMS \
      --out $OUT 2>&1 | grep --line-buffered -E "^\[lg\]|Error|Traceback|out of memory"
    src=$OUT/leakgain$([ -n "$G" ] && echo _gate)_gpt2_s$SEED.json
    [ -f "$src" ] && mv "$src" "$f"
    [ -f "$f" ] && echo "DONE $tag s$SEED $(date +%H:%M:%S)" || echo "** FAILED $tag s$SEED"
  done
done
echo LEAKGATE_DONE
