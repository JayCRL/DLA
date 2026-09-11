#!/bin/bash
# 二维 factorial sweep：分离 STRUCTURE (fast_decay) 与 STRENGTH (lambda)
#
# λ 网格从 {0.75,1.0,1.05,1.1} 改为 {0.25,0.5,0.75,1.0}：快速版显示 λ>1 段归一化
# 选择性已饱和（cv 只动 +0.76%）且写回优势翻负，继续扫那里是浪费算力。低 λ 端
# 才是信息量所在（B 从 +1.14 掉到 −1.14 的转折发生在 0.5~1.0 之间）。
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
cd /root/dla-v0.2
OUT=/root/autodl-tmp/dla_leak2d
mkdir -p $OUT
for SEED in 0 1 2 3 4 5; do
  f=$OUT/leak2d_gpt2_s$SEED.json
  [ -f "$f" ] && { echo "SKIP 2d s$SEED"; continue; }
  echo "RUN  2d s$SEED $(date +%H:%M:%S)"
  $PY analysis/wfast_geom/leak_gain.py --model gpt2 --seed $SEED \
    --variants direct,shufwrite,nocons --stages 3 --chunk 60000 \
    --d-train 60000 --d-eval 20000 --history-steps 40 --probe-steps 40 \
    --eval-batches 8 --snap-every 10 --state-dtype bf16 \
    --fast-decays 0.001,0.005,0.01,0.02 --lambdas 0.25,0.5,0.75,1.0 \
    --out $OUT 2>&1 | grep --line-buffered -E "^\[lg\]|Error|Traceback|out of memory"
  [ -f $OUT/leakgain_gpt2_s$SEED.json ] && mv $OUT/leakgain_gpt2_s$SEED.json $f
  [ -f "$f" ] && echo "DONE 2d s$SEED $(date +%H:%M:%S)" || echo "** FAILED 2d s$SEED"
done
echo LEAK2D_DONE
