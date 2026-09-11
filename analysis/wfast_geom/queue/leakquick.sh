#!/bin/bash
# 线程限制：不限的话 torch 会和同机其它任务互相踩踏（load 29 → GPU 饿死到 5%）
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
cd /root/dla-v0.2
OUT=/root/autodl-tmp/dla_leakquick
mkdir -p $OUT
for SEED in 0 1 2 3; do
  f=$OUT/quick_gpt2_s$SEED.json
  [ -f "$f" ] && { echo "SKIP quick s$SEED"; continue; }
  echo "RUN  quick s$SEED $(date +%H:%M:%S)"
  $PY analysis/wfast_geom/leak_gain.py --model gpt2 --seed $SEED \
    --variants direct,shufwrite,nocons --stages 3 --chunk 60000 \
    --d-train 60000 --d-eval 20000 --history-steps 20 --probe-steps 20 \
    --eval-batches 4 --snap-every 10 --state-dtype bf16 \
    --fast-decays 0.02 --lambdas 0.5,0.75,1.0,1.05,1.1,1.2 \
    --out $OUT 2>&1 | grep --line-buffered -E "^\[lg\]|Error|Traceback|out of memory"
  [ -f $OUT/leakgain_gpt2_s$SEED.json ] && mv $OUT/leakgain_gpt2_s$SEED.json $f
  [ -f "$f" ] && echo "DONE quick s$SEED $(date +%H:%M:%S)" || echo "** FAILED quick s$SEED"
done
echo LEAKQUICK_DONE
