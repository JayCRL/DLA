#!/bin/bash
# 并行队列：只跑阶段一B（λ 扫描）——回答核心问题 Q3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
cd /root/dla-v0.2
OUT=/root/autodl-tmp/dla_leakgain
mkdir -p $OUT
C="--model gpt2 --variants direct,shufwrite,nocons --stages 3 --chunk 60000 \
 --d-train 60000 --d-eval 20000 --history-steps 40 --probe-steps 40 \
 --eval-batches 8 --snap-every 10 --state-dtype bf16"
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11; do
  f=$OUT/leakgainB_gpt2_s$SEED.json
  [ -f "$f" ] && { echo "SKIP B s$SEED"; continue; }
  echo "RUN  B s$SEED $(date +%H:%M:%S)"
  $PY analysis/wfast_geom/leak_gain.py $C \
    --fast-decays 0.02 --lambdas 0.5,0.75,1.0,1.05,1.1,1.2 \
    --seed $SEED --out $OUT 2>&1 | grep --line-buffered -E "^\[lg\]|Error|Traceback|out of memory"
  if [ -f $OUT/leakgain_gpt2_s$SEED.json ]; then
    mv $OUT/leakgain_gpt2_s$SEED.json $f; echo "DONE B s$SEED $(date +%H:%M:%S)"
  else echo "** FAILED B s$SEED"; fi
done
echo LEAKQB_DONE
