#!/bin/bash
# 分段队列：先跑 fd 扫描（回答 Q1/Q2），再跑 λ 扫描（回答 Q3），最后二维
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
cd /root/dla-v0.2
OUT=/root/autodl-tmp/dla_leakgain
mkdir -p $OUT
C="--model gpt2 --variants direct,shufwrite,nocons --stages 3 --chunk 60000 \
 --d-train 60000 --d-eval 20000 --history-steps 40 --probe-steps 40 \
 --eval-batches 8 --snap-every 10 --state-dtype bf16"

run() { local tag=$1; shift
  local f=$OUT/leakgain$tag\_gpt2_s$SEED.json
  [ -f "$f" ] && { echo "SKIP $tag s$SEED"; return; }
  echo "RUN  $tag s$SEED $(date +%H:%M:%S)"
  $PY analysis/wfast_geom/leak_gain.py $C "$@" --seed $SEED --out $OUT 2>&1 \
    | grep --line-buffered -E "^\[lg\]|Error|Traceback|error:|out of memory" 
  if [ -f $OUT/leakgain_gpt2_s$SEED.json ]; then
    mv $OUT/leakgain_gpt2_s$SEED.json $f; echo "DONE $tag s$SEED $(date +%H:%M:%S)"
  else echo "** FAILED $tag s$SEED"; fi
}

SEEDS="0 1 2 3 4 5 6 7 8 9 10 11"
echo "########## 阶段一A：fast_decay 扫描 (λ=1) ##########"
for SEED in $SEEDS; do run A --fast-decays 0.001,0.005,0.01,0.02 --lambdas 1.0; done
echo "########## 阶段一B：λ 扫描 (fd=0.02) ##########"
for SEED in $SEEDS; do run B --fast-decays 0.02 --lambdas 0.5,0.75,1.0,1.05,1.1,1.2; done
echo "########## 阶段二：二维 (4 fd × 4 λ)，6 seeds ##########"
for SEED in 0 1 2 3 4 5; do run C --fast-decays 0.001,0.005,0.01,0.02 --lambdas 0.75,1.0,1.05,1.1; done
echo LEAKQ_DONE
