#!/bin/bash
# CL benchmark 上的 (fd, lambda) DLA 变体 —— 只跑新增的，baseline 复用已有 10 seeds
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
cd /root/dla-v0.2
OUT=/root/autodl-tmp/dla_cl_gain
mkdir -p $OUT
M="dla@fd=0.001,lam=1.0,dla@fd=0.005,lam=1.0,dla@fd=0.01,lam=1.0,dla@fd=0.02,lam=1.0"
M="$M,dla@fd=0.02,lam=0.5,dla@fd=0.02,lam=0.75,dla@fd=0.02,lam=1.05,dla@fd=0.02,lam=1.1,dla@fd=0.02,lam=1.2"
M="$M,adamw@lr=1e-5"
for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
  f=$OUT/cl_gpt2_s$s.json
  [ -f "$f" ] && { echo "SKIP cl_gain s$s"; continue; }
  echo "RUN  cl_gain s$s $(date +%H:%M:%S)"
  $PY analysis/wfast_geom/cl_baselines.py --model gpt2 --seed $s --methods "$M" \
    --steps 60 --eval-batches 8 --out $OUT 2>&1 \
    | grep -E "^\[cl\] ===|Error|Traceback|out of memory|INERT"
  [ -f "$f" ] && echo "DONE cl_gain s$s $(date +%H:%M:%S)" || echo "** FAILED cl_gain s$s"
done
echo CLGAIN_DONE
