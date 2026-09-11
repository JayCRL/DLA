#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/dla-v0.2
/root/miniconda3/bin/python analysis/wfast_geom/leak_gain.py \
  --model gpt2 --variants direct,shufwrite,nocons --stages 3 --chunk 60000 \
  --d-train 60000 --d-eval 20000 --history-steps 40 --probe-steps 40 \
  --eval-batches 8 --snap-every 10 --state-dtype bf16 \
  --fast-decays 0.001,0.005,0.01,0.02 --lambdas 1.0 \
  --seed 0 --out /root/autodl-tmp/lgtest2
echo "TEST2_DONE exit=$?"
