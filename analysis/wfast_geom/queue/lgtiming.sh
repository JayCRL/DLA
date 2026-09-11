#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/dla-v0.2
P=/root/miniconda3/bin/python
C="--model gpt2 --variants direct --stages 3 --chunk 60000 --d-train 60000 \
 --d-eval 20000 --history-steps 40 --probe-steps 40 --state-dtype bf16 --seed 77"
echo "=== A: snap=0 (无快照), eval-batches=8 ==="
s=$(date +%s); $P analysis/wfast_geom/leak_gain.py $C --snap-every 0 --eval-batches 8 \
  --fast-decays 0.02 --lambdas 1.0 --out /root/autodl-tmp/lgt_a >/dev/null 2>&1
echo "  耗时 $(( $(date +%s) - s )) 秒"
echo "=== B: snap=10, eval-batches=8 ==="
s=$(date +%s); $P analysis/wfast_geom/leak_gain.py $C --snap-every 10 --eval-batches 8 \
  --fast-decays 0.02 --lambdas 1.0 --out /root/autodl-tmp/lgt_b >/dev/null 2>&1
echo "  耗时 $(( $(date +%s) - s )) 秒"
echo "=== C: snap=0, eval-batches=2 ==="
s=$(date +%s); $P analysis/wfast_geom/leak_gain.py $C --snap-every 0 --eval-batches 2 \
  --fast-decays 0.02 --lambdas 1.0 --out /root/autodl-tmp/lgt_c >/dev/null 2>&1
echo "  耗时 $(( $(date +%s) - s )) 秒"
echo TIMING_DONE
