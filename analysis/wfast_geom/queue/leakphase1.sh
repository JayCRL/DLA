#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
cd /root/dla-v0.2
OUT=/root/autodl-tmp/dla_leakgain
mkdir -p $OUT
COMMON="--model gpt2 --variants direct,shufwrite,nocons --stages 3 --chunk 60000 \
 --d-train 60000 --d-eval 20000 --history-steps 40 --probe-steps 40 \
 --eval-batches 8 --snap-every 10 --state-dtype bf16"

run() {  # run <tag> <extra args...>
  local tag=$1; shift
  local f=$OUT/leakgain$tag\_gpt2_s$SEED.json
  if [ -f "$f" ]; then echo "SKIP $tag seed $SEED"; return; fi
  echo "RUN  $tag seed $SEED  $(date +%H:%M:%S)"
  # keep the tail of the real output: a bare grep for the [lg] tag hid every
  # traceback and made a total failure look like a silent no-op.
  $PY analysis/wfast_geom/leak_gain.py $COMMON "$@" --seed $SEED --out $OUT 2>&1 \
    | grep -E "^\[lg\]|Error|Traceback|error:|Killed|CUDA out of memory"
  if [ -f $OUT/leakgain_gpt2_s$SEED.json ]; then
    mv $OUT/leakgain_gpt2_s$SEED.json $f
    echo "DONE $tag seed $SEED  $(date +%H:%M:%S)"
  else
    echo "** FAILED $tag seed $SEED  (no output file)"
  fi
}

for SEED in 0 1 2 3 4 5 6 7 8 9 10 11; do
  run A --fast-decays 0.001,0.005,0.01,0.02 --lambdas 1.0
  run B --fast-decays 0.02 --lambdas 0.5,0.75,1.0,1.05,1.1,1.2
done
echo PHASE1_DONE
