#!/bin/bash
# 断点续跑：已完成 seed 直接跳过
# usage: run_job.sh <v2|alloc|cl> <model> <seeds...>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
cd /root/dla-v0.2
JOB=$1; M=$2; shift 2

case $JOB in
  # v2: 所有 arm 以同一块 D 收尾，消除 v1 的起点混淆
  v2)    BASE=/root/autodl-tmp/dla_scale_v2; PAT="v2_%s_s%s.json"
         CMD="$PY analysis/wfast_geom/audit_scale_v2.py --model $M --state-dtype bf16 --device cuda \
              --chunk 40000 --history-steps 40 --probe-steps 40 --block 128 --batch 4 \
              --d-train 60000 --d-eval 20000 --eval-batches 8";;
  alloc) BASE=/root/autodl-tmp/dla_alloc; PAT="alloc_%s_s%s.json"
         CMD="$PY analysis/wfast_geom/audit_alloc.py --model $M --variants direct,shufwrite,nocons \
              --stages 3 --chunk 60000 --d-train 60000 --d-eval 20000 \
              --history-steps 40 --probe-steps 40 --eval-batches 8";;
  cl)    BASE=/root/autodl-tmp/dla_cl; PAT="cl_%s_s%s.json"
         CMD="$PY analysis/wfast_geom/cl_baselines.py --model $M \
              --methods dla,replay,adamw@lr=3e-5,adamw@lr=1e-4,adamw@lr=3e-4,ewc@1e4,ewc@1e5,ewc@1e6,si@1e4,si@1e6 \
              --steps 60 --eval-batches 8";;
  *) echo "unknown job $JOB"; exit 1;;
esac

mkdir -p $BASE/$M
for s in "$@"; do
  JSON=$(printf "$BASE/$M/$PAT" "$M" "$s")
  if [ -f "$JSON" ]; then echo "SKIP $JOB $M seed $s (已完成)"; continue; fi
  echo "RUN  $JOB $M seed $s"
  $CMD --seed $s --out $BASE/$M 2>&1 | grep -E "^\[(v2|alloc|cl)\] (seed|==|arm)|OutOfMemory|Error|INERT|LEVEL"
done
echo "JOB_DONE $JOB $M"
