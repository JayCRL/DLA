#!/bin/bash
# topk_signed sweep -- the sign-preserving, energy-matched top-k write.
#
# Why this run exists: `topwrite` was the only "concentration" arm, and it replaced the
# kept coordinates with a constant magnitude, destroying the sign pattern
# (docs/mechanism_audit.md 3.2). It therefore could not separate "the write is
# concentrated" from "the write lands on the aligned coordinates". topk_signed keeps the
# signs and rescales to exactly the `direct` write energy, so placement is the only
# variable left.
#
# Design: frac in {0.05, 0.2, 0.5, 1.0} x seeds 0-11, HE arm.
#   0.2   head-to-head with topwrite's 20% concentration, now signed
#   1.0   the self-check: must reproduce the archived `direct` arm, which also measures
#         this batch's harness noise floor
#   0.05 / 0.5  bracket it, so the verdict is a dose-response rather than one point
#
# Verdict the sweep is meant to settle:
#   topk_signed ~ direct      -> sparse aligned selection is sufficient
#   topk_signed ~ shufwrite   -> only the distributed aligned pattern works
#   in between / dose-dependent -> report the dose-response, claim neither label
#
# Launch. NOTE: the cloud queue scripts use `setsid nohup ...`; the Linux instance has
# setsid and macOS does not, and a run launched that way here silently never starts.
# The Mac equivalent is a nohup'd background subshell:
#   (nohup analysis/wfast_geom/queue/run_topk.sh > ~/llm-lab/dla_audit/topk_driver.log 2>&1 &)
set -u
OUT="$HOME/llm-lab/dla_audit"
J="$OUT/topkjobs.txt"
LOG="$OUT/topk_run.log"

{
  # frac=1.0 first: it is the self-check, so a broken implementation or environment
  # shows up in the first few minutes instead of at the end of a 90-minute sweep.
  for f in 1.0 0.2 0.5 0.05; do
    for s in 0 1 2 3 4 5 6 7 8 9 10 11; do printf '%s|%s\n' "$f" "$s"; done
  done
} > "$J"
echo "jobs: $(wc -l < "$J")" >> "$LOG"

# Thread caps: without them four concurrent jobs each spawn a full thread pool and the
# machine thrashes (same reason the cloud queue scripts pin OMP/MKL).
export TORCH_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
cat "$J" | xargs -P4 -n1 "$(dirname "$0")/topkjob.sh" >> "$LOG" 2>&1
echo "TOPK_DONE" >> "$LOG"
