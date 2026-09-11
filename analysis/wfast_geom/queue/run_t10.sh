#!/bin/bash
set -u
OUT="$HOME/llm-lab/dla_audit_t10"
mkdir -p "$OUT"
{
  for s in 1 2 3 4 5 6 7; do
    for v in direct nocons nosleep; do printf '%s|%s\n' "$s" "$v"; done
  done
  printf '0|nocons\n0|nosleep\n'
} > "$OUT/t10jobs.txt"
echo "jobs: $(wc -l < "$OUT/t10jobs.txt")"
export TORCH_THREADS=3
cat "$OUT/t10jobs.txt" | xargs -P4 -n1 /tmp/t10job.sh >> "$OUT/t10_run.log" 2>&1
echo "T10_DONE" >> "$OUT/t10_run.log"
