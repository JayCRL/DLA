#!/bin/bash
set -u
cd "$HOME/Desktop/dla-v0.2"
OUT="$HOME/llm-lab/dla_audit"
{
  for s in 0 1 2; do
    for v in direct shufwrite; do
      for g in 0.05 0.5; do printf 'g%s|%s|HE|%s|%s\n' "$g" "$s" "$v" "$g"; done
      printf 'g1|%s|HE|%s|1\n' "$s" "$v"
    done
  done
} > "$OUT/a1v3jobs.txt"
echo "jobs: $(wc -l < "$OUT/a1v3jobs.txt")"
export TORCH_THREADS=3
cat "$OUT/a1v3jobs.txt" | xargs -P4 -n1 /tmp/a1job.sh >> "$OUT/a1v3_run.log" 2>&1
echo "A1V3_DONE" >> "$OUT/a1v3_run.log"
