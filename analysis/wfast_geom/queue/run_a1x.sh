#!/bin/bash
set -u
OUT="$HOME/llm-lab/dla_audit"; J="$OUT/a1xjobs.txt"
{
  # gamma expansion: gamma=0.5 (seeds 3-11) and gamma=1.5 (seeds 0-11) for direct & shufwrite
  for s in 3 4 5 6 7 8 9 10 11; do
    for v in direct shufwrite; do printf 'g0.5|%s|%s|0.5|-\n' "$s" "$v"; done
  done
  for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
    for v in direct shufwrite; do printf 'g1.5|%s|%s|1.5|-\n' "$s" "$v"; done
  done
  # decay sweep: sleep decay 0.25 / 0.75, direct only, seeds 0-5
  for s in 0 1 2 3 4 5; do
    printf 'd0.25|%s|direct|-|0.25\n' "$s"
    printf 'd0.75|%s|direct|-|0.75\n' "$s"
  done
} > "$J"
echo "jobs: $(wc -l < "$J")"
export TORCH_THREADS=3
cat "$J" | xargs -P4 -n1 /tmp/a1xjob.sh >> "$OUT/a1x_run.log" 2>&1
echo "A1X_DONE" >> "$OUT/a1x_run.log"
