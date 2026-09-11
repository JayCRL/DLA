#!/bin/bash
set -u
OUT="$HOME/llm-lab/dla_audit_second"; mkdir -p "$OUT"
for s in 1 2 3 4 5 6 7 8; do echo "$s"; done > "$OUT/secjobs.txt"
export TORCH_THREADS=3
cat "$OUT/secjobs.txt" | xargs -P4 -n1 /tmp/secjob.sh >> "$OUT/sec_run.log" 2>&1
echo "SEC_DONE" >> "$OUT/sec_run.log"
