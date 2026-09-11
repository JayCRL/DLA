#!/bin/bash
set -u
cd "$HOME/Desktop/dla-v0.2"
OUT="$HOME/llm-lab/dla_audit"
{
  for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
    for v in direct nocons nosleep uniformwrite topwrite shufwrite; do
      printf '%s|HE|%s\n' "$s" "$v"
    done
  done
} > "$OUT/unifiedjobs.txt"
echo "jobs: $(wc -l < "$OUT/unifiedjobs.txt")"
export TORCH_THREADS=3
cat "$OUT/unifiedjobs.txt" | xargs -P4 -I{} bash -c '
  IFS="|" read -r s o v <<< "{}"
  "$HOME/.venv/bin/python" analysis/wfast_geom/audit_b3.py --seed "$s" --order "$o" --variant "$v" --out "$HOME/llm-lab/dla_audit"
' >> "$OUT/unified_run.log" 2>&1
echo "UNIFIED_DONE" >> "$OUT/unified_run.log"
tail -2 "$OUT/unified_run.log"
