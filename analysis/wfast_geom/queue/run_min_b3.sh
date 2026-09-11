#!/bin/bash
set -u
cd "$HOME/Desktop/dla-v0.2"
OUT="$HOME/llm-lab/dla_audit"
mkdir -p "$OUT"
{
  for s in 0 1 2 3; do
    for o in EH HE; do
      for v in nocons direct; do
        printf '%s|%s|%s\n' "$s" "$o" "$v"
      done
    done
  done
} > "$OUT/minjobs.txt"
echo "jobs: $(wc -l < "$OUT/minjobs.txt")"
export TORCH_THREADS=3
cat "$OUT/minjobs.txt" | xargs -P4 -I{} bash -c '
  IFS="|" read -r s o v <<< "{}"
  "$HOME/.venv/bin/python" analysis/wfast_geom/audit_b3.py --seed "$s" --order "$o" --variant "$v" --out "$HOME/llm-lab/dla_audit"
' >> "$OUT/min_run.log" 2>&1
echo "MIN_B3_DONE" >> "$OUT/min_run.log"
tail -2 "$OUT/min_run.log"
