#!/bin/bash
set -u
cd "$HOME/Desktop/dla-v0.2"
OUT="$HOME/llm-lab/dla_audit"
{
  for s in 0 1 2; do
    for v in direct shufwrite; do
      for g in 0.05 0.5; do
        printf '%s|HE|%s|%s\n' "$s" "$v" "$g"
      done
    done
  done
} > "$OUT/a1jobs.txt"
echo "A1 jobs: $(wc -l < "$OUT/a1jobs.txt")"
export TORCH_THREADS=3
cat "$OUT/a1jobs.txt" | xargs -P4 -I{} bash -c '
  IFS="|" read -r s o v g <<< "{}"
  "$HOME/.venv/bin/python" analysis/wfast_geom/audit_b3.py --seed "$s" --order "$o" --variant "$v" --gamma "$g" --out "$HOME/llm-lab/dla_audit"
' >> "$OUT/a1_run.log" 2>&1
echo "A1_DONE" >> "$OUT/a1_run.log"
