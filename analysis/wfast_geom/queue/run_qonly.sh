#!/bin/bash
set -u
cd "$HOME/Desktop/dla-v0.2"
OUT="$HOME/llm-lab/dla_audit"
{
  for s in 0 1 2 3; do
    for o in EH HE; do
      printf '%s|%s|qonly\n' "$s" "$o"
    done
  done
} > "$OUT/qonlyjobs.txt"
echo "qonly jobs: $(wc -l < "$OUT/qonlyjobs.txt")"
export TORCH_THREADS=3
cat "$OUT/qonlyjobs.txt" | xargs -P4 -I{} bash -c '
  IFS="|" read -r s o v <<< "{}"
  "$HOME/.venv/bin/python" analysis/wfast_geom/audit_b3.py --seed "$s" --order "$o" --variant "$v" --out "$HOME/llm-lab/dla_audit"
' >> "$OUT/qonly_run.log" 2>&1
echo "QONLY_DONE" >> "$OUT/qonly_run.log"
