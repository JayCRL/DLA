#!/bin/bash
set -u
cd "$HOME/Desktop/dla-v0.2"
OUT="$HOME/llm-lab/dla_audit"
{
  for s in 4 5 6 7 8 9 10 11; do
    for o in EH HE; do
      for v in direct nocons shufwrite; do
        printf '%s|%s|%s\n' "$s" "$o" "$v"
      done
    done
  done
} > "$OUT/extjobs.txt"
echo "ext jobs: $(wc -l < "$OUT/extjobs.txt")"
export TORCH_THREADS=3
cat "$OUT/extjobs.txt" | xargs -P4 -I{} bash -c '
  IFS="|" read -r s o v <<< "{}"
  "$HOME/.venv/bin/python" analysis/wfast_geom/audit_b3.py --seed "$s" --order "$o" --variant "$v" --out "$HOME/llm-lab/dla_audit"
' >> "$OUT/ext_run.log" 2>&1
echo "EXT_DONE" >> "$OUT/ext_run.log"
