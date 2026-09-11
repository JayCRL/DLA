#!/bin/bash
set -u
cd "$HOME/Desktop/dla-v0.2"
OUT="$HOME/llm-lab/dla_audit"
{
  for s in 0 1 2 3; do
    for o in EH HE; do
      if [ "$s" = 0 ] && [ "$o" = EH ]; then continue; fi
      printf '%s|%s|shufwrite\n' "$s" "$o"
    done
  done
} > "$OUT/shufjobs.txt"
echo "jobs: $(wc -l < "$OUT/shufjobs.txt")"
export TORCH_THREADS=3
cat "$OUT/shufjobs.txt" | xargs -P4 -I{} bash -c '
  IFS="|" read -r s o v <<< "{}"
  "$HOME/.venv/bin/python" analysis/wfast_geom/audit_b3.py --seed "$s" --order "$o" --variant "$v" --out "$HOME/llm-lab/dla_audit"
' >> "$OUT/shuf_run.log" 2>&1
echo "SHUF_DONE" >> "$OUT/shuf_run.log"
