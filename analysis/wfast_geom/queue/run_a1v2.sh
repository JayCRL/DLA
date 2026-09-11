#!/bin/bash
set -u
cd "$HOME/Desktop/dla-v0.2"
OUT="$HOME/llm-lab/dla_audit"
mkdir -p "$OUT"
# gamma-separated out dirs: $OUT/g0.05/{variant}, $OUT/g0.5/{variant}
{
  for s in 0 1 2; do
    for v in direct shufwrite; do
      for g in 0.05 0.5; do
        printf 'g%s|%s|HE|%s|%s\n' "$g" "$s" "$v" "$g"
      done
      # also restore the default gamma=1 file for these seeds
      printf 'g1|%s|HE|%s|1\n' "$s" "$v"
    done
  done
} > "$OUT/a1v2jobs.txt"
echo "jobs: $(wc -l < "$OUT/a1v2jobs.txt")"
export TORCH_THREADS=3
cat "$OUT/a1v2jobs.txt" | xargs -P4 -I{} bash -c '
  IFS="|" read -r gd s o v g <<< "{}"
  if [ "$gd" = "g1" ]; then tgt="$HOME/llm-lab/dla_audit"; else tgt="$HOME/llm-lab/dla_audit/$gd"; fi
  "$HOME/.venv/bin/python" analysis/wfast_geom/audit_b3.py --seed "$s" --order "$o" --variant "$v" --gamma "$g" --out "$tgt"
' >> "$OUT/a1v2_run.log" 2>&1
echo "A1V2_DONE" >> "$OUT/a1v2_run.log"
