#!/usr/bin/env bash
# B3 consolidation-pathway ablation grid on the Mac (72 jobs: 12 seeds x EH/HE x
# {qonly,direct,nocons}). Baseline "full" = archived bodies/probes (no rerun).
set -u
cd "$HOME/Desktop/dla-v0.2"
OUT="$HOME/llm-lab/dla_audit"
mkdir -p "$OUT"
rm -f "$OUT/jobs.txt" "$OUT/run.log"
{
  for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
    for o in EH HE; do
      for v in qonly direct nocons; do
        printf '%s|%s|%s\n' "$s" "$o" "$v"
      done
    done
  done
} > "$OUT/jobs.txt"
echo "jobs: $(wc -l < "$OUT/jobs.txt")"
export TORCH_THREADS=3
cat "$OUT/jobs.txt" | xargs -P4 -I{} bash -c '
  IFS="|" read -r s o v <<< "{}"
  "$HOME/.venv/bin/python" analysis/wfast_geom/audit_b3.py --seed "$s" --order "$o" --variant "$v" --out "$HOME/llm-lab/dla_audit"
' >> "$OUT/run.log" 2>&1
echo "ALL_AUDIT_DONE" >> "$OUT/run.log"
tail -2 "$OUT/run.log"
