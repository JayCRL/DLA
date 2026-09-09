#!/usr/bin/env bash
# P3 confirmatory grid: per-seed W_fast interpolation probes.
# Grid via env: ALPHAS_HE="-0.5,0.5" ALPHAS_EH="0.5"  (alpha=0/1 endpoints exist in archives)
set -u
cd "$HOME/llm-lab/dla-v0.2"
mkdir -p results/analysis_wfast/p3/seeds
rm -f results/analysis_wfast/p3/jobs.txt
: "${ALPHAS_HE:=-0.5,0.5}"
: "${ALPHAS_EH:=0.5}"
{
  for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
    IFS=',' read -ra ah <<< "$ALPHAS_HE"
    for a in "${ah[@]}"; do printf 'HE|%s\n' "$a"; done | while read -r line; do
      printf '%s|%s\n' "$s" "$line"
    done
    IFS=',' read -ra ae <<< "$ALPHAS_EH"
    for a in "${ae[@]}"; do printf '%s|EH|%s\n' "$s" "$a"; done
  done
} > results/analysis_wfast/p3/jobs.txt
echo "p3 jobs: $(wc -l < results/analysis_wfast/p3/jobs.txt)"
export OMP_NUM_THREADS=2
cat results/analysis_wfast/p3/jobs.txt | xargs -P5 -I{} bash -c '
  IFS="|" read -r s b a <<< "{}"
  ~/llm-lab/venv/bin/python analysis/wfast_geom/p3_interpolate.py --seed "$s" --body "$b" --alpha "$a" --out results/analysis_wfast/p3/seeds
' >> results/analysis_wfast/p3/run.log 2>&1
echo "ALL_P3_DONE" >> results/analysis_wfast/p3/run.log
tail -2 results/analysis_wfast/p3/run.log
