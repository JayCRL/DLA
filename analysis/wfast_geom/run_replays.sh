#!/usr/bin/env bash
# Instrumented replay of all Stage5.5e D-probes (12 seeds x 4 arms), 5 workers.
set -u
cd "$HOME/llm-lab/dla-v0.2"
mkdir -p results/analysis_wfast/replays/seeds
rm -f results/analysis_wfast/replays/jobs.txt results/analysis_wfast/replays/run.log
{
  for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
    for t in "EH/EH" "HE/HE" "EH_body+HE_fast" "HE_body+EH_fast"; do
      printf '%s|%s\n' "$s" "$t"
    done
  done
} > results/analysis_wfast/replays/jobs.txt
echo "jobs: $(wc -l < results/analysis_wfast/replays/jobs.txt)"
export OMP_NUM_THREADS=2
cat results/analysis_wfast/replays/jobs.txt | xargs -P5 -I{} bash -c '
  IFS="|" read -r s t <<< "{}"
  ~/llm-lab/venv/bin/python analysis/wfast_geom/replay.py --seed "$s" --arm "$t" --out results/analysis_wfast/replays
' >> results/analysis_wfast/replays/run.log 2>&1
echo "ALL_REPLAYS_DONE" >> results/analysis_wfast/replays/run.log
tail -3 results/analysis_wfast/replays/run.log
