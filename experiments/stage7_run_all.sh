#!/usr/bin/env bash
# Runs 20 Stage-7 seeds, 5 concurrent workers, OMP_NUM_THREADS=2 each.
cd "$(dirname "$0")/.."
mkdir -p logs results/stage7/seeds
for s in $(seq 0 19); do
    while [ "$(jobs -r | wc -l | tr -d ' ')" -ge 5 ]; do
        sleep 20
    done
    OMP_NUM_THREADS=2 ~/llm-lab/venv/bin/python experiments/stage7_cross_domain.py --seed $s > logs/stage7_seed${s}.log 2>&1 &
    echo "started seed $s pid $!"
done
wait
echo "ALL_STAGE7_DONE"
