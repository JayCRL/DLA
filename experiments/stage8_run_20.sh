#!/usr/bin/env bash
# Run 20 Stage-8 physics seeds, 5 concurrent workers.
cd "$(dirname "$0")/.."
mkdir -p logs results/stage8_physics_20/seeds
started=0
for s in $(seq 0 19); do
    while [ "$(jobs -r | wc -l | tr -d ' ')" -ge 5 ]; do
        sleep 20
    done
    OMP_NUM_THREADS=2 ~/llm-lab/venv/bin/python experiments/stage8_physics.py \
        --seeds $s --max-steps 40 --train-chars 100000 --val-chars 10000 \
        --out results/stage8_physics_20 > logs/stage8_physics20_seed${s}.log 2>&1 &
    started=$((started+1))
    echo "started seed $s ($started/20)"
done
wait
echo "ALL_STAGE8_20_DONE"
