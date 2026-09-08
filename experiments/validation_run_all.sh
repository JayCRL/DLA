#!/usr/bin/env bash
# Run validation P0 controls for seeds 0..11, 5 concurrent workers.
cd "$(dirname "$0")/.."
mkdir -p logs results/validation/seeds
for s in $(seq 0 11); do
    while [ "$(jobs -r | wc -l | tr -d ' ')" -ge 5 ]; do
        sleep 20
    done
    OMP_NUM_THREADS=2 ~/llm-lab/venv/bin/python experiments/validation_p0_controls.py \
        --seeds $s --d-steps 40 --out results/validation \
        > logs/validation_seed${s}.log 2>&1 &
    echo "started seed $s"
done
wait
echo "ALL_VALIDATION_DONE"
