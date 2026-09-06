"""One-shot smoke run of the whole pipeline with tiny settings.

    ~/llm-lab/venv/bin/python experiments/run_all_smoke.py
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.makedirs(ROOT / "results" / "smoke", exist_ok=True)
PY = sys.executable

commands = [
    f"{PY} {ROOT}/experiments/stage1_adaptive_plasticity.py --seeds 0 --epochs 1 --n_meta_train 2 --n_eval 3 --hidden 8 --out {ROOT}/results/smoke/stage1",
    f"{PY} {ROOT}/experiments/stage2_learned_rule.py --seeds 0 --epochs 1 --n_meta_train 2 --n_eval 3 --hidden 8 --out {ROOT}/results/smoke/stage2",
    f"{PY} {ROOT}/experiments/stage3_learning_rule_development.py --seeds 0 --epochs 1 --n_meta_train 2 --n_eval 3 --hidden 8 --out {ROOT}/results/smoke/stage3",
    f"{PY} {ROOT}/experiments/dna_lifetimes.py --seeds 0 --epochs 1 --n_meta_train 2 --n_eval 3 --hidden 8 --out {ROOT}/results/smoke/dna",
]

for cmd in commands:
    print("\n" + "=" * 80)
    print(cmd)
    print("=" * 80, flush=True)
    rc = subprocess.call(cmd, shell=True, cwd=str(ROOT))
    if rc != 0:
        print(f"FAILED (rc={rc}): {cmd}")
        sys.exit(rc)

print("\nSMOKE PIPELINE OK")
