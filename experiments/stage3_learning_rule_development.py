"""Stage 3 - "learning itself can be a developmental state".

Core hypothesis of v0.2:

    Learning ability_{t+1} > Learning ability_t   is possible.

Two individuals start from the same meta-trained phi:

* FROZEN  : phi stays fixed for the whole lifetime.
* ADAPTIVE: phi is updated inside the lifetime by meta-gradient steps on the
            individual's own replay buffer H_t (``LifetimeAdapter``), i.e.

                phi_{t+1} = phi_t + G(experience history).

We operationalise learning ability as

    Lambda_t = 1 / steps_to_reach_85%_on_task_t,

and test whether Lambda_t rises over the lifetime more in the adaptive arm.

Run:
    ~/llm-lab/venv/bin/python experiments/stage3_learning_rule_development.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dla import CoreConfig, DLAConfig, DevelopmentalNet, LifetimeAdapter, make_task_stream, meta_train
from dla.metrics import run_dla_stream, sanitize_for_json
from dla.plotting import plot_lambda, plot_trajectories


def mean_series(records, key, per_task=True):
    """Average a per-task series across seeds."""
    n = len(records[0][key])
    out = []
    for i in range(n):
        vals = [r[key][i] for r in records]
        out.append(sum(vals) / len(vals))
    return out


def slope(xs, ys):
    return float(np.polyfit(xs, ys, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_meta_train", type=int, default=12)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--adapt_every", type=int, default=1)
    ap.add_argument("--out", type=str, default="results/stage3")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg = DLAConfig(core=CoreConfig(input_dim=6, hidden_dims=(args.hidden,), output_dim=2))
    meta_tasks, eval_tasks = make_task_stream(args.n_meta_train, args.n_eval, input_dim=6)
    os.makedirs(args.out, exist_ok=True)

    frozen_recs, adaptive_recs = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        print(f"\n===== seed {seed} =====")
        base = DevelopmentalNet(cfg)
        print(f"  meta-training base phi ({args.epochs} epochs) ...")
        meta_train(base, meta_tasks, epochs=args.epochs, batch_size=8, lr_rule=3e-3, lr_slow=1e-2, lifetime_len=2, retain_weight=1.0)

        frozen = copy.deepcopy(base)
        print("  lifetime arm: frozen phi ...")
        frozen_recs.append(run_dla_stream(frozen, eval_tasks, batch_size=8))

        adaptive = copy.deepcopy(base)
        adapter = LifetimeAdapter(
            adaptive,
            capacity=128,
            unroll_len=3,
            meta_batch=8,
            lr=args.adapt_lr,
            every=args.adapt_every,
            min_items=24,
            seed=seed,
        )
        print("  lifetime arm: adaptive phi ...")
        adaptive_recs.append(run_dla_stream(adaptive, eval_tasks, batch_size=8, adapter=adapter))

        print(
            "  seed summary: "
            f"frozen post={frozen_recs[-1]['avg_post_acc']:.3f} steps={frozen_recs[-1]['avg_steps']:.2f} | "
            f"adaptive post={adaptive_recs[-1]['avg_post_acc']:.3f} steps={adaptive_recs[-1]['avg_steps']:.2f}"
        )

    frozen_steps = mean_series(frozen_recs, "steps_to_threshold")
    adaptive_steps = mean_series(adaptive_recs, "steps_to_threshold")
    frozen_lambda = [1.0 / (s + 1) for s in frozen_steps]
    adaptive_lambda = [1.0 / (s + 1) for s in adaptive_steps]
    xs = list(range(1, len(frozen_steps) + 1))

    print("\n===== Stage 3 results (mean over seeds) =====")
    print(f"{'task':<6}{'frozen steps':>14}{'adaptive steps':>16}{'frozen Lambda':>15}{'adaptive Lambda':>17}")
    for i in range(len(frozen_steps)):
        print(
            f"{xs[i]:<6}{frozen_steps[i]:>14.2f}{adaptive_steps[i]:>16.2f}"
            f"{frozen_lambda[i]:>15.3f}{adaptive_lambda[i]:>17.3f}"
        )
    print(f"slope of steps vs task:      frozen={slope(xs, frozen_steps):+.3f}  adaptive={slope(xs, adaptive_steps):+.3f}")
    print(f"slope of Lambda vs task:     frozen={slope(xs, frozen_lambda):+.3f}  adaptive={slope(xs, adaptive_lambda):+.3f}")

    plot_trajectories(
        {"frozen phi": frozen_steps, "adaptive phi": adaptive_steps},
        f"{args.out}/stage3_steps.png",
        xlabel="task index in lifetime",
        ylabel="steps to reach 85% test accuracy",
        title="Stage 3: learning efficiency across the lifetime",
    )
    plot_lambda({"frozen phi": frozen_lambda, "adaptive phi": adaptive_lambda}, f"{args.out}/stage3_lambda.png")

    result = {
        "config": vars(args),
        "frozen": {"raw": frozen_recs, "steps_mean": frozen_steps, "lambda_mean": frozen_lambda},
        "adaptive": {"raw": adaptive_recs, "steps_mean": adaptive_steps, "lambda_mean": adaptive_lambda},
    }
    with open(f"{args.out}/results.json", "w") as f:
        json.dump(sanitize_for_json(result), f, indent=2)
    print(f"\nsaved -> {args.out}/results.json + plots")


if __name__ == "__main__":
    main()
