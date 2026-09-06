"""Stage 2 - "the learning rule itself matters".

Question: a meta-learned Learning Rule Network F_phi should outperform a fixed
Hebbian plasticity rule.  Both arms share:

* the same Neural Core,
* the same meta-objective and unrolling procedure,
* a meta-learned initial W_slow (the fixed-Hebbian arm optimises W_slow only,
  while its rule and tempos stay at their DNA values).

So any gap isolates the contribution of *learning the rule* versus merely having
plasticity.

Run:
    ~/llm-lab/venv/bin/python experiments/stage2_learned_rule.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dla import CoreConfig, DLAConfig, DevelopmentalNet, make_task_stream, meta_train
from dla.baselines import hebbian_dla_config
from dla.metrics import run_dla_stream, sanitize_for_json, summarize_records
from dla.plotting import plot_bars, plot_learning_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_meta_train", type=int, default=12)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=str, default="results/stage2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg = DLAConfig(core=CoreConfig(input_dim=6, hidden_dims=(args.hidden,), output_dim=2))
    meta_tasks, eval_tasks = make_task_stream(args.n_meta_train, args.n_eval, input_dim=6)
    os.makedirs(args.out, exist_ok=True)

    learned_recs, hebbian_recs = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        print(f"\n===== seed {seed} =====")

        learned = DevelopmentalNet(cfg)
        print("  meta-training learned rule ...")
        meta_train(learned, meta_tasks, epochs=args.epochs, batch_size=8, lr_rule=3e-3, lr_slow=1e-2, lifetime_len=2, retain_weight=1.0)
        learned_recs.append(run_dla_stream(learned, eval_tasks, batch_size=8))

        torch.manual_seed(seed)
        hebbian = DevelopmentalNet(hebbian_dla_config(cfg))
        print("  meta-training fixed-Hebbian (initial W_slow only) ...")
        meta_train(hebbian, meta_tasks, epochs=args.epochs, batch_size=8, include_tempos=False, lr_slow=1e-2, lifetime_len=2, retain_weight=1.0)
        hebbian_recs.append(run_dla_stream(hebbian, eval_tasks, batch_size=8))

        print(
            "  seed summary: "
            f"learned post={learned_recs[-1]['avg_post_acc']:.3f} forget={learned_recs[-1]['forgetting']:.3f} | "
            f"hebbian post={hebbian_recs[-1]['avg_post_acc']:.3f} forget={hebbian_recs[-1]['forgetting']:.3f}"
        )

    names = ["Fixed Hebbian", "Learned rule (F_phi)"]
    summaries = [summarize_records(hebbian_recs), summarize_records(learned_recs)]

    print("\n===== Stage 2 results (mean +/- std over seeds) =====")
    print(f"{'metric':<18}{'Fixed Hebbian':>22}{'Learned F_phi':>22}")
    for key, label in [
        ("avg_post_acc", "post-task acc"),
        ("avg_slow_final", "final slow acc"),
        ("avg_steps", "steps->85%"),
        ("forgetting", "forgetting"),
        ("bwt", "backward transfer"),
    ]:
        row = f"{label:<18}"
        for s in summaries:
            v = s[key]
            row += f"{v['mean']:>12.3f}+/-{v['std']:.3f}"
        print(row)

    curve_data = {
        "Fixed Hebbian": [c for r in hebbian_recs for c in r["curves"]],
        "Learned F_phi": [c for r in learned_recs for c in r["curves"]],
    }
    plot_learning_curves(curve_data, f"{args.out}/stage2_curves.png", title="Stage 2: fixed Hebbian vs learned plasticity rule")
    plot_bars(
        names,
        [s["avg_post_acc"]["mean"] for s in summaries],
        [s["avg_post_acc"]["std"] for s in summaries],
        f"{args.out}/stage2_post_acc.png",
        "avg post-task accuracy",
        "Stage 2: learned rule vs fixed Hebbian",
        ylim=(0.3, 1.0),
    )
    plot_bars(
        names,
        [s["forgetting"]["mean"] for s in summaries],
        [s["forgetting"]["std"] for s in summaries],
        f"{args.out}/stage2_forgetting.png",
        "forgetting (lower is better)",
        "Stage 2: forgetting",
    )

    result = {
        "config": {"epochs": args.epochs, "seeds": seeds, "hidden": args.hidden},
        "learned": {"raw": learned_recs, "summary": summaries[1]},
        "hebbian": {"raw": hebbian_recs, "summary": summaries[0]},
    }
    with open(f"{args.out}/results.json", "w") as f:
        json.dump(sanitize_for_json(result), f, indent=2)
    print(f"\nsaved -> {args.out}/results.json + plots")


if __name__ == "__main__":
    main()
