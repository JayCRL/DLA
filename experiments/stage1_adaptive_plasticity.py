"""Stage 1 - "plasticity is effective".

Question: does an adaptive developmental system beat a static MLP (whose learning
rule - SGD - lives outside the model) on the same continual lifetime?

Arms:
    DLA (learned rule, meta-trained phi)
    StaticMLP (random init + SGD, lr selected on meta-train tasks)
    StaticMLP-meta-init (DLA's meta-learned W_slow init + SGD) - an ablation that
                         separates "good initialisation" from "developmental
                         plasticity".

Run:
    cd ~/llm-lab/dla-v0.2
    ~/llm-lab/venv/bin/python experiments/stage1_adaptive_plasticity.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dla import CoreConfig, DLAConfig, DevelopmentalNet, StaticMLP, make_task_stream, meta_train
from dla.metrics import run_dla_stream, run_static_stream, sanitize_for_json, summarize_records
from dla.plotting import plot_bars, plot_learning_curves


def pick_best_static_lr(model_template, meta_tasks, lrs=(0.02, 0.05, 0.1, 0.2, 0.4), batch_size=8):
    """Tune the SGD learning rate of a static baseline on the meta-train stream."""
    best_lr, best_score = lrs[0], -1.0
    for lr in lrs:
        m = copy.deepcopy(model_template)
        rec = run_static_stream(m, meta_tasks, lr=lr, batch_size=batch_size)
        if rec["avg_post_acc"] > best_score:
            best_score, best_lr = rec["avg_post_acc"], lr
    return best_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_meta_train", type=int, default=12)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=str, default="results/stage1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg = DLAConfig(core=CoreConfig(input_dim=6, hidden_dims=(args.hidden,), output_dim=2))
    meta_tasks, eval_tasks = make_task_stream(args.n_meta_train, args.n_eval, input_dim=6)
    os.makedirs(args.out, exist_ok=True)

    dla_recs, static_recs, static_init_recs = [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        print(f"\n===== seed {seed} =====")
        net = DevelopmentalNet(cfg)
        print(f"  meta-training DLA ({args.epochs} epochs) ...")
        hist = meta_train(net, meta_tasks, epochs=args.epochs, batch_size=8, lr_rule=3e-3, lr_slow=1e-2, lifetime_len=2, retain_weight=1.0)
        print(f"    meta loss {hist[0]:.4f} -> {hist[-1]:.4f}")
        rec_dla = run_dla_stream(net, eval_tasks, batch_size=8)
        dla_recs.append(rec_dla)

        torch.manual_seed(seed)
        plain_template = StaticMLP(cfg.core)
        lr_plain = pick_best_static_lr(plain_template, meta_tasks)
        meta_init_template = StaticMLP.from_dla(net)
        lr_init = pick_best_static_lr(meta_init_template, meta_tasks)
        print(f"  static baselines, best lr: plain={lr_plain}  meta-init={lr_init}")

        static_recs.append(run_static_stream(copy.deepcopy(plain_template), eval_tasks, lr=lr_plain, batch_size=8))
        static_init_recs.append(run_static_stream(copy.deepcopy(meta_init_template), eval_tasks, lr=lr_init, batch_size=8))

        print(
            "  seed summary: "
            f"DLA post={rec_dla['avg_post_acc']:.3f} forget={rec_dla['forgetting']:.3f} | "
            f"static={static_recs[-1]['avg_post_acc']:.3f} | static+metaInit={static_init_recs[-1]['avg_post_acc']:.3f}"
        )

    names = ["StaticMLP", "StaticMLP\n(meta init)", "DLA (adaptive plasticity)"]
    summaries = [summarize_records(static_recs), summarize_records(static_init_recs), summarize_records(dla_recs)]

    print("\n===== Stage 1 results (mean +/- std over seeds) =====")
    print(f"{'metric':<18}{'StaticMLP':>22}{'StaticMLP+init':>22}{'DLA':>22}")
    for key, label in [("avg_post_acc", "post-task acc"), ("avg_steps", "steps->85%"), ("forgetting", "forgetting")]:
        row = f"{label:<18}"
        for s in summaries:
            v = s[key]
            row += f"{v['mean']:>12.3f}+/-{v['std']:.3f}"
        print(row)

    curve_data = {
        "StaticMLP": [c for r in static_recs for c in r["curves"]],
        "StaticMLP (meta init)": [c for r in static_init_recs for c in r["curves"]],
        "DLA": [c for r in dla_recs for c in r["curves"]],
    }
    plot_learning_curves(curve_data, f"{args.out}/stage1_curves.png", title="Stage 1: static MLP vs adaptive plasticity")
    plot_bars(
        names,
        [s["avg_post_acc"]["mean"] for s in summaries],
        [s["avg_post_acc"]["std"] for s in summaries],
        f"{args.out}/stage1_post_acc.png",
        "avg post-task accuracy",
        "Stage 1: adaptive plasticity effectiveness",
        ylim=(0.3, 1.0),
    )
    plot_bars(
        names,
        [s["forgetting"]["mean"] for s in summaries],
        [s["forgetting"]["std"] for s in summaries],
        f"{args.out}/stage1_forgetting.png",
        "forgetting (lower is better)",
        "Stage 1: stability-plasticity",
    )

    result = {
        "config": {"epochs": args.epochs, "seeds": seeds, "hidden": args.hidden},
        "DLA": {"raw": dla_recs, "summary": summaries[2]},
        "StaticMLP": {"raw": static_recs, "summary": summaries[0]},
        "StaticMLP_meta_init": {"raw": static_init_recs, "summary": summaries[1]},
    }
    with open(f"{args.out}/results.json", "w") as f:
        json.dump(sanitize_for_json(result), f, indent=2)
    print(f"\nsaved -> {args.out}/results.json + plots")


if __name__ == "__main__":
    main()
