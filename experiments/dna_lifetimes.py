"""DNA experiment - "same life, different developmental trajectories".

Four individuals share the same meta-learned Learning Rule Network F_phi but are
born with different DNA priors (plasticity prior, tempos, signal-channel weights):

    DNA-A high plasticity   DNA-B high stability
    DNA-C novelty seeking   DNA-D conservative

They live through the same sequence of tasks; we record plasticity P(t),
knowledge drift, learning capacity Lambda_t and forgetting.

Run:
    ~/llm-lab/venv/bin/python experiments/dna_lifetimes.py
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
from dla.dna import dna_presets
from dla.metrics import run_dla_stream, sanitize_for_json
from dla.plotting import plot_bars, plot_lambda, plot_trajectories


def mean_series(records, key):
    n = len(records[0][key])
    out = []
    for i in range(n):
        out.append(sum(r[key][i] for r in records) / len(records))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_meta_train", type=int, default=12)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=str, default="results/dna")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg = DLAConfig(core=CoreConfig(input_dim=6, hidden_dims=(args.hidden,), output_dim=2))
    meta_tasks, eval_tasks = make_task_stream(args.n_meta_train, args.n_eval, input_dim=6)
    os.makedirs(args.out, exist_ok=True)
    presets = dna_presets(cfg)

    all_records = {name: [] for name in presets}
    for seed in seeds:
        torch.manual_seed(seed)
        print(f"\n===== seed {seed}: meta-training shared F_phi =====")
        base = DevelopmentalNet(cfg)
        meta_train(base, meta_tasks, epochs=args.epochs, batch_size=8, lr_rule=3e-3, lr_slow=1e-2, lifetime_len=2, retain_weight=1.0)

        for name, dna_cfg in presets.items():
            individual = DevelopmentalNet(dna_cfg).copy_rule_from(base)
            print(f"  life of {name} ...")
            rec = run_dla_stream(individual, eval_tasks, batch_size=8)
            all_records[name].append(rec)
            print(
                f"    post={rec['avg_post_acc']:.3f} slow={rec['avg_slow_final']:.3f} "
                f"forget={rec['forgetting']:.3f} steps={rec['avg_steps']:.2f}"
            )

    print("\n===== DNA trajectories (mean over seeds) =====")
    print(f"{'DNA':<20}{'post acc':>10}{'slow acc':>10}{'forgetting':>12}{'avg steps':>12}")
    scalar_summary = {}
    for name in presets:
        recs = all_records[name]
        post = sum(r["avg_post_acc"] for r in recs) / len(recs)
        slow = sum(r["avg_slow_final"] for r in recs) / len(recs)
        forget = sum(r["forgetting"] for r in recs) / len(recs)
        steps = sum(r["avg_steps"] for r in recs) / len(recs)
        scalar_summary[name] = {"post": post, "slow": slow, "forgetting": forget, "steps": steps}
        print(f"{name:<20}{post:>10.3f}{slow:>10.3f}{forget:>12.3f}{steps:>12.2f}")

    plot_bars(
        list(presets),
        [scalar_summary[n]["post"] for n in presets],
        [0.0] * len(presets),
        f"{args.out}/dna_post_acc.png",
        "avg post-task accuracy",
        "Same F_phi, different DNA priors",
        ylim=(0.3, 1.0),
    )
    plot_bars(
        list(presets),
        [scalar_summary[n]["forgetting"] for n in presets],
        [0.0] * len(presets),
        f"{args.out}/dna_forgetting.png",
        "forgetting (lower is better)",
        "Stability-plasticity profiles of the four DNA variants",
    )
    plot_trajectories(
        {n: mean_series(all_records[n], "plasticity") for n in presets},
        f"{args.out}/dna_plasticity.png",
        xlabel="task index",
        ylabel="mean expressed plasticity softplus(P)",
        title="Plasticity development over the same life",
    )
    plot_trajectories(
        {n: mean_series(all_records[n], "knowledge_drift") for n in presets},
        f"{args.out}/dna_knowledge.png",
        xlabel="task index",
        ylabel="relative W_slow drift (knowledge proxy)",
        title="Knowledge accumulation over the same life",
    )
    plot_lambda(
        {n: [1.0 / (s + 1) for s in mean_series(all_records[n], "steps_to_threshold")] for n in presets},
        f"{args.out}/dna_lambda.png",
    )

    result = {"config": vars(args), "scalars": scalar_summary, "records": {n: sanitize_for_json(v) for n, v in all_records.items()}}
    with open(f"{args.out}/results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {args.out}/results.json + plots")


if __name__ == "__main__":
    main()
