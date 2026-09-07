#!/usr/bin/env python3
"""Per-module effective rank + W_fast_HE - W_fast_EH difference analysis.

Uses saved 5.5c/e bodies. For every seed:
  * group W_fast matrices into embedding / attention / mlp / head
  * report effective rank per group for EH and HE
  * compute D = W_fast_HE - W_fast_EH
  * report effective rank of D (weighted), plus top-10 explained variance
"""

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("s55c", ROOT / "experiments" / "stage55c_causality.py")
s55c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s55c)


class Args:
    eta_fast = 6e-4


def is_emb(key):
    return key.startswith("transformer.wte") or key.startswith("transformer.wpe") or key.startswith("lm_head")


def is_attn(key):
    return ".attn." in key


def is_mlp(key):
    return ".mlp." in key


def effective_rank_tensor(x: torch.Tensor) -> float:
    if x.numel() == 0 or x.shape[0] == 0 or x.shape[1] == 0:
        return float("nan")
    s = torch.linalg.svdvals(x.double())
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(-(p * torch.log(p + 1e-30)).sum().item())


def weighted_er(keys, store):
    total = 0.0
    w = 0.0
    for k in keys:
        if k not in store:
            continue
        er = effective_rank_tensor(store[k]["w_fast"])
        if not math.isnan(er):
            total += er * store[k]["w_fast"].numel()
            w += store[k]["w_fast"].numel()
    return total / w if w > 0 else float("nan")


def find_body(seed, order):
    c1 = ROOT / "results" / "stage55c" / f"seed{seed}" / "bodies" / f"seed{seed}_{order}_meta.pt"
    c2 = ROOT / "results" / "stage55e" / "bodies" / f"seed{seed}_{order}_meta.pt"
    return c1 if c1.exists() else c2


def group_keys(state):
    keys = list(state.store.keys())
    groups = {
        "embedding": [k for k in keys if is_emb(k)],
        "attention": [k for k in keys if is_attn(k)],
        "mlp": [k for k in keys if is_mlp(k)],
        "head": [k for k in keys if k.startswith("lm_head")],
        "all": keys,
    }
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out", type=str, default="results/wfast_decompose.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    group_names = ["embedding", "attention", "mlp", "all"]
    per_group = {g: [] for g in group_names}  # list of dicts seed EH HE diff
    diff_rows = []

    for seed in seeds:
        paths = {}
        states = {}
        for order in ("easy_hard", "hard_easy"):
            path = find_body(seed, order)
            if not path.exists():
                print("missing", path)
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                _, state = s55c.load_body(str(path), Args(), "cpu")
            states[order] = state

        if "easy_hard" not in states or "hard_easy" not in states:
            continue
        eh, he = states["easy_hard"], states["hard_easy"]
        row = {"seed": seed}
        for g in group_names:
            eh_keys = group_keys(eh)[g]
            he_keys = group_keys(he)[g]
            er_eh = weighted_er(eh_keys, eh.store)
            er_he = weighted_er(he_keys, he.store)
            row[f"EH_{g}"] = er_eh
            row[f"HE_{g}"] = er_he
            row[f"diff_{g}"] = er_he - er_eh
        # difference matrix effective rank and top variance
        d_ers = []
        var_top10 = []
        for k in eh.store:
            if k in he.store:
                D = he.store[k]["w_fast"].double() - eh.store[k]["w_fast"].double()
                d_ers.append((effective_rank_tensor(D), D.numel()))
                s = torch.linalg.svdvals(D)
                if s.numel() > 0:
                    p = (s * s).sum() if s.numel() > 0 else torch.tensor(0.0)
                    if s.numel() > 0:
                        total = (s * s).sum().item()
                        if total > 0:
                            cum = (s * s).cumsum(0).cpu().numpy()
                            var_top10.append(float(cum[min(9, len(cum) - 1)] / total))
        wsum = sum(w for _, w in d_ers)
        diff_er = sum(er * w for er, w in d_ers) / wsum if wsum > 0 else float("nan")
        row["diff_matrix_er"] = diff_er
        row["diff_top10_var"] = sum(var_top10) / len(var_top10) if var_top10 else float("nan")
        diff_rows.append(row)
        per_group["all"].append(row)
        for g in group_names:
            per_group[g].append(row)
        print(row, flush=True)

    # summary
    summary = {"n": len(diff_rows), "group_diff": {}, "diff_matrix": {}}
    for g in group_names:
        vals = [r[f"diff_{g}"] for r in diff_rows]
        summary["group_diff"][g] = {
            "mean": sum(vals) / len(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "per_seed": vals,
        }
    summary["diff_matrix"]["er_mean"] = sum(r["diff_matrix_er"] for r in diff_rows) / len(diff_rows)
    summary["diff_matrix"]["er_std"] = statistics.stdev([r["diff_matrix_er"] for r in diff_rows])
    summary["diff_matrix"]["top10_var_mean"] = sum(r["diff_top10_var"] for r in diff_rows) / len(diff_rows)
    summary["per_seed"] = diff_rows
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # plot group diffs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["embedding", "attention", "mlp", "all"]
    means = [summary["group_diff"][g]["mean"] for g in labels]
    stds = [summary["group_diff"][g]["std"] for g in labels]
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.axhline(0, color="black", lw=0.8)
    plt.ylabel("HE effective rank - EH effective rank")
    plt.title("Per-module W_fast effective rank difference")
    plt.tight_layout()
    png = str(Path(args.out).with_suffix("")) + "_groups.png"
    plt.savefig(png, dpi=150)
    print("saved", png)


if __name__ == "__main__":
    main()
