#!/usr/bin/env python3
"""Compare W_fast effective rank between easy->hard (EH) and hard->easy (HE) bodies.

Uses saved DLA bodies from Stage 5.5c/e (seeds 0-9). Effective rank is computed
per 2D weight matrix:

    p_i = s_i / sum(s)
    effective_rank = exp(-sum_i p_i log p_i)

Reports mean effective rank per order and a paired test.
"""

import argparse
import importlib.util
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


def effective_rank_tensor(x: torch.Tensor) -> float:
    if x.numel() == 0 or x.shape[0] == 0 or x.shape[1] == 0:
        return float("nan")
    s = torch.linalg.svdvals(x.double())
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    log_p = torch.log(p + 1e-30)
    return float(-(p * log_p).sum().item())


def effective_rank_weighted(store) -> float:
    """Mean effective rank weighted by number of elements."""
    total = 0.0
    weight = 0.0
    for s in store.values():
        w = s["w_fast"]
        er = effective_rank_tensor(w)
        if not math.isnan(er):
            total += er * w.numel()
            weight += w.numel()
    return total / weight if weight > 0 else float("nan")


def find_body(seed: int, order: str) -> Path:
    cand1 = ROOT / "results" / "stage55c" / f"seed{seed}" / "bodies" / f"seed{seed}_{order}_meta.pt"
    cand2 = ROOT / "results" / "stage55e" / "bodies" / f"seed{seed}_{order}_meta.pt"
    if cand1.exists():
        return cand1
    return cand2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out", type=str, default="results/effective_rank.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        rec = {"seed": seed}
        for order in ("easy_hard", "hard_easy"):
            path = find_body(seed, order)
            if not path.exists():
                print(f"missing body {path}")
                continue
            with open(os.devnull, "w") as devnull:
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    model, state = s55c.load_body(str(path), Args(), "cpu")
            rec[order] = effective_rank_weighted(state.store)
        if "easy_hard" in rec and "hard_easy" in rec:
            rec["diff_HE_minus_EH"] = rec["hard_easy"] - rec["easy_hard"]
            rows.append(rec)
        print(rec, flush=True)

    # aggregate
    eh = [r["easy_hard"] for r in rows]
    he = [r["hard_easy"] for r in rows]
    diff = [r["diff_HE_minus_EH"] for r in rows]
    means = lambda x: sum(x) / len(x)
    stdev = lambda x: statistics.stdev(x) if len(x) > 1 else 0.0

    # paired t-test
    n = len(diff)
    m = means(diff)
    sd = stdev(diff)
    sem = sd / math.sqrt(n) if n > 0 else 0.0
    t = m / sem if sem > 0 else 0.0

    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))

    p_two = 2 * (1 - norm_cdf(abs(t))) if abs(t) < 10 else 0.0
    summary = {
        "n": n,
        "EH_mean": means(eh),
        "EH_std": stdev(eh),
        "HE_mean": means(he),
        "HE_std": stdev(he),
        "diff_mean": m,
        "diff_std": sd,
        "t": t,
        "p_two": p_two,
        "per_seed": rows,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(len(seeds))
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(xs, [means(eh)] * len(xs), yerr=[stdev(eh)] * len(xs), fmt="o-", label="EH (easy->hard)")
    plt.errorbar(xs, [means(he)] * len(xs), yerr=[stdev(he)] * len(xs), fmt="s-", label="HE (hard->easy)")
    # per seed points
    plt.plot(xs, eh, "o", alpha=0.4)
    plt.plot(xs, he, "s", alpha=0.4)
    plt.xlabel("seed")
    plt.ylabel("effective rank of W_fast (weighted mean)")
    plt.title("W_fast effective rank: EH vs HE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    png = str(Path(args.out).with_suffix(".png"))
    plt.savefig(png, dpi=150)
    print("saved", png)


if __name__ == "__main__":
    main()
