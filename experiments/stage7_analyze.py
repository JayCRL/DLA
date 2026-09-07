#!/usr/bin/env python3
"""Analyze Stage 7 cross-domain 10-task results.

Usage:
    python experiments/stage7_analyze.py --out results/stage7
"""

import argparse
import glob
import json
import math
import os
import statistics

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/stage7")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.out, "seeds", "seed*.json")))
    data = {}
    for f in files:
        s = int(f.split("seed")[-1].split(".")[0])
        data[s] = json.load(open(f))["rows"]
    seeds = sorted(data)
    print("seeds", len(seeds))
    xs = np.arange(1, 11)
    norm_means, raw_means, se = [], [], []
    per_task = []
    for t in range(10):
        vals = [data[s][t]["norm_slope10"] for s in seeds]
        raws = [data[s][t]["raw_slope10"] for s in seeds]
        mu = sum(vals) / len(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        sem = sd / math.sqrt(len(vals))
        norm_means.append(mu)
        raw_means.append(sum(raws) / len(raws))
        se.append(sem)
        per_task.append(vals)
        print(f"task{t+1}: norm_slope {mu:+.4f}+/-{sem:.4f}")

    # primary test: per-seed linear regression slopes of task index on norm_slope
    per_seed_slopes = []
    for s in seeds:
        ys = [data[s][t]["norm_slope10"] for t in range(10)]
        per_seed_slopes.append(float(np.polyfit(xs, ys, 1)[0]))
    m = sum(per_seed_slopes) / len(per_seed_slopes)
    sd = statistics.stdev(per_seed_slopes) if len(per_seed_slopes) > 1 else 0.0
    sem = sd / math.sqrt(len(per_seed_slopes))
    tstat = m / sem if sem > 0 else 0.0

    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))

    p_two = 2 * (1 - norm_cdf(abs(tstat))) if abs(tstat) < 10 else 0.0
    ci = (m - 1.96 * sem, m + 1.96 * sem)
    print(f"\nmean per-seed regression slope: {m:+.5f} +/- {sd:.5f}, t={tstat:.3f}, p~{p_two:.4f}")
    print(f"95% CI: {ci}")
    print("norm_means", [round(x, 5) for x in norm_means])
    print("se", [round(x, 5) for x in se])
    summary = {
        "norm_means": norm_means,
        "se": se,
        "per_task": per_task,
        "regression_slope": m,
        "p": p_two,
        "ci": list(ci),
        "per_seed_slopes": per_seed_slopes,
    }
    with open(os.path.join(args.out, "analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("saved analysis.json")


if __name__ == "__main__":
    main()
