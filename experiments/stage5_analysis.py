"""Stage 5 post-hoc analysis: raw Lambda vs difficulty-normalized Learning Efficiency.

Problem identified during the run: with a progressive (harder over time) curriculum,
the fixed-threshold metric

    Lambda_t = 1 / steps_to_reach_4%_relative_gain

confounds Learning Capacity with Task Difficulty.  This script reports BOTH:

* raw Lambda_t (kept exactly as measured, for transparency)
* normalized Learning Efficiency:

      G_max(t)   = max gain observed for this individual/stage within the budget
      LE_t       = G(final) / G_max                (did it keep its own best?)
      T80_t      = min k : G(k) >= 0.8 * G_max     (how fast it reached ~80% of best)
      speed80_t  = 1 / T80_t

The decisive test is the WITHIN-INDIVIDUAL trend over developmental stages:

      LE_DLA(t+1) > LE_DLA(t)    ?

and its contrast with AdamW.

Run after stage5_progressive_curriculum.py:

    ~/llm-lab/venv/bin/python experiments/stage5_analysis.py --in results/stage5/stage5.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def curve_stats(curve, max_steps):
    if not curve:
        return None
    pre_gain_max = max(point["gain"] for point in curve)
    final_gain = curve[-1]["gain"]
    g_max = max(pre_gain_max, 0.0)
    le = final_gain / g_max if g_max > 0 else 0.0
    t80 = max_steps
    for point in curve:
        if g_max > 0 and point["gain"] >= 0.8 * g_max:
            t80 = point["step"]
            break
    return {
        "best_gain": pre_gain_max,
        "g_max": g_max,
        "final_gain": final_gain,
        "LE": le,
        "T80": t80,
        "speed80": 1.0 / t80 if t80 > 0 else 0.0,
    }


def analyze_runs(runs, max_steps, use_slow=False):
    out = []
    for rec in runs:
        curves = rec["curve_slow"] if use_slow else rec["curve"]
        out.append([curve_stats(c, max_steps) for c in curves])
    return out


def mean_series(list_of_lists, key):
    n = len(list_of_lists[0])
    means, stds = [], []
    for i in range(n):
        vals = [row[i][key] for row in list_of_lists if row[i] is not None]
        mu = sum(vals) / len(vals)
        means.append(mu)
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return means, stds


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def plot_series(series_by_name, out_path, ylabel, title):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    xs = list(range(len(next(iter(series_by_name.values()))[0])))
    for name, (means, stds) in series_by_name.items():
        plt.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=name)
    plt.xlabel("developmental stage (easy -> hard)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, default="results/stage5/stage5.json")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--out", type=str, default="results/stage5/analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = json.load(open(args.in_path))
    summ = data["summary"]
    order = summ.get("curriculum_order", {})
    max_steps = data["args"].get("max_steps", args.max_steps)

    base_runs = data.get("baseline_runs", [])
    dla_runs = data.get("dla_runs", [])

    report = {"raw_lambda": {}, "normalized": {}, "order": order}

    # -------- raw Lambda (already computed, keep as-is) -----------------------
    if "baseline" in summ:
        report["raw_lambda"]["AdamW"] = summ["baseline"]["lambda_mean"]
    if "dla" in summ:
        report["raw_lambda"]["DLA_fast"] = summ["dla"]["lambda_mean"]
        report["raw_lambda"]["DLA_slow"] = summ["dla"]["lambda_slow_mean"]

    # -------- normalized LE ---------------------------------------------------
    xs = [0, 1, 2]
    if base_runs:
        stats = analyze_runs(base_runs, max_steps)
        for key in ("LE", "speed80", "T80", "best_gain", "final_gain"):
            m, s = mean_series(stats, key)
            report["normalized"][f"AdamW_{key}"] = {"mean": m, "std": s}
        report["normalized"]["AdamW_LE_slope"] = slope(xs, report["normalized"]["AdamW_LE"]["mean"])
        report["normalized"]["AdamW_speed80_slope"] = slope(xs, report["normalized"]["AdamW_speed80"]["mean"])

    if dla_runs:
        for label, slow in (("DLA_fast", False), ("DLA_slow", True)):
            stats = analyze_runs(dla_runs, max_steps, use_slow=slow)
            for key in ("LE", "speed80", "T80", "best_gain", "final_gain"):
                m, s = mean_series(stats, key)
                report["normalized"][f"{label}_{key}"] = {"mean": m, "std": s}
            report["normalized"][f"{label}_LE_slope"] = slope(xs, report["normalized"][f"{label}_LE"]["mean"])
            report["normalized"][f"{label}_speed80_slope"] = slope(xs, report["normalized"][f"{label}_speed80"]["mean"])

    print(json.dumps(report, indent=2, ensure_ascii=False))

    # -------- plots ------------------------------------------------------------
    series = {}
    if "AdamW_LE" in report["normalized"]:
        series["AdamW"] = (report["normalized"]["AdamW_LE"]["mean"], report["normalized"]["AdamW_LE"]["std"])
    if "DLA_fast_LE" in report["normalized"]:
        series["DLA fast"] = (report["normalized"]["DLA_fast_LE"]["mean"], report["normalized"]["DLA_fast_LE"]["std"])
    if "DLA_slow_LE" in report["normalized"]:
        series["DLA slow"] = (report["normalized"]["DLA_slow_LE"]["mean"], report["normalized"]["DLA_slow_LE"]["std"])
    plot_series(series, os.path.join(args.out, "LE_curriculum.png"),
                "LE = G(final) / G_max", "Normalized learning efficiency across development")

    series2 = {}
    if "AdamW_speed80" in report["normalized"]:
        series2["AdamW"] = (report["normalized"]["AdamW_speed80"]["mean"], report["normalized"]["AdamW_speed80"]["std"])
    if "DLA_fast_speed80" in report["normalized"]:
        series2["DLA fast"] = (report["normalized"]["DLA_fast_speed80"]["mean"], report["normalized"]["DLA_fast_speed80"]["std"])
    if "DLA_slow_speed80" in report["normalized"]:
        series2["DLA slow"] = (report["normalized"]["DLA_slow_speed80"]["mean"], report["normalized"]["DLA_slow_speed80"]["std"])
    plot_series(series2, os.path.join(args.out, "speed80_curriculum.png"),
                "1 / T80%", "Speed to 80% of the individual's own reachable gain")

    with open(os.path.join(args.out, "analysis.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}/analysis.json + plots")


if __name__ == "__main__":
    main()
