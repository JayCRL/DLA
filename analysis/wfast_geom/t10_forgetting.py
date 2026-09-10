"""T10 forgetting audit — how much old-task knowledge is lost as tasks accumulate.

Uses ONLY recorded raw data:
  * rows[t].post_ppl   = PPL on task t's val slice immediately after finishing task t
                         (the "birth / baseline right after learning" reference)
  * end_ppl[domain_t]  = PPL on the same slice after ALL 10 tasks were learned

forgetting_t = (end_ppl_t - post_ppl_t) / post_ppl_t     (project convention:
positive = worse / forgotten, negative = improved).

Availability note (no estimation): intermediate retention checkpoints
(retention after k<10 tasks) were NOT recorded by audit_t10.py, so the
"forgetting vs number of tasks already learned" longitudinal curve cannot be
computed from existing data; we report each task's forgetting measured at the end
of the 10-task sequence, plus averages over the first k tasks (k-subsets all
measured at end of 10) and the cross-sectional view vs number of SUBSEQUENT tasks.

Outputs: t10_forgetting.json / t10_forgetting.csv / t10_forgetting.png
Run: python analysis/wfast_geom/t10_forgetting.py
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

OUT = os.path.expanduser("~/llm-lab/dla_audit_t10")
REPO_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_output", "t10_forgetting")
VARIANTS = ["direct", "nocons", "nosleep"]
SEEDS = list(range(8))


def load(v, s):
    return json.load(open(os.path.join(OUT, f"t10_seed{s}_{v}.json")))


def ms(x):
    x = np.asarray(x, float)
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else 0.0)


def tpair(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    m, s = ms(d)
    return m, s, (m / (s / np.sqrt(len(d))) if s > 0 else float("nan"))


def main():
    data = {v: [load(v, s) for s in SEEDS] for v in VARIANTS}
    domains = [r["domain"] for r in data["direct"][0]["rows"]]

    # per-seed per-task forgetting
    per = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for t in range(10):
            vals = []
            for i in range(len(SEEDS)):
                r = data[v][i]["rows"][t]
                post = r["post_ppl"]
                end = data[v][i]["end_ppl"][r["domain"]]
                vals.append((end - post) / post)
            per[v][t] = vals

    lines = []
    def P(s=""):
        lines.append(s)
        print(s)

    P("== Per-task relative forgetting at the end of the 10-task sequence "
      "(positive = forgotten, negative = improved) ==")
    P(f"{'task':<6}{'domain':<14}{'direct':>18}{'nocons':>18}{'nosleep':>18}")
    table = []
    for t in range(10):
        row = {"task": t + 1, "domain": domains[t]}
        cells = []
        for v in VARIANTS:
            m, s = ms(per[v][t])
            row[v] = {"mean": m, "sd": s}
            cells.append(f"{m*100:+.1f}%±{s*100:.1f}")
        row["direct_minus_nocons"] = tpair(per["direct"][t], per["nocons"][t])
        row["direct_minus_nosleep"] = tpair(per["direct"][t], per["nosleep"][t])
        table.append(row)
        P(f"t{t+1:<5}{domains[t]:<14}{cells[0]:>18}{cells[1]:>18}{cells[2]:>18}")

    P("\n== Paired differences on the same tasks (negative Δ = direct forgets less) ==")
    for t in range(10):
        m1, s1, tt1 = table[t]["direct_minus_nocons"]
        m2, s2, tt2 = table[t]["direct_minus_nosleep"]
        P(f"t{t+1:<3} direct-nocons Δ={m1*100:+.2f}% (t={tt1:+.2f})   "
          f"direct-nosleep Δ={m2*100:+.2f}% (t={tt2:+.2f})")

    P("\n== Averages over the first k tasks (all measured at end of 10 tasks) ==")
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    avg_rows = []
    for k in ks:
        cells = []
        row = {"k": k}
        for v in VARIANTS:
            vals = [x for t in range(k) for x in per[v][t]]
            m, s = ms(vals)
            row[v] = {"mean": m, "sd": s}
            cells.append(f"{m*100:+.1f}%")
        avg_rows.append(row)
        P(f"first {k:2d} tasks: direct {cells[0]:>8}  nocons {cells[1]:>8}  nosleep {cells[2]:>8}")

    P("\n== Cross-sectional view: forgetting of a task vs how many tasks came AFTER it ==")
    for t in range(10):
        P(f"task t{t+1} ({10-t-1} subsequent tasks): direct {ms(per['direct'][t])[0]*100:+.1f}% "
          f"| nocons {ms(per['nocons'][t])[0]*100:+.1f}% | nosleep {ms(per['nosleep'][t])[0]*100:+.1f}%")

    P("\n== MISSING DATA (not estimated) ==")
    P("- audit_t10.py recorded end_ppl only after ALL 10 tasks; there are no retention "
      "checkpoints after k<10 tasks, so the longitudinal curve "
      "'forgetting vs number of tasks already learned' cannot be computed from existing data.")
    P("- post_ppl and end_ppl were evaluated with different eval RNG seeds "
      "(730000+seed+t vs 740000+seed+t), so each per-task forgetting value carries "
      "eval-subsample noise; magnitudes <1-2% should be read with that in mind.")

    os.makedirs(REPO_OUT, exist_ok=True)
    j = {"seeds": SEEDS, "variants": VARIANTS, "per_task": table, "first_k_average": avg_rows,
         "raw_per_seed_per_task": {v: {str(t + 1): per[v][t] for t in range(10)} for v in VARIANTS}}
    with open(os.path.join(REPO_OUT, "t10_forgetting.json"), "w") as f:
        json.dump(j, f, indent=1, default=float)
    with open(os.path.join(REPO_OUT, "t10_forgetting.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "domain", "direct_mean", "direct_sd", "nocons_mean", "nocons_sd",
                    "nosleep_mean", "nosleep_sd"])
        for row in table:
            w.writerow([row["task"], row["domain"], row["direct"]["mean"], row["direct"]["sd"],
                        row["nocons"]["mean"], row["nocons"]["sd"], row["nosleep"]["mean"],
                        row["nosleep"]["sd"]])

    if plt is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        xs = np.arange(1, 11)
        for v, c in (("direct", "#2b7bba"), ("nocons", "#d9534f"), ("nosleep", "#f0ad4e")):
            means = [ms(per[v][t])[0] * 100 for t in range(10)]
            sds = [ms(per[v][t])[1] * 100 for t in range(10)]
            ax[0].errorbar(xs, means, yerr=sds, marker="o", capsize=3, label=v, color=c)
        ax[0].axhline(0, color="k", lw=0.6)
        ax[0].set_xlabel("task index (learning order)")
        ax[0].set_ylabel("relative forgetting at end of 10 tasks (%)")
        ax[0].set_title("Per-task forgetting (all measured after 10 tasks)")
        ax[0].legend(fontsize=8)
        for v, c in (("direct", "#2b7bba"), ("nocons", "#d9534f"), ("nosleep", "#f0ad4e")):
            means = [ms([x for t in range(k) for x in per[v][t]])[0] * 100 for k in ks]
            ax[1].plot(ks, means, marker="o", label=v, color=c)
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xlabel("first k tasks averaged")
        ax[1].set_ylabel("mean relative forgetting (%)")
        ax[1].set_title("Average forgetting over the first k tasks")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(REPO_OUT, "t10_forgetting.png"), dpi=140)
        P(f"\nsaved figure -> {REPO_OUT}/t10_forgetting.png")
    P(f"saved {REPO_OUT}/t10_forgetting.json + .csv")

    # headline numbers for the plain-language answer
    m1, s1 = ms([x for t in range(10) for x in per["direct"][t]])
    m2, s2 = ms([x for t in range(10) for x in per["nocons"][t]])
    m3, s3 = ms([x for t in range(10) for x in per["nosleep"][t]])
    P("\nHEADLINE (mean over all 10 tasks, end of sequence):")
    P(f"  direct  {m1*100:+.2f}% ± {s1*100:.2f}")
    P(f"  nocons  {m2*100:+.2f}% ± {s2*100:.2f}")
    P(f"  nosleep {m3*100:+.2f}% ± {s3*100:.2f}")
    with open(os.path.join(REPO_OUT, "t10_forgetting.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
