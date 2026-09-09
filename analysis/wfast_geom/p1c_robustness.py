"""P1c - cheap robustness on the P1 correlation table (no new probes).

For every predictor->outcome pair with n=12:
  * leave-one-seed-out Pearson range (is the signal outlier-driven?)
  * Benjamini-Hochberg FDR q-value over all tested pairs
Run: python analysis/wfast_geom/p1c_robustness.py --base results/analysis_wfast
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import ROOT


def pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return 0.0 if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/analysis_wfast")
    args = ap.parse_args()
    p1 = json.load(open(os.path.join(args.base, "p1_correlations.json")))
    rows = p1["rows"]
    tbl = p1["correlations"]

    out = {"loo": [], "fdr": []}
    ps = []
    for c in tbl:
        pairs = [(r[c["predictor"]], r[c["outcome"]]) for r in rows
                 if r.get(c["predictor"]) is not None and r.get(c["outcome"]) is not None]
        if len(pairs) < 10:
            continue
        x = [a for a, _ in pairs]
        y = [b for _, b in pairs]
        loo = []
        for i in range(len(pairs)):
            xv = x[:i] + x[i + 1:]
            yv = y[:i] + y[i + 1:]
            loo.append(pear(xv, yv))
        out["loo"].append({"predictor": c["predictor"], "outcome": c["outcome"],
                           "r_full": c["pearson"], "loo_min": round(min(loo), 4),
                           "loo_max": round(max(loo), 4),
                           "loo_all_same_sign": (max(loo) < 0) or (min(loo) > 0)})
        ps.append((c["pearson"], abs(c["pearson"]), c["perm_p"], c["predictor"], c["outcome"]))
    ps.sort(key=lambda t: t[2])
    m = len(ps)
    for i, (r_full, ra, pv, pk, ok) in enumerate(ps):
        q = pv * m / (i + 1)
        out["fdr"].append({"predictor": pk, "outcome": ok, "r": r_full,
                           "perm_p": pv, "q": round(min(q, 1.0), 4)})
    with open(os.path.join(args.base, "p1c_robustness.json"), "w") as f:
        json.dump(out, f, indent=1)
    for e in out["loo"]:
        print(f"LOO {e['predictor']:26s}->{e['outcome']:20s} r={e['r_full']:+.3f} "
              f"[{e['loo_min']:+.3f},{e['loo_max']:+.3f}] same_sign={e['loo_all_same_sign']}")
    print("--- FDR (q<0.1): ---")
    for e in out["fdr"]:
        if e["q"] < 0.1:
            print(f"  q={e['q']:.3f} {e['predictor']}->{e['outcome']} (r={e['r']:+.3f})")
    print(f"saved {args.base}/p1c_robustness.json")


if __name__ == "__main__":
    main()
