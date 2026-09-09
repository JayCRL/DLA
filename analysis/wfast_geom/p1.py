"""P1 - correlate ΔW_fast geometry & gradient-trajectory stats with adaptation
outcomes (gain@40 / LE_D / T80_D) per seed (n=12), with permutation tests and
mechanism-neutral null controls.

Predictors (per seed):
  geom   : |ΔW_fast| (global), rel |Δ|/|W_fast|, group energy shares, cos(EH,HE)
  replay : trajectory stats per arm: ||g||, ||g_f||, cos(g_f,Δ) at step 1 and
           mean |cos| & mean signed cos over the 40-step D probe
Outcomes (from the archived p0 JSON, canonical):
  gain@40 (= curve[-1].gain), LE_D, T80_D per arm

Nulls:
  (a) permutation of seed labels (5000) for every correlation
  (b) shuffled-Δ null: cos(g0, shuffle(Δ)) replacing true Δ (within-matrix
      permutation, norm preserved) -> null correlation distribution
  (c) cross-seed mismatched-Δ null: cos(g0, W_fast(HE_j)-W_fast(EH_j)), j!=i

No mechanism is assumed; all stats are reported with direction + effect size.

Run: python analysis/wfast_geom/p1.py --base results/analysis_wfast
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random

import numpy as np

from common import ROOT


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    s = np.argsort(inv, kind="mergesort")
    starts = np.cumsum(counts)
    for i, c in enumerate(counts):
        if c > 1:
            ranks[inv == i] = (starts[i] - c + 1 + starts[i]) / 2.0
    return ranks


def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def perm_p(r_obs, x, y, n=5000, seed=0):
    rng = random.Random(seed)
    xs = list(x)
    cnt = 0
    for _ in range(n):
        ys = list(y)
        rng.shuffle(ys)
        r = _pearson(xs, ys)
        if abs(r) >= abs(r_obs):
            cnt += 1
    return (cnt + 1) / (n + 1)


def load_p0(seed):
    p = ROOT / "results" / "stage55e" / "seeds" / f"seed{seed}.json"
    return json.load(open(p))["results"]


def arm_outcome(res, tag):
    g = res[tag]
    return {"gain40": g["curve"][-1]["gain"], "LE_D": g["LE_D"], "T80_D": g["T80_D"]}


def load_replays(base, seed, tag):
    from common import slug
    p = os.path.join(base, "replays", "seeds", f"seed{seed}_{slug(tag)}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def traj_stats(replay):
    """Summaries of the 40-step gradient trajectory (per arm)."""
    steps = replay["steps"]
    gfn = [s["global"]["gfn"] for s in steps]
    gn = [s["global"]["gn"] for s in steps]
    cos = [s["cos_d"]["global"] for s in steps]
    cos = [c for c in cos if c is not None]
    out = {
        "g0_norm_gf": gfn[0], "g_mean_norm_gf": float(np.mean(gfn)),
        "cos0": cos[0] if cos else None,
        "cos_mean_signed": float(np.mean(cos)) if cos else None,
        "cos_mean_abs": float(np.mean([abs(c) for c in cos])) if cos else None,
        "cos_min_abs": float(min(abs(c) for c in cos)) if cos else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/analysis_wfast")
    ap.add_argument("--nperm", type=int, default=5000)
    args = ap.parse_args()

    seeds = list(range(12))
    geom = json.load(open(os.path.join(args.base, "geometry_summary.json")))

    # outcomes per arm
    rows = []
    for s in seeds:
        res = load_p0(s)
        row = {"seed": s}
        for tag in ("EH/EH", "HE/HE", "EH_body+HE_fast", "HE_body+EH_fast"):
            oc = arm_outcome(res, tag)
            row[f"out_{tag}"] = oc
            for metric in ("gain40", "LE_D", "T80_D"):
                row[f"out_{tag}_{metric}"] = oc[metric]
        g = geom["per_seed"][str(s)]
        row["delta_norm"] = g["delta"]["norm"]
        row["rel_delta_HE"] = g["delta"]["rel_norm_vs_HE"]
        row["rel_delta_EH"] = g["delta"]["rel_norm_vs_EH"]
        row["cos_eh_he"] = g["delta"]["cos_EH_HE_global"]
        row["share_emb"] = g["delta"]["group_energy_share"]["emb"]
        row["share_attn"] = g["delta"]["group_energy_share"]["attn"]
        row["share_mlp"] = g["delta"]["group_energy_share"]["mlp"]
        row["eh_gate_mean"] = g["EH"]["gate_mean"]
        row["he_gate_mean"] = g["HE"]["gate_mean"]
        for tag in ("EH/EH", "HE/HE", "EH_body+HE_fast", "HE_body+EH_fast"):
            rep = load_replays(args.base, s, tag)
            if rep is None:
                continue
            ts = traj_stats(rep)
            for k, v in ts.items():
                row[f"tr_{tag}_{k}"] = v
            row[f"parity_{tag}"] = (rep.get("parity") or {}).get("max_ppl_diff")
        rows.append(row)
    out = {"rows": rows}

    # ---------------- correlation table ---------------------------------
    def vals(key):
        return [r[key] for r in rows if r.get(key) is not None]

    # predictor groups: (predictor_key, outcome_key)
    specs = [
        ("delta_norm", "out_HE/HE_gain40"),
        ("delta_norm", "out_HE/HE_LE_D"),
        ("delta_norm", "out_EH/EH_gain40"),
        ("rel_delta_HE", "out_HE/HE_gain40"),
        ("cos_eh_he", "out_HE/HE_gain40"),
        ("cos_eh_he", "out_HE/HE_LE_D"),
        ("share_mlp", "out_HE/HE_gain40"),
        ("tr_HE/HE_cos0", "out_HE/HE_gain40"),
        ("tr_HE/HE_cos0", "out_HE/HE_LE_D"),
        ("tr_HE/HE_cos_mean_abs", "out_HE/HE_gain40"),
        ("tr_HE/HE_g0_norm_gf", "out_HE/HE_gain40"),
        ("tr_HE/HE_g0_norm_gf", "out_HE/HE_LE_D"),
        ("tr_EH/EH_cos0", "out_EH/EH_gain40"),
        ("tr_EH/EH_cos0", "out_EH/EH_LE_D"),
        # destructive contrast vs geometry of its two carriers
        ("delta_norm", "contrast_HE_delta"),
        ("tr_HE/HE_cos_mean_abs", "contrast_HE_delta"),
    ]
    # add destructive-contrast outcome column
    for r in rows:
        r["contrast_HE_delta"] = (r["out_HE_body+EH_fast"]["gain40"] if r.get("out_HE_body+EH_fast") else 0) - r["out_HE/HE"]["gain40"]
        r["contrast_EH_plus"] = (r["out_EH_body+HE_fast"]["gain40"] if r.get("out_EH_body+HE_fast") else 0) - r["out_EH/EH"]["gain40"]

    corr_tbl = []
    for pk, ok in specs:
        x = vals(pk)
        y = vals(ok)
        if len(x) < 8 or len(y) < 8:
            continue
        # align on seeds present in both
        pr = [(r[pk], r[ok]) for r in rows if r.get(pk) is not None and r.get(ok) is not None]
        if len(pr) < 8:
            continue
        x = [a for a, _ in pr]
        y = [b for _, b in pr]
        r_obs = _pearson(x, y)
        sp = _spearman(x, y)
        p = perm_p(r_obs, x, y, n=args.nperm)
        corr_tbl.append({"predictor": pk, "outcome": ok, "n": len(pr),
                         "pearson": round(r_obs, 4), "spearman": round(sp, 4), "perm_p": p})
    out["correlations"] = corr_tbl
    for c in corr_tbl:
        print(f"{c['predictor']:32s} -> {c['outcome']:22s} r={c['pearson']:+.3f} "
              f"rho={c['spearman']:+.3f} perm_p={c['perm_p']:.3f} (n={c['n']})")

    # ---------------- aggregate trajectory means --------------------------
    agg = {}
    for tag in ("HE/HE", "EH/EH"):
        keys = [f"tr_{tag}_cos0", f"tr_{tag}_cos_mean_signed", f"tr_{tag}_cos_mean_abs",
                f"tr_{tag}_g0_norm_gf"]
        agg[tag] = {}
        for k in keys:
            v = vals(k)
            agg[tag][k] = {"mean": float(np.mean(v)) if v else None,
                           "std": float(np.std(v)) if v else None,
                           "seeds": len(v)}
    out["traj_agg"] = agg

    with open(os.path.join(args.base, "p1_correlations.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.base}/p1_correlations.json")


if __name__ == "__main__":
    main()
