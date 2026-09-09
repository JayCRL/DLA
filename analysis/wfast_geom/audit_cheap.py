"""Mechanism audit - cheap read-only diagnostics (A3 confound controls, B2
sleep-write magnitude decomposition, per-stage Q/W_fast/W_slow trace from the
Stage 5.5c meta_trace). No new model runs.

Run: python analysis/wfast_geom/audit_cheap.py --base results/analysis_wfast
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

from common import canonical_keys, load_body_ckpt, all_seeds, GROUPS


def pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return 0.0 if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])


def partial(x, y, zs):
    # partial corr of x,y after regressing each on zs
    X = np.column_stack([np.asarray(z, float) for z in zs] + [np.ones(len(x))])
    def resid(v):
        v = np.asarray(v, float)
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        return v - X @ beta
    rx = resid(x); ry = resid(y)
    return pear(rx, ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/analysis_wfast")
    args = ap.parse_args()
    B = args.base
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    out = {}
    # ---------------- A3: partial-confound controls for cos0 -> gain ---------
    p1 = json.load(open(os.path.join(B, "p1_correlations.json")))
    geom = json.load(open(os.path.join(B, "geometry_summary.json")))
    rows = {r["seed"]: r for r in p1["rows"]}
    seeds = list(range(12))

    def col(r, k):
        return r.get(k)

    gain = [rows[s]["out_HE/HE_gain40"] for s in seeds]
    cos0 = [rows[s]["tr_HE/HE_cos0"] for s in seeds]
    gnorm = [rows[s]["tr_HE/HE_g0_norm_gf"] for s in seeds]
    dnorm = [rows[s]["delta_norm"] for s in seeds]
    rel_d = [rows[s]["rel_delta_HE"] for s in seeds]
    ceh = [rows[s]["cos_eh_he"] for s in seeds]
    # module cos0 from replays
    mc = {"emb": [], "attn": [], "mlp": []}
    for s in seeds:
        d = json.load(open(os.path.join(B, "replays", "seeds", f"seed{s}_HE_HE.json")))
        c0 = d["steps"][0]["cos_d"]
        for g in mc:
            mc[g].append(c0.get(g))
    a3 = {"zero_order": {}, "partial_vs_gain": {}}
    for name, v in (("cos0", cos0), ("g0_norm", gnorm), ("delta_norm", dnorm),
                    ("cos_emb", mc["emb"]), ("cos_attn", mc["attn"]), ("cos_mlp", mc["mlp"])):
        a3["zero_order"][name] = round(pear(v, gain), 3)
    controls = {"delta_norm": dnorm, "rel_delta": rel_d, "cos_eh_he": ceh, "g0_norm": gnorm}
    for name, v in (("cos0", cos0), ("cos_attn", mc["attn"]), ("cos_mlp", mc["mlp"])):
        a3["partial_vs_gain"][name] = {}
        for cn, z in controls.items():
            a3["partial_vs_gain"][name][cn] = round(partial(v, gain, [z]), 3)
        # joint controls
        a3["partial_vs_gain"][name]["all_norm_confounds"] = round(
            partial(v, gain, [dnorm, rel_d, ceh, gnorm]), 3)
    # how much of cos0 variance is explained by norm confounds?
    X = np.column_stack([np.asarray(dnorm), np.asarray(rel_d), np.asarray(ceh), np.asarray(gnorm),
                         np.ones(12)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(cos0), rcond=None)
    pred = X @ beta
    r2 = 1 - float(np.sum((np.asarray(cos0) - pred) ** 2) / np.sum((np.asarray(cos0) - np.mean(cos0)) ** 2))
    a3["confound_R2_of_cos0"] = round(r2, 3)
    out["A3"] = a3

    # ---------------- B2: sleep-write magnitude decomposition ----------------
    rows_b2 = []
    for s in seeds:
        for order, name in (("easy_hard", "EH"), ("hard_easy", "HE")):
            ck = load_body_ckpt(s, order)
            keys = canonical_keys(ck["store"].keys())
            qn = math.sqrt(sum(float(ck["store"][k]["q"].pow(2).sum()) for k in keys))
            wn = math.sqrt(sum(float(ck["store"][k]["w_fast"].pow(2).sum()) for k in keys))
            slow_n = math.sqrt(sum(float(ck["model"][k + ".weight"].pow(2).sum()) for k in keys))
            phi = ck.get("phi", {})
            beta = float(phi.get("consolidate_beta", np.nan))
            cfd = float(phi.get("consolidate_fast_direct", np.nan))
            rows_b2.append({"seed": s, "order": name, "Q_norm": qn, "Wfast_norm": wn,
                            "Wslow_norm": slow_n, "beta": beta, "cfd": cfd,
                            "Q_write_est": beta * qn, "direct_write_est": cfd * wn})
    qn_m = np.mean([r["Q_norm"] for r in rows_b2]); wn_m = np.mean([r["Wfast_norm"] for r in rows_b2])
    slow_m = np.mean([r["Wslow_norm"] for r in rows_b2])
    qw = np.mean([r["Q_write_est"] for r in rows_b2]); dw = np.mean([r["direct_write_est"] for r in rows_b2])
    out["B2"] = {"mean_Q_norm": qn_m, "mean_Wfast_norm": wn_m, "mean_Wslow_norm": slow_m,
                 "mean_Q_ratio_wf": qn_m / wn_m,
                 "est_write_Q_per_sleep": qw, "est_write_direct_per_sleep": dw,
                 "write_Q_to_direct_ratio": qw / dw if dw > 0 else None,
                 "est_total_write_per_sleep_rel_Wslow": (qw + dw) / slow_m}
    print("B2:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out["B2"].items()}))

    # ---------------- per-stage Q/W_fast/W_slow trace (5.5c, seeds 0-2) ------
    stage_rows = []
    for s in range(3):
        f = os.path.join(ROOT, "results", "stage55c", f"seed{s}", "causality.json")
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        mt = j.get("meta_trace", {})
        for key in mt:
            if "meta" not in key:
                continue
            tr = mt[key].get("trace", [])
            for e in tr:
                st = e.get("state", {})
                stage_rows.append({"seed": s, "order": key.split("_")[1] + "_" + key.split("_")[2],
                                   "stage": e["stage"], "Q": st.get("Q_norm"), "wf": st.get("W_fast_norm"),
                                   "slow": st.get("W_slow_norm")})
    if stage_rows:
        agg = {}
        for e in stage_rows:
            k = (e["stage"])
            agg.setdefault(k, []).append(e)
        trace = []
        for k in sorted(agg):
            es = agg[k]
            q = [e["Q"] for e in es if e["Q"] is not None]
            wf = [e["wf"] for e in es if e["wf"] is not None]
            slow = [e["slow"] for e in es if e["slow"] is not None]
            trace.append({"stage": k, "Q_mean": float(np.mean(q)), "wf_mean": float(np.mean(wf)),
                          "slow_mean": float(np.mean(slow))})
        out["stage_trace"] = trace
        print("stage_trace:", json.dumps([{k: round(v, 3) if isinstance(v, float) else v for k, v in t.items()} for t in trace]))
    # Q-norm history divergence vs cross-seed spread at each stage (SNR)
    if stage_rows:
        for stage in sorted(set(e["stage"] for e in stage_rows)):
            he = [e for e in stage_rows if e["stage"] == stage and e["order"].startswith("hard")]
            eh = [e for e in stage_rows if e["stage"] == stage and e["order"].startswith("easy")]
            if len(he) >= 2 and len(eh) >= 2:
                qhe = [e["Q"] for e in he]; qeh = [e["Q"] for e in eh]
                sig = abs(float(np.mean(qhe) - np.mean(qeh)))
                noise = float(np.std(qhe + qeh))
                out.setdefault("Q_snr_by_stage", {})[stage] = round(sig / noise, 3) if noise > 0 else None
    with open(os.path.join(B, "audit_cheap.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("A3 partials:", json.dumps(a3["partial_vs_gain"], indent=0))
    print(f"saved {B}/audit_cheap.json")


if __name__ == "__main__":
    main()
