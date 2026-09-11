"""summary_leakgain.py -- analysis of the leak/gain disentangling experiment.

Reads the per-seed JSON written by leak_gain.py and answers, in order:

  Q1  does fast_decay control the W_fast memory horizon?
  Q2  does the memory horizon change normalized selectivity?
  Q3  does lambda change selectivity, or only magnitude?
      -- the central distinction: every normalized metric (cv, top-k mass,
         normalized entropy, gini, hhi) is exactly scale-invariant, so a sleep
         lambda that only multiplies W_fast cannot move any of them.  This script
         correlates the two metric families directly and reports the slope of
         normalized selectivity on ||W_fast||.
  Q4  does the native-coordinate writeback advantage B = final_ppl(shufwrite)
      - final_ppl(direct) vary systematically with (fast_decay, lambda)?
  Q5  (with --cl) does selectivity correspond to lower forgetting / higher retention?

Nothing here selects a "best" configuration before reporting the mechanism.
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import statistics as st


# ---------------------------------------------------------------- statistics
def boot_ci(v, B=10000, seed=11):
    if len(v) < 2:
        return (float("nan"), float("nan"))
    r = random.Random(seed)
    k = len(v)
    ms = sorted(st.mean([v[r.randrange(k)] for _ in range(k)]) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def paired(v):
    """mean, sd, t, CI, Cohen's dz, sign counts for a paired contrast."""
    n = len(v)
    if n < 2:
        return {"n": n, "mean": float("nan")}
    m, s = st.mean(v), st.stdev(v)
    lo, hi = boot_ci(v)
    return {"n": n, "mean": m, "sd": s, "t": m / (s / math.sqrt(n)) if s > 0 else float("nan"),
            "ci": [lo, hi], "dz": m / s if s > 0 else float("nan"),
            "pos": sum(1 for x in v if x > 0), "neg": sum(1 for x in v if x < 0)}


def corr(a, b):
    a = [x for x, y in zip(a, b) if x == x and y == y]
    b = [y for x, y in zip(a, b) if y == y]
    if len(a) < 3:
        return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else float("nan")


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    return corr(rank(a), rank(b))


# ---------------------------------------------------------------- loading
def load(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            out.append(json.load(f))
    return out


def cell_key(fd, lam):
    return f"fd{fd:g}|lam{lam:g}"


def collect(runs):
    """(fd, lam, variant) -> list of per-seed records."""
    acc = {}
    for run in runs:
        for ck, cell in run.get("cells", {}).items():
            for variant in ("direct", "shufwrite", "nocons"):
                r = cell.get(variant)
                if not r:
                    continue
                fd, lam = r["fast_decay"], r["lambda"]
                acc.setdefault((fd, lam, variant), []).append(
                    {"seed": run["seed"], **r})
    return acc


MAG = ("w_fast_norm", "abs_mean", "abs_std", "abs_max")
SEL = ("cv", "top1pct_mass", "top5pct_mass", "top10pct_mass",
       "normalized_entropy", "gini", "hhi")


def mean_metric(recs, group, key):
    vals = [r.get(group, {}).get(key) for r in recs]
    vals = [v for v in vals if v is not None and v == v]
    return st.mean(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/cloud/dla_leakgain/leakgain*_gpt2_s*.json")
    ap.add_argument("--out", default="results/cloud/dla_leakgain/summary")
    ap.add_argument("--cl-glob", default="results/cloud/dla_cl/gpt2/cl_gpt2_s*.json")
    ap.add_argument("--md", default=None,
                    help="also write a markdown report to this path")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Tee everything the analysis prints into a markdown report, so the written
    # report cannot drift from the numbers: it IS the analysis output.
    _lines = []
    import builtins
    _print = builtins.print   # `def print` below makes `print` a local name here

    def print(*args, **kw):        # noqa: A001  (deliberate shadow inside main)
        _print(*args, **kw)
        _lines.append(" ".join(str(x) for x in args))

    runs = load(a.glob)
    if not runs:
        raise SystemExit(f"no runs matched {a.glob}")
    acc = collect(runs)
    fds = sorted({k[0] for k in acc})
    lams = sorted({k[1] for k in acc})
    seeds = sorted({r["seed"] for r in runs})
    print(f"  loaded {len(runs)} runs, seeds={len(seeds)}, "
          f"fast_decay={fds}, lambda={lams}")

    # ---------------------------------------------------------- raw long table
    rows = []
    for (fd, lam, variant), recs in sorted(acc.items()):
        for r in recs:
            row = {"fast_decay": fd, "lambda": lam, "variant": variant,
                   "seed": r["seed"], "pre": r["pre"], "final_ppl": r["final_ppl"],
                   "gain40": r["gain40"], "drop": r.get("drop")}
            for k in MAG:
                row[k] = (r.get("magnitude") or {}).get(k)
            for k in SEL:
                row[k] = (r.get("selectivity") or {}).get(k)
            ts = r.get("timescale") or {}
            for k, v in ts.items():
                row[f"ts_{k}"] = v
            lg = r.get("lag_decay") or {}
            row["horizon_steps"] = lg.get("horizon_steps")
            row["horizon_censored"] = lg.get("censored")
            row["cos_lag1"] = lg.get("cos_lag1")
            rows.append(row)
    with open(os.path.join(a.out, "leakgain_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote leakgain_raw.csv ({len(rows)} rows)")

    def cell_rows(variant):
        out = {}
        for (fd, lam, v), recs in acc.items():
            if v != variant:
                continue
            d = {"fast_decay": fd, "lambda": lam, "n": len(recs)}
            for k in ("pre", "final_ppl", "gain40", "drop"):
                vals = [r[k] for r in recs if r.get(k) == r.get(k)]
                d[k] = st.mean(vals) if vals else float("nan")
                d[k + "_sd"] = st.stdev(vals) if len(vals) > 1 else float("nan")
            for k in MAG:
                d[k] = mean_metric(recs, "magnitude", k)
            for k in SEL:
                d[k] = mean_metric(recs, "selectivity", k)
            d["horizon_steps"] = mean_metric(recs, "lag_decay", "horizon_steps")
            ts = [r.get("timescale") or {} for r in recs]
            for k in [x for x in (ts[0].keys() if ts else []) if x.startswith("cos_")]:
                vals = [t.get(k) for t in ts if t.get(k) is not None and t.get(k) == t.get(k)]
                d[f"ts_{k}"] = st.mean(vals) if vals else float("nan")
            out[(fd, lam)] = d
        return out

    direct, shuf, nocons = cell_rows("direct"), cell_rows("shufwrite"), cell_rows("nocons")

    # ---------------------------------------------------------- B(fd, lambda)
    Bs = {}
    for key in sorted(set(direct) & set(shuf)):
        fd, lam = key
        ds = {r["seed"]: r for r in acc[(fd, lam, "direct")]}
        ss = {r["seed"]: r for r in acc[(fd, lam, "shufwrite")]}
        ns = {r["seed"]: r for r in acc.get((fd, lam, "nocons"), [])}
        common = sorted(set(ds) & set(ss))
        if not common:
            continue
        b = [ss[s]["final_ppl"] - ds[s]["final_ppl"] for s in common]
        bg = [ss[s]["gain40"] - ds[s]["gain40"] for s in common]
        pre = [ss[s]["pre"] - ds[s]["pre"] for s in common]
        bn = ([ns[s]["final_ppl"] - ds[s]["final_ppl"] for s in common
               if s in ns] if ns else [])
        Bs[key] = {"fast_decay": fd, "lambda": lam,
                   "B_shuf": paired(b), "B_shuf_gain": paired(bg),
                   "pre_imbalance": paired(pre),
                   "B_nocons": paired(bn) if bn else None,
                   "horizon": direct[key]["horizon_steps"],
                   "w_fast_norm": direct[key]["w_fast_norm"],
                   "cv": direct[key]["cv"], "gini": direct[key]["gini"],
                   "top1": direct[key]["top1pct_mass"]}

    # ---------------------------------------------------------- report
    def block(title):
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)

    block("Q3 (CENTRAL)  does lambda change selectivity, or only magnitude?")
    print("  Every normalized metric is scale-invariant by construction, so a pure")
    print("  scaling by lambda must leave cv/gini/top1/entropy exactly unchanged.")
    print(f"\n  {'fd':>7s} {'lam':>6s} {'|W_fast|':>10s} {'cv':>9s} {'gini':>9s} "
          f"{'top1%':>8s} {'nent':>9s}")
    for fd in fds:
        for lam in lams:
            k = (fd, lam)
            if k not in direct:
                continue
            d = direct[k]
            print(f"  {fd:>7g} {lam:>6g} {d['w_fast_norm']:>10.3f} {d['cv']:>9.5f} "
                  f"{d['gini']:>9.5f} {d['top1pct_mass']:>8.4f} "
                  f"{d['normalized_entropy']:>9.5f}")
    # within each fd, correlate magnitude vs normalized selectivity across lambda
    print("\n  within each fast_decay, across lambda:")
    print(f"  {'fd':>7s} {'Δ|W| rel':>10s} {'Δcv rel':>10s} {'corr(|W|,cv)':>14s} "
          f"{'corr(|W|,gini)':>15s}")
    for fd in fds:
        ks = [(fd, l) for l in lams if (fd, l) in direct]
        if len(ks) < 2:
            continue
        nrm = [direct[k]["w_fast_norm"] for k in ks]
        cvs = [direct[k]["cv"] for k in ks]
        gis = [direct[k]["gini"] for k in ks]
        dn = (max(nrm) - min(nrm)) / min(nrm) if min(nrm) else float("nan")
        dc = (max(cvs) - min(cvs)) / min(cvs) if min(cvs) else float("nan")
        print(f"  {fd:>7g} {dn:>10.4f} {dc:>10.5f} "
              f"{corr(nrm, cvs):>14.3f} {corr(nrm, gis):>15.3f}")
    fit = [(direct[k]["w_fast_norm"], direct[k]["cv"]) for k in direct]
    r_mag_cv = corr([x for x, _ in fit], [y for _, y in fit])
    print(f"\n  pooled across all {len(fit)} cells: corr(||W_fast||, cv) = {r_mag_cv:+.3f}")
    # The verdict has to follow the data, not a preset story.  What actually
    # separates "lambda is pure gain" from "lambda also reshapes the store" is the
    # RATIO of the two relative changes, not the sign of the correlation: a pure
    # scaling knob gives a large relative change in ||W_fast|| and ~0 in cv.
    rel = []
    for fd in fds:
        ks = [(fd, l) for l in lams if (fd, l) in direct]
        if len(ks) < 2:
            continue
        nrm = [direct[k]["w_fast_norm"] for k in ks]
        cvs = [direct[k]["cv"] for k in ks]
        if min(nrm) > 0 and min(cvs) > 0:
            rel.append(((max(nrm) - min(nrm)) / min(nrm), (max(cvs) - min(cvs)) / min(cvs)))
    for dn, dc in rel:
        ratio = dn / dc if dc > 0 else float("inf")
        print(f"  relative change across lambda: ||W_fast|| {dn*100:+.1f}%  vs  cv {dc*100:+.2f}%"
              f"   -> magnitude moves {ratio:.1f}x more than the normalized distribution")
        if ratio > 10:
            print("  => lambda is predominantly a GAIN on an existing structure: it sets the")
            print("     STRENGTH of the selective writeback far more than the EMERGENCE of")
            print("     selectivity.  Report both families separately and never quote the")
            print("     normalized metrics as if lambda had produced them.")
        else:
            print("  => lambda reshapes the store, not just its scale; the normalized metrics")
            print("     must not be dismissed as pure gain.")

    block("Q1  does fast_decay control the memory horizon?")
    print(f"  {'fd':>7s} {'lag-1e horizon (steps)':>24s} {'censored':>9s} "
          f"{'cos(W_final,stage1)':>20s} {'cos(W_final,stage2)':>20s}")
    for fd in fds:
        ks = [(fd, l) for l in lams if (fd, l) in direct]
        if not ks:
            continue
        hz = st.mean([direct[k]["horizon_steps"] for k in ks
                      if direct[k]["horizon_steps"] == direct[k]["horizon_steps"]])
        cen = direct[ks[0]].get("horizon_censored")
        c1 = st.mean([direct[k].get("ts_cos_stage1", float("nan")) for k in ks])
        c2 = st.mean([direct[k].get("ts_cos_stage2", float("nan")) for k in ks])
        print(f"  {fd:>7g} {hz:>24.1f} {str(cen):>9s} {c1:>20.4f} {c2:>20.4f}")
    hs = [direct[k]["horizon_steps"] for k in direct]
    xs = [direct[k]["fast_decay"] for k in direct]
    print(f"\n  spearman(fast_decay, horizon) = {spearman(xs, hs):+.3f}  "
          f"(expect NEGATIVE: a bigger leak forgets sooner)")
    c1s = [direct[k].get("ts_cos_stage1", float("nan")) for k in direct]
    print(f"  spearman(fast_decay, cos with oldest stage) = {spearman(xs, c1s):+.3f}")

    block("Q2  does the horizon change normalized selectivity?")
    for m in ("cv", "gini", "top1pct_mass", "normalized_entropy"):
        v = [direct[k][m] for k in direct]
        print(f"  spearman(horizon, {m:20s}) = {spearman(hs, v):+.3f}")

    block("Q4  B = final_ppl(shufwrite) - final_ppl(direct)   (>0 => direct better)")
    print(f"  {'fd':>7s} {'lam':>6s} {'B':>9s} {'sd':>8s} {'t':>7s} {'dz':>7s} "
          f"{'pos/n':>7s} {'95% CI':>20s}")
    for key in sorted(Bs):
        b = Bs[key]["B_shuf"]
        if b["n"] < 2:
            continue
        ci = f"[{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}]"
        print(f"  {key[0]:>7g} {key[1]:>6g} {b['mean']:>+9.4f} {b['sd']:>8.4f} "
              f"{b['t']:>+7.2f} {b['dz']:>+7.2f} {str(b['pos'])+'/'+str(b['n']):>7s} {ci:>20s}")
    bm = [Bs[k]["B_shuf"]["mean"] for k in Bs if Bs[k]["B_shuf"]["n"] > 1]
    if bm:
        print(f"\n  across {len(bm)} cells: B range [{min(bm):+.3f}, {max(bm):+.3f}], "
              f"positive in {sum(1 for x in bm if x > 0)}/{len(bm)} cells")
    else:
        print(f"\n  only single-seed cells so far ({len(Bs)} cells); paired statistics "
              f"need >=2 seeds -- re-run once more seeds land")
    bk = sorted(Bs, key=lambda k: -Bs[k]["B_shuf"]["mean"])
    if bk and Bs[bk[0]]["B_shuf"]["n"] > 1:
        print(f"  largest B at fast_decay={bk[0][0]:g}, lambda={bk[0][1]:g} "
              f"(but treat any single-cell maximum as exploratory, not a result)")

    # ---------------------------------------------------------- CL linkage (Q5)
    cl = load(a.cl_glob)
    if cl:
        block("Q5  link to continual learning")
        dla = {}
        for run in cl:
            for label, m in run.get("methods", {}).items():
                if label.startswith("dla"):
                    dla.setdefault(label, []).append(m)
        print(f"  {'dla variant':>22s} {'forward':>10s} {'retention':>11s} "
              f"{'forgetting':>12s}")
        for label in sorted(dla):
            recs = dla[label]
            f_ = st.mean([r["forward_mean"] for r in recs])
            r_ = st.mean([r["retention_mean"] for r in recs])
            g_ = st.mean([r.get("forgetting_mean", float("nan")) for r in recs])
            print(f"  {label:>22s} {f_:>+10.4f} {r_:>11.4f} {g_:>+12.4f}")
        if dla:
            print("\n  join B(fd,lambda) against the CL point of the same config:")
            joined = []
            for label, recs in dla.items():
                parts = dict(p.split("=") for p in label.split("@")[1].split(",")) \
                    if "@" in label and "=" in label else {}
                try:
                    fd = float(parts["fd"]); lam = float(parts.get("g", parts.get("lam", 1.0)))
                except (KeyError, ValueError):
                    continue
                key = (fd, lam)
                if key in Bs and Bs[key]["B_shuf"]["n"] > 1:
                    joined.append((Bs[key]["B_shuf"]["mean"],
                                   st.mean([r["forgetting_mean"] for r in recs]),
                                   st.mean([r["retention_mean"] for r in recs]),
                                   st.mean([r["forward_mean"] for r in recs]), fd, lam))
            if joined:
                print(f"  {'B':>9s} {'forgetting':>11s} {'retention':>10s} "
                      f"{'forward':>9s} {'fd':>7s} {'lam':>6s}")
                for b, fg, rt, fw, fd, lam in joined:
                    print(f"  {b:>+9.4f} {fg:>11.4f} {rt:>10.4f} {fw:>+9.4f} "
                          f"{fd:>7g} {lam:>6g}")
                print(f"\n  corr(B, forgetting) = {corr([j[0] for j in joined], [j[1] for j in joined]):+.3f}")
                print(f"  corr(B, retention)  = {corr([j[0] for j in joined], [j[2] for j in joined]):+.3f}")
                print(f"  corr(B, forward)    = {corr([j[0] for j in joined], [j[3] for j in joined]):+.3f}")

    with open(os.path.join(a.out, "leakgain_cells.json"), "w") as f:
        json.dump({"fds": fds, "lambdas": lams, "n_seeds": len(seeds),
                   "direct": {f"{k[0]:g}|{k[1]:g}": v for k, v in direct.items()},
                   "B": {f"{k[0]:g}|{k[1]:g}": {kk: vv for kk, vv in v.items()}
                         for k, v in Bs.items()}}, f, indent=1)
    print(f"\n  wrote leakgain_cells.json")

    if a.md:
        os.makedirs(os.path.dirname(os.path.abspath(a.md)), exist_ok=True)
        with open(a.md, "w") as f:
            f.write("# Leak / gain sweep -- mechanism report\n\n")
            f.write("**Disentangling the emergence and expression of parameter-level "
                    "selective writeback in leaky fast-weight dynamics.**\n\n")
            f.write("Auto-generated by `analysis/wfast_geom/summary_leakgain.py`; every "
                    "number below is the analysis output verbatim.\n\n")
            f.write(f"- runs: {len(runs)} JSON files, {len(seeds)} seeds\n")
            f.write(f"- fast_decay grid: {fds}\n- lambda grid: {lams}\n")
            f.write(f"- variants per cell: direct / shufwrite / nocons (energy-matched "
                    f"shuffle; only the coordinate allocation differs)\n\n")
            f.write("```\n")
            f.write("\n".join(_lines))
            f.write("\n```\n")
        print(f"  wrote {a.md}")


if __name__ == "__main__":
    main()
