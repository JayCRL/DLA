"""figures_leakgain.py -- Figures A-F for the leak/gain disentangling experiment.

  A  fast_decay -> effective memory timescale   (survival of each history stage)
  B  fast_decay -> normalized W_fast selectivity (with magnitude on a second axis
     so an apparent selectivity change cannot be read as a size change)
  C  lambda -> writeback magnitude / influence
  D  (fast_decay, lambda) -> direct - shufwrite   B on final_ppl
  E  (fast_decay, lambda) -> forgetting / retention
  F  DLA parameter plane against the AdamW / EWC / SI / Replay Pareto frontier

Every panel is built from the JSON/CSV written by leak_gain.py and
summary_leakgain.py, so the figures cannot drift from the numbers.  Panels whose
inputs are not on disk yet are skipped with a printed reason rather than faked.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_D = "#1f77b4"   # direct
C_S = "#d62728"   # shufwrite
C_N = "#7f7f7f"   # nocons
C_M = "#2ca02c"   # magnitude


def load_runs(pattern):
    runs = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def cells(runs):
    """(fd, lam, variant) -> list of per-seed records (seed carried through)."""
    acc = {}
    for run in runs:
        for _, cell in run.get("cells", {}).items():
            for v in ("direct", "shufwrite", "nocons"):
                r = cell.get(v)
                if r:
                    acc.setdefault((r["fast_decay"], r["lambda"], v), []).append(
                        {"seed": run["seed"], **r})
    return acc


def m(recs, group, key):
    vals = [r.get(group, {}).get(key) for r in recs]
    vals = [v for v in vals if v is not None and v == v]
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0) if vals else (float("nan"), 0.0)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/cloud/dla_leakgain/leakgain*_gpt2_s*.json")
    ap.add_argument("--cl-glob", default="results/cloud/dla_cl_gain/cl_gpt2_s*.json")
    ap.add_argument("--cl-base-glob", default="results/cloud/dla_cl/gpt2/cl_gpt2_s*.json")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = load_runs(a.glob)
    if not runs:
        raise SystemExit(f"no runs matched {a.glob}")
    acc = cells(runs)
    fds = sorted({k[0] for k in acc})
    lams = sorted({k[1] for k in acc})
    n_seed = len({r["seed"] for r in runs})
    print(f"  {len(runs)} runs (n={n_seed} seeds), fast_decay={fds}, lambda={lams}")

    # ------------------------------------------------------------------ A
    fd_used = [fd for fd in fds if any((fd, l, "direct") in acc for l in lams)]
    if len(fd_used) >= 2:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for i, (lbl, color) in enumerate((("cos_stage1", C_D), ("cos_stage2", C_S))):
            ys = []
            for fd in fd_used:
                recs = [r for l in lams for r in acc.get((fd, l, "direct"), [])]
                ys.append(m(recs, "timescale", lbl)[0])
            ax[0].plot(fd_used, ys, "o-", color=color,
                       label=f"cos(W_final, stage{i+1})")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("fast_decay  (wake leak per step)")
        ax[0].set_ylabel("cosine with the final W_fast")
        ax[0].set_title("A1  how much of each stage survives")
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        # empirical lag-cosine decay curve, one line per fast_decay
        for fd in fd_used:
            recs = [r for l in lams for r in acc.get((fd, l, "direct"), [])]
            curves = [r.get("lag_decay", {}).get("lag_cos", {}) for r in recs]
            curves = [c for c in curves if c]
            if not curves:
                continue
            lags = sorted({float(k) for c in curves for k in c})
            mean = [st.mean([c[str(int(l))] for c in curves
                             if str(int(l)) in c and c[str(int(l))] == c[str(int(l))]])
                    for l in lags]
            ax[1].plot(lags, mean, "o-", label=f"fd={fd:g}")
        ax[1].axhline(1 / math.e, ls="--", c="k", lw=1, label="1/e")
        ax[1].set_xlabel("lag (wake steps)")
        ax[1].set_ylabel("mean cos(W_t, W_t+lag)")
        ax[1].set_title("A2  measured lag decay (trajectory, not the formula)")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.suptitle("Figure A -- fast_decay sets the memory timescale", y=1.02)
        save(fig, a.out, "figA_timescale")
    else:
        print("  [skip] Figure A: needs >=2 fast_decay values")

    # ------------------------------------------------------------------ B
    if len(fd_used) >= 2:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        metrics = [("cv", "CV of |W_fast|"), ("gini", "Gini"), ("top1pct_mass", "top-1% mass")]
        for j, (key, lbl) in enumerate(metrics):
            ys = [m([r for l in lams for r in acc.get((fd, l, "direct"), [])],
                    "selectivity", key)[0] for fd in fd_used]
            ax[j].plot(fd_used, ys, "o-", color=C_S)
            ax[j].set_xscale("log")
            ax[j].set_xlabel("fast_decay")
            ax[j].set_ylabel(lbl)
            ax[j].set_title(f"B{j+1}  {lbl}  (scale-invariant)")
            ax[j].grid(alpha=0.3)
            ax2 = ax[j].twinx()
            nrm = [m([r for l in lams for r in acc.get((fd, l, "direct"), [])],
                     "magnitude", "w_fast_norm")[0] for fd in fd_used]
            ax2.plot(fd_used, nrm, "s--", color=C_M, alpha=0.7)
            ax2.set_ylabel("||W_fast||", color=C_M)
            ax2.tick_params(axis="y", colors=C_M)
        fig.suptitle("Figure B -- selectivity vs fast_decay (solid) with magnitude "
                     "(green dashed): the two move together, so fd cannot separate them",
                     y=1.03)
        save(fig, a.out, "figB_selectivity_fd")
    else:
        print("  [skip] Figure B: needs >=2 fast_decay values")

    # ------------------------------------------------------------------ C
    lam_used = [l for l in lams if any((fd, l, "direct") in acc for fd in fds)]
    if len(lam_used) >= 2:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        base_fd = max(fds)
        for j, (key, grp, lbl) in enumerate((
                ("w_fast_norm", "magnitude", "||W_fast||"),
                ("cv", "selectivity", "CV of |W_fast|  (scale-invariant)"),
                ("gini", "selectivity", "Gini  (scale-invariant)"))):
            ys, es = zip(*[m(acc.get((base_fd, l, "direct"), []), grp, key)
                           for l in lam_used])
            color = C_M if grp == "magnitude" else C_S
            ax[j].errorbar(lam_used, ys, yerr=es, fmt="o-", color=color, capsize=3)
            ax[j].set_xlabel("lambda  (sleep W_fast gain)")
            ax[j].set_ylabel(lbl)
            ax[j].set_title(f"C{j+1}  {lbl}")
            ax[j].grid(alpha=0.3)
        fig.suptitle(f"Figure C -- lambda scales the store (C1) but the normalized "
                     f"distribution is flat (C2,C3)  [fast_decay={base_fd:g}]", y=1.03)
        save(fig, a.out, "figC_lambda_gain")
    else:
        print("  [skip] Figure C: needs >=2 lambda values")

    # ------------------------------------------------------------------ D
    keys = sorted({(fd, l) for (fd, l, v) in acc if v == "direct"})
    Bs = {}
    for fd, lam in keys:
        d = {r["seed"]: r for r in acc.get((fd, lam, "direct"), [])}
        s = {r["seed"]: r for r in acc.get((fd, lam, "shufwrite"), [])}
        common = sorted(set(d) & set(s))
        if common:
            Bs[(fd, lam)] = [s[x]["final_ppl"] - d[x]["final_ppl"] for x in common]
    if Bs:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        cmap = plt.get_cmap("viridis")
        for i, fd in enumerate(sorted({k[0] for k in Bs})):
            xs = sorted({k[1] for k in Bs if k[0] == fd})
            ys = [st.mean(Bs[(fd, l)]) for l in xs]
            es = [st.stdev(Bs[(fd, l)]) if len(Bs[(fd, l)]) > 1 else 0 for l in xs]
            ax[0].errorbar(xs, ys, yerr=es, fmt="o-", capsize=3,
                           color=cmap(i / max(1, len({k[0] for k in Bs}) - 1)),
                           label=f"fd={fd:g}")
        ax[0].axhline(0, ls="--", c="k", lw=1)
        ax[0].set_xlabel("lambda")
        ax[0].set_ylabel("B = final_ppl(shuf) - final_ppl(direct)")
        ax[0].set_title("D1  B vs lambda   (>0 => direct wins)")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        # heat map over (fast_decay, lambda)
        ufd = sorted({k[0] for k in Bs})
        ulam = sorted({k[1] for k in Bs})
        if len(ufd) >= 2 and len(ulam) >= 2:
            grid = [[st.mean(Bs[(fd, l)]) if (fd, l) in Bs else float("nan")
                     for l in ulam] for fd in ufd]
            im = ax[1].imshow(grid, aspect="auto", cmap="RdBu_r",
                              vmin=-max(abs(min(min(r) for r in grid)),
                                        abs(max(max(r) for r in grid))),
                              vmax=max(abs(min(min(r) for r in grid)),
                                       abs(max(max(r) for r in grid))))
            ax[1].set_xticks(range(len(ulam)), [f"{l:g}" for l in ulam])
            ax[1].set_yticks(range(len(ufd)), [f"{f:g}" for f in ufd])
            ax[1].set_xlabel("lambda")
            ax[1].set_ylabel("fast_decay")
            ax[1].set_title("D2  B(fast_decay, lambda)")
            fig.colorbar(im, ax=ax[1], label="B")
        else:
            ax[1].axis("off")
        fig.suptitle("Figure D -- causal selective-writeback advantage", y=1.03)
        save(fig, a.out, "figD_B_shuf")
    else:
        print("  [skip] Figure D: no paired direct/shufwrite cells")

    # ------------------------------------------------------------------ E/F
    cl = []
    for pat in (a.cl_base_glob, a.cl_glob):
        for p in sorted(glob.glob(pat)):
            with open(p) as f:
                cl.append(json.load(f))
    if cl:
        pts = {}
        for run in cl:
            for label, mm in run.get("methods", {}).items():
                pts.setdefault(label, []).append(mm)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        fam_color = {"adamw": "#1f77b4", "ewc": "#d62728", "si": "#9467bd",
                     "replay": "#8c564b", "dla": "#2ca02c"}
        for label, recs in sorted(pts.items()):
            base = label.split("@")[0]
            fx = st.mean([r["forward_mean"] for r in recs])
            ry = st.mean([r["retention_mean"] for r in recs])
            ax[0].scatter(fx, ry, color=fam_color.get(base, "k"),
                          marker="s" if base == "dla" else "o", s=42,
                          label=label if len(pts) < 22 else None)
            if len(pts) < 22:
                ax[0].annotate(label, (fx, ry), fontsize=6,
                               xytext=(3, 3), textcoords="offset points")
        ax[0].set_xlabel("forward adaptation  (higher better)")
        ax[0].set_ylabel("retention  (higher better)")
        ax[0].set_title("F1  forward-retention plane")
        ax[0].grid(alpha=0.3)
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k)
                   for k, c in fam_color.items()]
        ax[0].legend(handles=handles, fontsize=8)

        for label, recs in sorted(pts.items()):
            base = label.split("@")[0]
            if base != "dla":
                continue
            fg = st.mean([r.get("forgetting_mean", float("nan")) for r in recs])
            fx = st.mean([r["forward_mean"] for r in recs])
            ax[1].scatter(fg, fx, color=fam_color["dla"], s=42)
            ax[1].annotate(label, (fg, fx), fontsize=6,
                           xytext=(3, 3), textcoords="offset points")
        for label, recs in sorted(pts.items()):
            base = label.split("@")[0]
            if base == "dla":
                continue
            fg = st.mean([r.get("forgetting_mean", float("nan")) for r in recs])
            fx = st.mean([r["forward_mean"] for r in recs])
            ax[1].scatter(fg, fx, color=fam_color.get(base, "k"), marker="x", s=40)
        ax[1].set_xlabel("forgetting  (lower better)")
        ax[1].set_ylabel("forward adaptation")
        ax[1].set_title("F2  DLA (green dots) vs baselines (x)")
        ax[1].grid(alpha=0.3)
        fig.suptitle("Figure F -- continual-learning position "
                     "(DLA swept over (fast_decay, lambda), not a single tuned point)",
                     y=1.03)
        save(fig, a.out, "figF_pareto")
    else:
        print("  [skip] Figure E/F: no CL runs found")

    print("  done")


if __name__ == "__main__":
    main()
