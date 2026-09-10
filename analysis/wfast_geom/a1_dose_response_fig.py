"""A1 dose-response figure: allocation selectivity vs write strength.

Builds `paper/figures/fig_a1_dose_response.png` (2 panels) from the same data the
`a1_final.py` analysis reads:

  panel A  paired gap (direct - shufwrite) as a function of the write-strength SCALE,
           with 95% CI and the zero line -> "graded, not a knife-edge"
  panel B  the two arm means against the no-consolidation floor -> "misallocation is
           not rescued by more energy: shufwrite stays on the nocons floor at every
           scale, while the matched write rises off it"

Notation: the x axis is the `--gamma` SCALE passed to `audit_b3.py`, a multiplier on the
learned `consolidate_fast_direct` coefficient (config default 0.15) -- not the
coefficient itself. gamma=1.0 is the project default.

Usage: ~/.venv/bin/python analysis/wfast_geom/a1_dose_response_fig.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser("~/llm-lab/dla_audit")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "paper", "figures", "fig_a1_dose_response.png")

# x values are SCALES; 0.05 is n=3 (underpowered, drawn separately)
LEVELS = [(0.5, "g0.5", 12), (1.0, "", 12), (1.5, "g1.5", 12)]


def load(tag, v, s):
    d = OUT if tag == "" else os.path.join(OUT, tag)
    p = f"{d}/{v}/probe_seed{s}_HE.json"
    return json.load(open(p))["gain40"] if os.path.exists(p) else None


def arm(tag, v, n):
    return np.array([load(tag, v, s) for s in range(n)], float)


def ci95(d):
    d = np.asarray(d, float)
    n = len(d)
    crit = {2: 4.303, 5: 2.571, 11: 2.201}.get(n - 1, 2.0)
    half = crit * d.std(ddof=1) / np.sqrt(n)
    return d.mean(), half


def main():
    xs, gaps, ghalf, dirs, shufs = [], [], [], [], []
    for x, tag, n in LEVELS:
        d, h = arm(tag, "direct", n), arm(tag, "shufwrite", n)
        m, half = ci95(d - h)
        xs.append(x); gaps.append(m); ghalf.append(half)
        dirs.append(d.mean()); shufs.append(h.mean())
    xs = np.array(xs); gaps = np.array(gaps); ghalf = np.array(ghalf)
    dirs = np.array(dirs); shufs = np.array(shufs)

    noc = arm("", "nocons", 12)
    noc_m, noc_half = ci95(noc)  # CI of the mean, not of a difference

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # ---- panel A: paired gap vs write-strength scale -------------------------
    a = ax[0]
    a.axhline(0.0, color="0.35", lw=1.0, ls="--", zorder=1)
    a.errorbar(xs, gaps, yerr=ghalf, fmt="o-", color="#1f5fa8", lw=2.0, ms=8,
               capsize=5, elinewidth=1.6, zorder=3, label="paired gap (direct − shufwrite)")
    # fitted through-origin slope k=0.0123 (first-order account: gap ~ k*scale)
    xf = np.linspace(0.35, 1.65, 50)
    a.plot(xf, 0.0123 * xf, color="#c0392b", lw=1.4, ls=":", zorder=2,
           label="first-order fit  k·scale  (k=0.0123)")
    # n=3 tail point, drawn hollow to mark it as not part of the n=12 set
    d3, h3 = arm("g0.05", "direct", 3), arm("g0.05", "shufwrite", 3)
    m3, h3c = ci95(d3 - h3)
    a.errorbar([0.05], [m3], yerr=[h3c], fmt="s", mfc="white", mec="#1f5fa8",
               color="#1f5fa8", ms=7, capsize=5, elinewidth=1.4, zorder=3)
    a.annotate("n=3 (underpowered;\nCI includes 0)", xy=(0.05, m3), xytext=(0.20, 0.0118),
               fontsize=7.5, color="0.3",
               arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))
    a.annotate(f"gap {gaps[1]:+.4f}\nt={gaps[1]/(ghalf[1]/2.201):+.2f}", xy=(1.0, gaps[1]),
               xytext=(1.03, gaps[1] - 0.0082), fontsize=8, color="#1f5fa8")
    a.annotate(f"gap {gaps[2]:+.4f}\nt={gaps[2]/(ghalf[2]/2.201):+.2f}", xy=(1.5, gaps[2]),
               xytext=(1.21, gaps[2] + 0.0012), fontsize=8, color="#1f5fa8")
    a.annotate(f"gap {gaps[0]:+.4f}\nt={gaps[0]/(ghalf[0]/2.201):+.2f} (n.s.)",
               xy=(0.5, gaps[0]), xytext=(0.54, -0.0058), fontsize=8, color="0.35")
    a.set_xlabel("write-strength scale γ_scale  (× learned `consolidate_fast_direct`)")
    a.set_ylabel("paired gain@40 gap on D\n(direct − energy-matched shuffle)")
    a.set_title("A  Benefit is graded in write strength, not a knife-edge", fontsize=10.5)
    a.set_xlim(0.0, 1.78)
    a.set_ylim(-0.0075, 0.0335)
    a.legend(fontsize=8, loc="upper left", framealpha=0.92)
    a.grid(alpha=0.25)

    # ---- panel B: arms against the no-consolidation floor -------------------
    b = ax[1]
    b.axhspan(noc_m - noc_half, noc_m + noc_half, color="0.65", alpha=0.35, zorder=1)
    b.axhline(noc_m, color="0.35", lw=1.3, ls="--", zorder=2)
    b.annotate("no consolidation (`nocons`) floor\n+0.0027 ± 0.0103", xy=(0.52, noc_m),
               xytext=(0.42, -0.0125), fontsize=8, color="0.25",
               arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
    b.errorbar(xs, dirs, yerr=[arm(t, "direct", n).std(ddof=1) / np.sqrt(n) * 2.201
                               for _, t, n in LEVELS],
               fmt="o-", color="#1f5fa8", lw=2.0, ms=8, capsize=5, zorder=4,
               label="`direct`  (γ·W_fast, coordinate-matched)")
    b.errorbar(xs, shufs, yerr=[arm(t, "shufwrite", n).std(ddof=1) / np.sqrt(n) * 2.201
                                for _, t, n in LEVELS],
               fmt="s-", color="#c0392b", lw=2.0, ms=7, capsize=5, zorder=4,
               label="`shufwrite`  (same energy, placement destroyed)")
    b.annotate("+50% write energy,\nstill on the floor", xy=(1.5, shufs[2]),
               xytext=(1.04, 0.0132), fontsize=8, color="#c0392b",
               arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))
    b.set_xlabel("write-strength scale γ_scale")
    b.set_ylabel("mean gain@40 on D (HE arm, n=12)")
    b.set_title("B  Misallocation is not rescued by more energy", fontsize=10.5)
    b.set_xlim(0.35, 1.68)
    b.set_ylim(-0.0175, 0.0335)
    b.legend(fontsize=8, loc="upper left", framealpha=0.92)
    b.grid(alpha=0.25)

    fig.suptitle("Write-strength robustness of allocation selectivity "
                 "(HE arm, unseen domain D, n=12; Mac parity-validated harness)",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=170, bbox_inches="tight")
    print(f"figure written: {FIG}")
    print(f"  gaps   : {[f'{g:+.4f}' for g in gaps]}  (γ_scale {list(xs)})")
    print(f"  direct : {[f'{d:+.4f}' for d in dirs]}")
    print(f"  shuf   : {[f'{s:+.4f}' for s in shufs]}")
    print(f"  nocons : {noc_m:+.4f} ± {noc.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
