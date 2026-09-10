"""Visualise the mechanism chain:
   gradient leaky integration -> W_fast structure -> selective write-back -> future adaptation

Panels:
  Fig A (schematic): the four links with their equations and evidence tags.
  Fig B (empirical): (1) per-step gradient/alignment trajectory from the stored
      D-probe replays; (2) |W_fast| magnitude distribution (HE vs history contrast);
      (3) allocation-shuffle illustration (same energy, permuted coordinates);
      (4) outcome bars for the write-rule interventions (n=12) + retention.

Data: analysis/wfast_geom/report_output/mechanism_chain/data/* (replays, wfast_stats,
geometry_summary) and the committed unified-batch results.
Run: python analysis/wfast_geom/mechanism_chain_fig.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "report_output", "mechanism_chain", "data")
OUT = os.path.join(HERE, "report_output", "mechanism_chain")
os.makedirs(OUT, exist_ok=True)

# committed unified-batch numbers (n=12, HE arm)
GAIN = {"direct": 0.0140, "nocons": 0.0027, "nosleep": 0.0303, "uniformwrite": 0.0035,
        "topwrite": 0.0051, "shufwrite": 0.0025}
GAIN_SD = {"direct": 0.0148, "nocons": 0.0103, "nosleep": 0.0215, "uniformwrite": 0.0094,
           "topwrite": 0.0092, "shufwrite": 0.0086}
RET = {"direct": -0.0192, "nocons": -0.0152, "nosleep": -0.0049}


def schematic():
    fig, ax = plt.subplots(figsize=(12.4, 4.0))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 4.0); ax.axis("off")
    boxes = [
        ("1  gradient leaky integration",
         "$W_{fast,T}=-\\eta\\sum_k (1-\\lambda)^{T-1-k}\\,\\phi(P_k)\\odot \\hat a_k$\n"
         "$\\hat a = \\hat m/(\\sqrt{\\hat v}+\\epsilon)$   $1/\\lambda=50\\approx$ 1 stage",
         "identity (code)"),
        ("2  $W_{fast}$ spatial structure",
         "$\\Delta_{hist}=W_{fast}^{HE}-W_{fast}^{EH}$\n"
         "$\\|\\Delta\\|\\approx$ seed noise;  $\\cos(g_0,\\Delta)$ orders seeds: $r=0.87$ (FDR 0.003)",
         "structure: measured\nordering: correlational"),
        ("3  selective write-back",
         "$W_{slow} {+}{=} \\beta Q+\\gamma W_{fast}$;  $A_i=\\gamma W_{fast,i}$\n"
         "energy-matched shuffle: $A'=\\Pi_\\pi A$, $\\|A'\\|=\\|A\\|$  $\\Rightarrow$  $t=-4.86$",
         "causal (core)"),
        ("4  future adaptation",
         "$\\Delta L_D\\approx\\langle \\nabla_{W_{slow}}L_D,\\,A\\rangle$\n"
         "boundary $\\to$ retention;  write allocation $\\to$ forward gain",
         "causal (n=12; 10-task n=8)"),
    ]
    w, h, y = 2.85, 2.1, 1.15
    xs = [0.25, 3.35, 6.45, 9.55]
    colors = ["#eef3fa", "#eaf6ee", "#fdf1e6", "#f6eef7"]
    for (title, eq, tag), x, c in zip(boxes, xs, colors):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=c, ec="#444", lw=1.1))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=10.5, weight="bold")
        ax.text(x + w / 2, y + h - 1.05, eq, ha="center", va="top", fontsize=8.2)
        ax.text(x + w / 2, y + 0.22, tag, ha="center", va="bottom", fontsize=8.0,
                style="italic", color="#22558c")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((x0 + w + 0.02, y + h / 2), (x1 - 0.03, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=18, lw=1.6, color="#333"))
    ax.text(6.2, 0.42,
            "no explicit alignment objective exists anywhere in the code  ·  "
            "allocation selectivity is emergent from the trace, not designed",
            ha="center", fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figA_chain_schematic.png"), dpi=150)
    plt.close(fig)


def empirical():
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))

    # (1) per-step gradient / alignment trajectory (HE/HE arm, 12 seeds)
    files = sorted(glob.glob(os.path.join(DATA, "replays", "seeds", "seed*_HE_HE.json")))
    gfn, cos = [], []
    for fp in files:
        steps = json.load(open(fp))["steps"]
        gfn.append([s["global"]["gfn"] for s in steps])
        cos.append([(s["cos_d"].get("global") or 0.0) for s in steps])
    gfn = np.array(gfn); cos = np.array(cos)
    xs = np.arange(1, gfn.shape[1] + 1)
    ax = axes[0, 0]
    ax.plot(xs, gfn.mean(0), color="#2b7bba", label=r"$\Vert g_f\Vert$ (mean)")
    ax.fill_between(xs, gfn.mean(0) - gfn.std(0) / np.sqrt(len(gfn)),
                    gfn.mean(0) + gfn.std(0) / np.sqrt(len(gfn)), alpha=.2, color="#2b7bba")
    ax2 = ax.twinx()
    ax2.plot(xs, cos.mean(0) * 100, color="#d9534f", label=r"$\cos(g_f,\Delta)\times100$ (mean)")
    ax2.axhline(0, color="#d9534f", lw=.4, ls=":")
    ax2.set_ylabel(r"$\cos(g_f,\Delta)\times100$", color="#d9534f", fontsize=8)
    ax.set_xlabel("D-probe step"); ax.set_ylabel(r"$\Vert g_f\Vert$ (gated gradient)")
    ax.set_title("Link 1–2: gated-gradient trajectory and its (small) alignment\nwith the history contrast", fontsize=9.5)
    ax.tick_params(labelsize=8); ax2.tick_params(labelsize=8)

    # (2) |W_fast| magnitude distribution
    st = json.load(open(os.path.join(DATA, "wfast_stats.json"))) if os.path.exists(
        os.path.join(DATA, "wfast_stats.json")) else json.load(
        open(os.path.join(DATA, "wfast_stats_small.json")))
    key = "seed0_wf_he" if "seed0_wf_he" in st else next(k for k in st if k.endswith("wf_he"))
    dkey = "seed0_delta" if "seed0_delta" in st else next(k for k in st if k.endswith("delta"))
    a_he = np.array(st[key]["subsample_abs"][:20000])
    a_d = np.array(st[dkey]["subsample_abs"][:20000])
    ax = axes[0, 1]
    bins = np.logspace(-4, 0, 45)
    ax.hist(np.clip(a_he, 1e-4, None), bins=bins, alpha=.65, label=r"$|W_{fast}|$ (HE body)", color="#2b7bba")
    ax.hist(np.clip(a_d, 1e-4, None), bins=bins, alpha=.55, label=r"$|\Delta_{hist}|$", color="#f0ad4e")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("coordinate magnitude (log)"); ax.set_ylabel("count (log)")
    ax.set_title("Link 2: structure is fine-grained\n(heavy-tailed per-coordinate magnitudes)", fontsize=9.5)
    ax.legend(fontsize=7.5); ax.tick_params(labelsize=8)

    # (3) allocation-shuffle illustration
    rng = np.random.default_rng(0)
    A = np.abs(rng.standard_normal(400)) * rng.choice([0, 1], 400, p=[.55, .45])
    perm = rng.permutation(A.size)
    ax = axes[1, 0]
    ax.plot(A[:120], color="#2b7bba", lw=1.2, label=r"direct: $A_i=\gamma W_{fast,i}$")
    ax.plot(A[perm][:120], color="#d9534f", lw=1.0, ls="--", label=r"shuffle: $A'=\Pi_\pi A$")
    ax.set_xlabel("coordinate index (first 120)"); ax.set_ylabel("write magnitude")
    ax.set_title(f"Link 3: same energy, different placement\n"
                 f"$\\|A'\\|=\\|A\\|$ exactly; paired $t=-4.86$ on gain@40", fontsize=9.5)
    ax.legend(fontsize=7.5); ax.tick_params(labelsize=8)

    # (4) outcomes of the intervention family
    order = ["nocons", "shufwrite", "uniformwrite", "topwrite", "direct", "nosleep"]
    ax = axes[1, 1]
    xs = np.arange(len(order))
    cols = ["#d9534f", "#e08a5a", "#e0b25a", "#b8a24a", "#2b7bba", "#6b4a9a"]
    ax.bar(xs, [GAIN[k] for k in order], yerr=[GAIN_SD[k] for k in order],
           capsize=3, color=cols, alpha=.9, ec="k", lw=.5)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=20, fontsize=8)
    ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("gain@40 on unseen D (HE arm, n=12)")
    ax.set_title("Link 4: unselective / shuffled writes fall to no-consolidation;\n"
                 "boundary-off is highest forward but worst on retention", fontsize=9.5)
    ax.tick_params(labelsize=8)
    for x, k in zip(xs, order):
        ax.text(x, GAIN[k] + (0.0015 if GAIN[k] >= 0 else -0.004), f"{GAIN[k]:+.3f}",
                ha="center", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figB_chain_empirical.png"), dpi=150)
    plt.close(fig)
    print("wrote figA_chain_schematic.png and figB_chain_empirical.png to", OUT)


if __name__ == "__main__":
    schematic()
    empirical()
