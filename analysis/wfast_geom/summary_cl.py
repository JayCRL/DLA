"""P2 analysis: is DLA on a better point of the stability/plasticity front?

The point of the CL benchmark is not "does DLA beat AdamW" at one arbitrary setting.
A fine-tuning baseline's operating point is chosen by its learning rate, and EWC/SI
choose theirs with lambda -- so each family traces a FRONT in (forward, retention)
space.  The question that survives review is whether DLA's single operating point
lies above the fronts the other families can reach.

    forward    mean relative perplexity drop on each task's own held-out slice
    retention  mean(after / end) over tasks

Both are "higher is better".  A point is DOMINATED if some other point is at least as
good on both axes and strictly better on one.

Two things are checked before any ranking is believed:
  * lambda validity -- every ewc/si entry must report max penalty/loss above the
    acceptance floor.  An inert penalty makes the "baseline" a silent copy of AdamW.
  * that the grid actually moved the operating point, i.e. that the swept parameter
    produced distinguishable fronts rather than a single repeated number.
"""
import argparse
import glob
import json
import math
import os
import statistics as st

BASE_METHODS = {"dla", "adamw", "replay"}


def family(label):
    for b in BASE_METHODS:
        if label == b or label.startswith(b + "@"):
            return b
    for b in ("ewc", "si"):
        if label.startswith(b):
            return b
    return label


def load(model, base):
    out = []
    for p in sorted(glob.glob(os.path.join(base, model, "*.json"))):
        out.append(json.load(open(p)))
    return out


def agg(D, label, key):
    v = [d["methods"][label][key] for d in D if label in d["methods"]]
    if not v:
        return None
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))


def dominated(points, pt):
    """Return the label of a point dominating pt, or None."""
    for lab, (f, r) in points.items():
        if lab == pt[0]:
            continue
        if f >= pt[1][0] - 1e-12 and r >= pt[1][1] - 1e-12 and \
                (f > pt[1][0] + 1e-12 or r > pt[1][1] + 1e-12):
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--base", default="results/cloud/dla_cl")
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()

    D = load(a.model, a.base)
    if not D:
        raise SystemExit(f"no results under {a.base}/{a.model}")
    n = len(D)
    L = []
    def say(s=""):
        print(s)
        L.append(s)

    labels = list(D[0]["methods"].keys())
    fams = {}
    for lab in labels:
        fams.setdefault(family(lab), []).append(lab)

    say(f"P2  continual-learning comparison -- {a.model}, n={n} seeds, 10 tasks")
    say("=" * 94)
    say()
    say(f"  {'method':16s} {'forward':>18s} {'retention':>18s} {'forgetting':>18s} "
        f"{'pen/loss':>10s}")
    pts = {}
    for lab in sorted(labels, key=lambda x: (family(x), x)):
        f = agg(D, lab, "forward_mean")
        r = agg(D, lab, "retention_mean")
        g = agg(D, lab, "forgetting_mean")
        if not f:
            continue
        pr = [d["methods"][lab].get("penalty_ratio_max") for d in D
              if lab in d["methods"]
              and d["methods"][lab].get("penalty_ratio_max") is not None]
        prs = f"{max(pr):.2e}" if pr else "--"
        say(f"  {lab:16s} {f[0]:+9.4f} ± {f[1]:.4f} {r[0]:9.4f} ± {r[1]:.4f} "
            f"{g[0]:+9.4f} ± {g[1]:.4f} {prs:>10s}")
        pts[lab] = (f[0], r[0])

    say()
    say("=" * 94)
    say("Validity checks before any ranking")
    say("=" * 94)
    bad = []
    for lab in labels:
        if family(lab) not in ("ewc", "si"):
            continue
        pr = [d["methods"][lab].get("penalty_ratio_max") for d in D
              if lab in d["methods"]
              and d["methods"][lab].get("penalty_ratio_max") is not None]
        if not pr:
            continue
        if max(pr) < a.floor:
            bad.append((lab, max(pr)))
    if bad:
        for lab, m in bad:
            say(f"  INERT  {lab}: max penalty/loss = {m:.2e} < {a.floor:g} -- this "
                f"'baseline' is a copy of AdamW, drop it")
    else:
        say("  all ewc/si penalties are active (max penalty/loss >= "
            f"{a.floor:g})")
    for fam, labs in sorted(fams.items()):
        if len(labs) < 2:
            continue
        fs = [pts[l][0] for l in labs if l in pts]
        rs = [pts[l][1] for l in labs if l in pts]
        span = max(fs) - min(fs), max(rs) - min(rs)
        say(f"  {fam:8s} swept over {len(labs)} settings -> forward span {span[0]:.4f}, "
            f"retention span {span[1]:.4f}"
            f"{'   (grid did not move the point -- widen it)' if max(span) < 1e-3 else ''}")

    say()
    say("=" * 94)
    say("Fronts  (forward, retention; both higher is better)")
    say("=" * 94)
    for fam, labs in sorted(fams.items()):
        if fam in ("dla",):
            continue
        pts_f = {l: pts[l] for l in labs if l in pts}
        front = [l for l in pts_f if dominated(pts_f, (l, pts_f[l])) is None]
        say(f"  {fam:8s} front: " + ", ".join(
            f"{l} ({pts_f[l][0]:+.4f}, {pts_f[l][1]:.4f})"
            for l in sorted(front, key=lambda x: pts_f[x][0])))

    say()
    say("=" * 94)
    say("Verdict on DLA")
    say("=" * 94)
    dla = [l for l in pts if family(l) == "dla"]
    if not dla:
        say("  no DLA entry in these results")
    else:
        dl = dla[0]
        others = {l: p for l, p in pts.items() if family(l) != "dla"}
        d = dominated(others, (dl, pts[dl]))
        say(f"  DLA point: forward {pts[dl][0]:+.4f}, retention {pts[dl][1]:.4f}")
        if d is None:
            say("  -> NOT dominated by any baseline setting on these axes.")
        else:
            say(f"  -> DOMINATED by {d} "
                f"({pts[d][0]:+.4f}, {pts[d][1]:.4f}): that baseline is at least as "
                f"good on both axes.")
            fl = family(d)
            say(f"     The relevant statement is therefore about the {fl} front, not "
                f"about DLA beating a single baseline setting: on this benchmark DLA "
                f"does not occupy a point standard methods cannot reach.")
        if n < 2:
            say("  NOTE: n=1, these are single-seed numbers.")

    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        open(a.md, "w").write("\n".join(L) + "\n")
        print(f"\nwrote {a.md}")


if __name__ == "__main__":
    main()
