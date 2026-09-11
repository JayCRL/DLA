"""topk_signed: the sign-preserving, energy-matched top-k write.

Resolves the paper's naming question with a rule fixed BEFORE the data was seen:

    topk_signed ~ direct    -> "sparse aligned selection" (a sparse subset of the
                               aligned coordinates is sufficient)
    topk_signed ~ shufwrite -> "distributed aligned selection" (only the full
                               distributed pattern works; concentration is not enough)
    in between / dose-dependent -> report the dose-response and claim neither label

The contrast that makes this arm worth running is against `topwrite`: same 20%
concentration, but `topwrite` replaced the kept coordinates with a constant magnitude and
so threw the signs away (docs/mechanism_audit.md 3.2). If topk_signed@0.2 tracks `direct`
while topwrite@0.2 sits on the `nocons` floor, then what the write needs is the SIGNED
pattern, not concentration -- which is what "coordinate-aligned" should mean.

Reads <dir>/{tag}/{variant}/probe_seed{S}_HE.json (the project's {tag}/{variant} layout),
default <dir> = ~/llm-lab/dla_audit. Statistics are imported from a1_final.py so the paired
t, dz and the inlined t-table are identical to the write-strength report by construction.

Run: ~/.venv/bin/python analysis/wfast_geom/summary_topk.py [--dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import a1_final
from a1_final import OUT as DEFAULT_OUT
from a1_final import load, stats

FRACS = ["1.0", "0.5", "0.2", "0.05"]
# The comparison arms, all from the same Mac harness / same protocol.
REF = [("direct", ""), ("shufwrite", ""), ("nocons", ""), ("topwrite", "")]


def arm(v, tag="", n=12):
    """gain@40 per seed for one arm; NaN where a seed is missing."""
    return np.array([np.nan if (x := load(tag, v, s)) is None else x for s in range(n)], float)


def topk(frac, n=12):
    return arm("topk_signed", f"topk{frac}", n)


def common_seeds(*arrays):
    """Seeds where every arm is present -- paired tests only ever use these."""
    n = len(arrays[0])
    return [s for s in range(n) if all(not np.isnan(a[s]) for a in arrays)]


def paired(a, b, seeds):
    """Paired difference a-b on `seeds`, via the shared statistics."""
    d = np.array([a[s] - b[s] for s in seeds], float)
    return stats(d)


def fmt(x):
    return f"{x:+.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_OUT, help="root holding the {tag}/{variant} arms")
    args = ap.parse_args()
    # a1_final.load() resolves paths against *its own* module-level OUT, so the override
    # has to be applied there, not to this module's globals.
    a1_final.OUT = os.path.expanduser(args.dir)

    print("# topk_signed — sign-preserving, energy-matched top-k writeback\n")
    print("Regenerate with: `~/.venv/bin/python analysis/wfast_geom/summary_topk.py`\n")

    refs = {name: arm(v, tag) for name, (v, tag) in [(n, (n, "")) for n, _ in REF]}

    # ---- 0. self-check: frac=1.0 must reproduce `direct` -------------------------------
    print("## 0. Implementation self-check (frac=1.0 ≡ direct)\n")
    tk1, di = topk("1.0"), refs["direct"]
    seeds = common_seeds(tk1, di)
    if len(seeds) >= 2:
        m, sd, t, dz, n, ci = paired(tk1, di, seeds)
        print("At frac=1.0 the operator keeps every coordinate, so it must reduce to the "
              "`direct` arm exactly. Any difference is the batch's harness noise floor.\n")
        print(f"```\ntopk_signed(1.0) − direct = {fmt(m)} ± {sd:.4f}   t={t:+.2f}  dz={dz:+.2f}  "
              f"n={n}  CI=[{ci[0]:+.4f},{ci[1]:+.4f}]\n```\n")
        print("This is the yardstick for every contrast below: differences of this size are "
              "harness noise, not effects.\n")
    else:
        print("_not enough completed seeds yet_\n")

    # ---- 1. per-seed table ------------------------------------------------------------
    print("## 1. Per-seed gain@40 (HE arm)\n")
    cols = [("direct", refs["direct"]), ("shufwrite", refs["shufwrite"]),
            ("nocons", refs["nocons"]), ("topwrite20", refs["topwrite"])]
    cols += [(f"topk{fr}", topk(fr)) for fr in FRACS]
    print("| seed | " + " | ".join(c[0] for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for s in range(12):
        row = [("—" if np.isnan(a[s]) else fmt(a[s])) for _, a in cols]
        print(f"| {s} | " + " | ".join(row) + " |")
    means = [("—" if np.all(np.isnan(a)) else fmt(np.nanmean(a))) for _, a in cols]
    print(f"| **mean** | " + " | ".join(f"**{m}**" for m in means) + " |\n")

    # ---- 2. headline contrasts --------------------------------------------------------
    print("## 2. Where each fraction lands (paired, same seeds as `direct`)\n")
    print("| arm | mean | vs direct | vs shufwrite | vs nocons | position | reading |")
    print("|---|---|---|---|---|---|---|")
    pos_rows = []
    for fr in FRACS:
        tk = topk(fr)
        seeds = common_seeds(tk, refs["direct"], refs["shufwrite"], refs["nocons"])
        if len(seeds) < 2:
            print(f"| topk{fr} | — | — | — | — | — | _pending_ |")
            continue
        md, sdd, td, dzd, nd, cid = paired(tk, refs["direct"], seeds)
        ms_, sds, ts, dzs, ns_, cis = paired(tk, refs["shufwrite"], seeds)
        mn, sdn, tn, dzn, nn, cin = paired(tk, refs["nocons"], seeds)
        # position on the direct<->shufwrite axis: 1.0 = as good as direct, 0.0 = as bad as shuffle
        denom = np.nanmean(refs["direct"][seeds]) - np.nanmean(refs["shufwrite"][seeds])
        posi = ((np.nanmean(tk[seeds]) - np.nanmean(refs["shufwrite"][seeds])) / denom
                if denom != 0 else float("nan"))
        pos_rows.append((fr, posi, td, ts))
        if abs(td) < 2.2 and posi > 0.6:
            reading = "**≈ direct**"
        elif abs(ts) < 2.2 and posi < 0.4:
            reading = "**≈ shufwrite**"
        else:
            reading = "intermediate"
        print(f"| topk{fr} | {fmt(np.nanmean(tk[seeds]))} | {fmt(md)} t={td:+.2f} | "
              f"{fmt(ms_)} t={ts:+.2f} | {fmt(mn)} t={tn:+.2f} | {posi:+.2f} | {reading} |")
    print()
    print("`position` = (topk − shufwrite) / (direct − shufwrite): 1.0 means the arm is "
          "indistinguishable from `direct`, 0.0 means it sits on the shuffled floor.\n")

    # ---- 3. the signed-vs-unsigned payoff ---------------------------------------------
    print("## 3. Signed vs unsigned concentration (both at 20%)\n")
    tp, tk2 = refs["topwrite"], topk("0.2")
    seeds = common_seeds(tp, tk2, refs["direct"], refs["nocons"])
    if len(seeds) >= 2:
        m, sd, t, dz, n, ci = paired(tk2, tp, seeds)
        print(f"```\ntopk_signed(0.2) − topwrite(0.2) = {fmt(m)} ± {sd:.4f}  t={t:+.2f}  "
              f"dz={dz:+.2f}  n={n}\n```\n")
        print(f"- `topwrite` (signs dropped): {fmt(np.nanmean(tp[seeds]))}\n"
              f"- `topk_signed` (signs kept): {fmt(np.nanmean(tk2[seeds]))}\n"
              f"- `direct` (all coordinates, same signs): {fmt(np.nanmean(refs['direct'][seeds]))}\n"
              f"- `nocons` floor: {fmt(np.nanmean(refs['nocons'][seeds]))}\n")
    else:
        print("_pending_\n")

    # ---- 4. verdict -------------------------------------------------------------------
    print("## 4. Verdict\n")
    if not pos_rows:
        print("_pending_\n")
        return
    print("| frac | position | t vs direct | t vs shufwrite |")
    print("|---|---|---|---|")
    for fr, posi, td, ts in pos_rows:
        print(f"| {fr} | {posi:+.2f} | {td:+.2f} | {ts:+.2f} |")
    print()
    close_direct = [p for _, p, td, _ in pos_rows if abs(td) < 2.2 and p > 0.6]
    close_shuf = [p for _, p, _, ts in pos_rows if abs(ts) < 2.2 and p < 0.4]
    if len(close_direct) == len(pos_rows):
        print("**Every fraction tested is statistically indistinguishable from `direct` and "
              "clearly above the shuffled floor → \"sparse aligned selection\"**: a sparse, "
              "sign-carrying subset of coordinates reproduces the full write's benefit.")
    elif len(close_shuf) == len(pos_rows):
        print("**Every fraction tested tracks `shufwrite` → \"distributed aligned selection\"**: "
              "concentration alone does not carry the effect; the distributed signed pattern does.")
    else:
        print("**Dose-dependent**: the verdict depends on how much of the coordinate pattern is "
              "kept, so neither label is earned outright. Report the dose-response as the result "
              "and state explicitly which end is supported at which fraction.")


if __name__ == "__main__":
    main()
