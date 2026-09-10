"""A1 final: write-strength (gamma) and sleep-decay sweeps for allocation selectivity.

Reads ~/llm-lab/dla_audit/{tag}/{variant}/probe_seed{S}_HE.json where
tag in {"g0.05", "g0.5", "" (default gamma=1), "g1.5", "d0.25", "d0.75"};
"" means the directory root (unified-batch default).

Emits every number quoted in report_output/a1_gamma_final.md:
  - per-seed gain@40 table (direct vs shufwrite) by gamma
  - paired gap statistics per gamma (mean, sd, t, dz, 95% CI)
  - cross-gamma paired dose-response contrasts (same seeds)
  - linearity check of gap vs gamma (first-order account predicts gap ~ k*gamma)
  - sleep-decay sweep (direct only, seeds 0-5) vs the default decay

Run: ~/.venv/bin/python analysis/wfast_geom/a1_final.py
"""

import json
import os

import numpy as np

OUT = os.path.expanduser("~/llm-lab/dla_audit")

# tag -> directory name. gamma tags carry a "g" prefix in the launch scripts,
# the default (gamma=1.0) lives at the directory root.
GAMMAS = [("g0.05", "0.05", 3), ("g0.5", "0.5", 12), ("", "1.0", 12), ("g1.5", "1.5", 12)]
DECAYS = [("d0.25", "0.25"), ("", "0.50"), ("d0.75", "0.75")]

# Two-sided 95% t critical values, keyed by degrees of freedom (df = n - 1).
# scipy is not installed in ~/.venv, so the standard table is inlined. Below 1.960
# never happens for df >= 1; the df > 30 fallback of 2.0 is deliberately wider than
# the true value (which approaches 1.96), i.e. conservative rather than overclaiming.
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def load(tag, v, s):
    d = OUT if tag == "" else os.path.join(OUT, tag)
    p = f"{d}/{v}/probe_seed{s}_HE.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["gain40"]


def avail(tag, v, n=12):
    return [s for s in range(n) if load(tag, v, s) is not None]


def ms(x):
    x = np.asarray(x, float)
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else 0.0)


def stats(d):
    """mean, sd, paired t, Cohen dz, 95% CI for a difference vector."""
    d = np.asarray(d, float)
    n = len(d)
    m, sd = ms(d)
    t = m / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    dz = m / sd if sd > 0 else float("nan")
    crit = T95.get(n - 1, 2.0)  # keyed by df; >30 falls back to a conservative 2.0
    half = crit * sd / np.sqrt(n) if n > 1 else float("nan")
    return m, sd, t, dz, n, (m - half, m + half)


def arm(tag, v):
    return np.array([load(tag, v, s) for s in range(12)], float)


def main():
    print("== per-seed gain@40 (HE arm, direct vs shufwrite) ==")
    hdr = f"{'seed':>4}"
    for _, lab, _ in GAMMAS:
        hdr += f" | {'g' + lab + ' dir':>9} {'g' + lab + ' shuf':>11}"
    print(hdr)
    for s in range(12):
        row = f"{s:>4}"
        for tag, _, _ in GAMMAS:
            for v in ("direct", "shufwrite"):
                x = load(tag, v, s)
                row += f" | {(f'{x:+.4f}' if x is not None else '-'):>9}" if v == "direct" \
                    else f" {(f'{x:+.4f}' if x is not None else '-'):>11}"
        print(row)

    print("\n== gamma sweep: direct vs shufwrite (HE arm) ==")
    print("   NOTE: 'gamma' is the scale passed to audit_b3.py --gamma, applied to the")
    print("   LEARNED consolidate_fast_direct coefficient (config default 0.15). gamma=1.0")
    print("   is the project default, not a coefficient of 1.0.")
    gaps = {}
    for tag, lab, n in GAMMAS:
        sd_, ss_ = avail(tag, "direct", n), avail(tag, "shufwrite", n)
        common = sorted(set(sd_) & set(ss_))
        if not common:
            print(f"  gamma={lab}: no data")
            continue
        d = np.array([load(tag, "direct", s) for s in common], float)
        h = np.array([load(tag, "shufwrite", s) for s in common], float)
        gaps[lab] = d - h
        m, sd, t, dz, nn, ci = stats(d - h)
        print(f"  gamma={lab:<5} n={nn:2d}  direct {d.mean():+.4f}  shuf {h.mean():+.4f}  "
              f"gap {m:+.4f}+-{sd:.4f}  t={t:+.2f}  dz={dz:+.2f}  CI95=[{ci[0]:+.4f},{ci[1]:+.4f}]")

    print("\n== dose-response: paired gap contrasts across gamma (same seeds) ==")
    for a, b in [("1.0", "0.5"), ("1.5", "1.0"), ("1.5", "0.5")]:
        if a not in gaps or b not in gaps:
            continue
        m, sd, t, dz, n, ci = stats(gaps[a] - gaps[b])
        print(f"  gap(g={a}) - gap(g={b}) = {m:+.4f}+-{sd:.4f}  t={t:+.2f}  dz={dz:+.2f}  n={n}")

    print("\n== linearity of gap vs gamma (first-order account: gap ~ k*gamma) ==")
    labs = [l for _, l, _ in GAMMAS if l in gaps and l != "0.05"]
    gv = np.array([float(l) for l in labs])
    gm = np.array([gaps[l].mean() for l in labs])
    k = float(np.sum(gv * gm) / np.sum(gv ** 2))
    print(f"  through-origin slope k={k:.4f}")
    print("  predicted: " + "  ".join(f"g={x}: {k * x:+.4f}" for x in gv))
    print("  observed:  " + "  ".join(f"g={x}: {y:+.4f}" for x, y in zip(gv, gm)))
    print("  gap/gamma: " + "  ".join(f"g={x}: {y / x:+.4f}" for x, y in zip(gv, gm)))
    print("  deviation from linear: " + "  ".join(f"{y - k * x:+.4f}" for x, y in zip(gv, gm)))

    print("\n== anchoring against nocons (no-consolidation floor), n=12 ==")
    noc = np.array([load("", "nocons", s) for s in range(12)], float)
    if np.isnan(noc).any():
        print("  nocons incomplete")
    else:
        print(f"  nocons: {noc.mean():+.4f}+-{noc.std(ddof=1):.4f}")
        for tag, lab in (("g0.5", "0.5"), ("", "1.0"), ("g1.5", "1.5")):
            for v in ("direct", "shufwrite"):
                a = np.array([load(tag, v, s) for s in range(12)], float)
                m, sd, t, dz, n, ci = stats(a - noc)
                print(f"  {v:<9}(g={lab}) - nocons = {m:+.4f}+-{sd:.4f}  t={t:+.2f}  dz={dz:+.2f}  "
                      f"CI95=[{ci[0]:+.4f},{ci[1]:+.4f}]")

    print("\n== sleep-decay sweep: direct (seeds 0-5); default decay=0.5 ==")
    base = np.array([load("", "direct", s) for s in range(6)], float)
    print(f"  decay=0.50: {base.mean():+.4f}+-{base.std(ddof=1):.4f}  per-seed "
          f"{[f'{v:+.4f}' for v in base]}")
    for tag, lab in DECAYS:
        if lab == "0.50":
            continue
        x = np.array([load(tag, "direct", s) for s in range(6)], float)
        if np.isnan(x).any():
            print(f"  decay={lab}: incomplete")
            continue
        m, sd, t, dz, n, ci = stats(x - base)
        print(f"  decay={lab}: {x.mean():+.4f}+-{x.std(ddof=1):.4f}  vs 0.50 delta={m:+.4f}+-{sd:.4f} "
              f"t={t:+.2f}  dz={dz:+.2f}  n={n}")
        print(f"      per-seed {[f'{v:+.4f}' for v in x]}")

    print("\n== provenance / consistency ==")
    d10 = arm("", "direct")
    print(f"  default gamma=1.0 direct n=12 recomputed: {d10.mean():+.4f} "
          f"(docs/mechanism_chain.md, from the unified batch: +0.0140)")


if __name__ == "__main__":
    main()
