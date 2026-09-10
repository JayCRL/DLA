"""A1 final: write-strength (gamma) and sleep-decay sweeps for allocation selectivity.

Reads ~/llm-lab/dla_audit/{tag}/{variant}/probe_seed{S}_HE.json where
tag ∈ {"" (default gamma=1, decay=0.5), g0.05, g0.5, g1.5, d0.25, d0.75}.
Prints direct-vs-shufwrite paired gaps by gamma and direct gains by sleep decay.
"""

import json
import os

import numpy as np

OUT = os.path.expanduser("~/llm-lab/dla_audit")


def load(tag, v, s):
    d = OUT if tag in ("", None) else os.path.join(OUT, tag)
    return json.load(open(f"{d}/{v}/probe_seed{s}_HE.json"))["gain40"]


def ms(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def tpair(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    m, s = ms(d)
    return m, s, (m / (s / np.sqrt(len(d))) if s > 0 else float("nan"))


def main():
    gammas = [("0.05", list(range(0, 3))), ("0.5", list(range(12))), ("", list(range(12))), ("1.5", list(range(12)))]
    print("== gamma sweep: direct vs shufwrite (HE arm) ==")
    for tag, seeds in gammas:
        label = "1.0" if tag == "" else tag
        try:
            dd = [load(tag, "direct", s) for s in seeds]
            ss = [load(tag, "shufwrite", s) for s in seeds]
        except FileNotFoundError as e:
            print(f"  gamma={label}: missing file ({e})")
            continue
        m, s, t = tpair(dd, ss)
        print(f"  gamma={label:<5} n={len(seeds):2d} direct {np.mean(dd):+.4f} shuf {np.mean(ss):+.4f} "
              f"gap {m:+.4f}±{s:.4f} t={t:+.2f}")
    print("\n== sleep-decay sweep: direct (seeds 0-5); default decay=0.5 ==")
    base = [load("", "direct", s) for s in range(6)]
    print(f"  decay=0.50: {np.mean(base):+.4f}±{np.std(base, ddof=1):.4f}")
    for tag, lab in (("d0.25", "0.25"), ("d0.75", "0.75")):
        try:
            x = [load(tag, "direct", s) for s in range(6)]
        except FileNotFoundError as e:
            print(f"  decay={lab}: missing ({e})")
            continue
        m, s, t = tpair(x, base)
        print(f"  decay={lab}: {np.mean(x):+.4f}±{np.std(x, ddof=1):.4f}  vs 0.50 Δ={m:+.4f}±{s:.4f} t={t:+.2f}")


if __name__ == "__main__":
    main()
