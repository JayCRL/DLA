"""Summarize the unified HE-arm batch (retention + structured-allocation + nosleep).

Reads ~/llm-lab/dla_audit/{variant}/probe_seed{S}_HE.json for seeds 0..11 and
variants direct/nocons/nosleep/uniformwrite/topwrite/shufwrite and prints:

  * adaptation table (gain@40 mean±sd; paired t vs direct)
  * retention table: mean end-of-history ppl over the three seen curriculum domains
    (ret_ppl) vs birth ppl (birth_ppl) -> relative forgetting; paired vs direct
Run: python analysis/wfast_geom/unified_summary.py
"""

import json
import os

import numpy as np

OUT = os.path.expanduser("~/llm-lab/dla_audit")
VARIANTS = ["direct", "nocons", "nosleep", "uniformwrite", "topwrite", "shufwrite"]
SEEDS = list(range(12))


def load(v, s):
    return json.load(open(f"{OUT}/{v}/probe_seed{s}_HE.json"))


def tstat(a, b):
    d = np.asarray(a) - np.asarray(b)
    m = float(d.mean())
    s = float(d.std(ddof=1))
    return m, s, (m / (s / np.sqrt(len(d))) if s > 0 else float("nan"))


def main():
    data = {v: [load(v, s) for s in SEEDS] for v in VARIANTS}
    print("== HE arm gain@40 mean±sd ; paired vs direct ==")
    direct = [r["gain40"] for r in data["direct"]]
    for v in VARIANTS:
        g = [r["gain40"] for r in data[v]]
        m, s = float(np.mean(g)), float(np.std(g, ddof=1))
        if v == "direct":
            print(f"  {v:12s}: {m:+.4f}±{s:.4f}")
        else:
            dm, ds, t = tstat(g, direct)
            print(f"  {v:12s}: {m:+.4f}±{s:.4f}  vs direct {dm:+.4f}±{ds:.4f}  t={t:+.2f}")

    print("\n== retention: mean ret_ppl over seen domains, and rel-forgetting vs birth ==")
    doms = ["wiki", "sft", "science"]
    for v in VARIANTS:
        rels, rets = [], []
        for r in data[v]:
            bp = r["birth_ppl"]
            rp = r["ret_ppl"]
            rels.append(float(np.mean([(rp[d] - bp[d]) / bp[d] for d in doms])))
            rets.append(float(np.mean([rp[d] for d in doms])))
        print(f"  {v:12s}: rel_forget {np.mean(rels):+.4f}±{np.std(rels,ddof=1):.4f} | ret_ppl {np.mean(rets):.2f}")
    # retention paired vs direct
    for v in ("nocons", "nosleep"):
        a = []
        for r in data[v]:
            bp, rp = r["birth_ppl"], r["ret_ppl"]
            a.append(float(np.mean([(rp[d] - bp[d]) / bp[d] for d in doms])))
        b = []
        for r in data["direct"]:
            bp, rp = r["birth_ppl"], r["ret_ppl"]
            b.append(float(np.mean([(rp[d] - bp[d]) / bp[d] for d in doms])))
        m, s, t = tstat(a, b)
        print(f"  retention rel_forget {v} vs direct: Δ={m:+.4f}±{s:.4f} t={t:+.2f}")


if __name__ == "__main__":
    main()
