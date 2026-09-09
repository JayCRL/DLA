"""P1 Phase A - extract per-seed W_fast geometry from the 24 saved bodies.

Per seed (0..11) and per history (EH=easy_hard, HE=hard_easy):
  * W_fast norms (global + emb/attn/mlp groups)
  * ΔW_fast := W_fast(HE) - W_fast(EH)  norms, group energy shares, cosine(EH,HE)
  * P gate statistics (softplus(p)) per group for context

Null controls (no mechanism assumed):
  * within-history cross-seed deltas (EH_i - EH_j, HE_i - HE_j): magnitude and
    cosine distributions that define seed-idiosyncrasy baselines
  * cross-history same-seed delta (observed) placed against those baselines

Run:
  python analysis/wfast_geom/geom.py --out results/analysis_wfast
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

from common import (all_seeds, cosine, dot_blockwise, group_keys, load_body_ckpt,
                    norm2_blockwise, canonical_keys, group_of, GROUPS, store_of)

FIELDS = {"w_fast": "wf", "p": "p"}


def _stats(ckpt):
    store = store_of(ckpt)
    keys = canonical_keys(store.keys())
    gk = group_keys(store.keys())
    out = {}
    norm = norm2_blockwise(store, keys)
    gn = {g: math.sqrt(norm2_blockwise(store, gk[g])) for g in GROUPS}
    out["wf"] = {"norm": math.sqrt(norm),
                 "group_norms": {g: gn[g] for g in GROUPS},
                 "group_shares": {g: (gn[g] ** 2 / norm) if norm > 0 else 0.0 for g in GROUPS}}
    gates = {g: [] for g in GROUPS}
    for k in keys:
        gates[group_of(k)].append(float(torch.nn.functional.softplus(store[k]["p"]).mean()))
    out["gate_mean"] = {g: (sum(gates[g]) / len(gates[g])) for g in GROUPS if gates[g]}
    return out


def extract(seed, out_dir):
    eh = load_body_ckpt(seed, "easy_hard")
    he = load_body_ckpt(seed, "hard_easy")
    sk_eh = store_of(eh)
    sk_he = store_of(he)
    keys = canonical_keys(sk_eh.keys())
    gk = group_keys(sk_eh.keys())

    rec = {"seed": seed}
    rec["EH"] = _stats(eh)
    rec["HE"] = _stats(he)

    # delta norms (global + per group)
    d_norm2 = sum(float((sk_he[k]["w_fast"] - sk_eh[k]["w_fast"]).pow(2).sum()) for k in keys)
    d_group_norm2 = {g: sum(float((sk_he[k]["w_fast"] - sk_eh[k]["w_fast"]).pow(2).sum()) for k in gk[g]) for g in GROUPS}
    rec["delta"] = {
        "norm": math.sqrt(d_norm2),
        "group_norms": {g: math.sqrt(d_group_norm2[g]) for g in GROUPS},
        "group_energy_share": {g: (d_group_norm2[g] / d_norm2) if d_norm2 > 0 else 0.0 for g in GROUPS},
        "cos_EH_HE": {g: cosine(sk_eh, sk_he, gk[g]) for g in GROUPS},
        "cos_EH_HE_global": cosine(sk_eh, sk_he, keys),
    }
    # energy of delta relative to each history's own norm (effect size in state space)
    rec["delta"]["rel_norm_vs_HE"] = math.sqrt(d_norm2) / (math.sqrt(norm2_blockwise(sk_he, keys)) + 1e-12)
    rec["delta"]["rel_norm_vs_EH"] = math.sqrt(d_norm2) / (math.sqrt(norm2_blockwise(sk_eh, keys)) + 1e-12)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"geometry_seed{seed}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def cross_seed_nulls():
    """Within-history cross-seed baselines: (EH_i - EH_j) and (HE_i - HE_j)."""
    seeds = all_seeds()
    stores = {o: {} for o in ("easy_hard", "hard_easy")}
    for s in seeds:
        for o in stores:
            stores[o][s] = store_of(load_body_ckpt(s, o))
    out = {"easy_hard": {"pairs": [], "cos_pairs": [], "norm_diffs": []},
           "hard_easy": {"pairs": [], "cos_pairs": [], "norm_diffs": []}}
    for o, tag in (("easy_hard", "EH"), ("hard_easy", "HE")):
        keys = canonical_keys(next(iter(stores[o].values())).keys())
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                si, sj = seeds[i], seeds[j]
                d2 = sum(float((stores[o][si][k]["w_fast"] - stores[o][sj][k]["w_fast"]).pow(2).sum()) for k in keys)
                c = cosine(stores[o][si], stores[o][sj], keys)
                out[o]["norm_diffs"].append(math.sqrt(d2))
                out[o]["cos_pairs"].append(c)
                out[o]["pairs"].append([si, sj])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/analysis_wfast")
    args = ap.parse_args()
    out_dir = os.path.join(args.out, "geometry")
    per_seed = {}
    for s in all_seeds():
        per_seed[s] = extract(s, out_dir)
        print(f"seed {s}: delta_norm={per_seed[s]['delta']['norm']:.4f} "
              f"cos(EH,HE)={per_seed[s]['delta']['cos_EH_HE_global']:.4f} "
              f"rel/HE={per_seed[s]['delta']['rel_norm_vs_HE']:.4f}")
    nulls = cross_seed_nulls()
    agg = {"per_seed": per_seed, "nulls": nulls}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "geometry_summary.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print(f"saved {args.out}/geometry_summary.json + geometry/")


if __name__ == "__main__":
    main()
