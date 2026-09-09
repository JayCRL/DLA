"""P1 null controls for the initial-gradient alignment statistic.

For every seed we have g0 = g_f at step 0 of the D probe (native arms) and the
true history-contrast direction Δ_i = W_fast(HE_i) - W_fast(EH_i).

cos_i := cos(g0(HE)_i, Δ_i)  (and EH variant) is the "true" alignment.

Nulls (both preserve |Δ|, destroy structure):
  (A) shuffle-null : Δ replaced by within-matrix permutation of Δ_i
  (B) cross-seed-null : Δ replaced by Δ_j, j != i (mismatched history pair)
We report the distribution of cos under each null vs the true distribution and
the correlation of cos_i with gain@40(HE/HE) under the true mapping vs the null
mapping (permutation over null realisations).

Run: python analysis/wfast_geom/p1b_grad0_null.py --base results/analysis_wfast
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import canonical_keys, load_body_ckpt, all_seeds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import stage55_learning_rule_development as s55  # noqa: E402
import stage55c_causality as s55c  # noqa: E402


def _bp(seed, order):
    c1 = ROOT / "results" / "stage55c" / f"seed{seed}" / "bodies" / f"seed{seed}_{order}_meta.pt"
    c2 = ROOT / "results" / "stage55e" / "bodies" / f"seed{seed}_{order}_meta.pt"
    return c1 if c1.exists() else c2


def grad0(seed, order, args, stoi):
    device = "cpu"
    _d, _p, d_train, _dv = s55.build_curriculum(seed, args, stoi)
    model, state = s55c.load_body(str(_bp(seed, order)), args, device)
    model.train()
    rng = random.Random(800000 + seed)
    x, y = s55.get_batch(d_train, args.block, args.batch, rng, device)
    model.set_dla_state(state)
    model.zero_grad(set_to_none=True)
    _lg, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), model.dla_cfg.grad_clip)
    keys = canonical_keys(state.store.keys())
    with torch.no_grad():
        parts = []
        for k in keys:
            mod = model.key_modules[k]
            g = mod.weight.grad
            if g is None:
                continue
            parts.append((F.softplus(state.store[k]["p"]) * g).reshape(-1).float())
    return torch.cat(parts)


def delta_vec(seed):
    eh = load_body_ckpt(seed, "easy_hard")["store"]
    he = load_body_ckpt(seed, "hard_easy")["store"]
    keys = canonical_keys(eh.keys())
    return torch.cat([(he[k]["w_fast"] - eh[k]["w_fast"]).reshape(-1).float() for k in keys]), keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/analysis_wfast")
    ap.add_argument("--nshuf", type=int, default=300)
    ap.add_argument("--nperm", type=int, default=2000)
    args = ap.parse_args()

    import pickle
    from common import Args
    a = Args()
    stoi = pickle.load(open(s55.META, "rb"))["stoi"]
    seeds = all_seeds()

    # per-seed per-key shapes to shuffle blockwise later
    keys_of = {}
    deltas = {}
    g0 = {"HE": {}, "EH": {}}
    shapes = {}
    for s in seeds:
        dv, keys = delta_vec(s)
        deltas[s] = dv
        keys_of[s] = keys
        shapes[s] = {k: tuple(load_body_ckpt(s, "easy_hard")["store"][k]["w_fast"].shape) for k in keys}
        g0["HE"][s] = grad0(s, "hard_easy", a, stoi)
        g0["EH"][s] = grad0(s, "easy_hard", a, stoi)
        print(f"seed {s}: g0 done", flush=True)

    def cos(u, v):
        return float((u * v).sum() / (u.norm() * v.norm() + 1e-12))

    # ---- build true + null alignments properly (blockwise shuffle) ----
    true_cos = {"HE": [], "EH": []}
    shuf_cos = {"HE": [], "EH": []}
    cross_cos = {"HE": [], "EH": []}
    true_idx = {"HE": [], "EH": []}

    # precompute per-seed delta block views (start:end in flattened delta)
    blocks = {}
    for s in seeds:
        off = 0
        blocks[s] = {}
        for k in keys_of[s]:
            n = int(np.prod(shapes[s][k]))
            blocks[s][k] = (off, off + n)
            off += n

    def split(vec):
        return {k: vec[a:b] for k, (a, b) in blocks[s].items()}

    rng = random.Random(12345)
    for arm, key in (("HE", "hard_easy"), ("EH", "easy_hard")):
        for s in seeds:
            gf = g0[arm][s]
            d_true = deltas[s]
            true_cos[arm].append(cos(gf, d_true))
            true_idx[arm].append(s)
            # cross-seed null: use another seed's delta
            others = [j for j in seeds if j != s]
            cs = [cos(gf, deltas[j]) for j in others]
            cross_cos[arm].append(float(np.mean(cs)))
            # shuffled null: permute within each key block
            gf_blocks = split(gf)  # same block layout for gradient (keys order same)
            d_parts = []
            for k in keys_of[s]:
                n = blocks[s][k][1] - blocks[s][k][0]
                perm = torch.randperm(n, generator=torch.Generator().manual_seed(rng.randint(0, 10 ** 9)))
                dk = deltas[s][blocks[s][k][0]:blocks[s][k][1]][perm]
                d_parts.append(dk)
            d_sh = torch.cat(d_parts)
            shuf_cos[arm].append(cos(gf, d_sh))

    # outcomes
    def gain_he(s):
        j = json.load(open(ROOT / "results" / "stage55e" / "seeds" / f"seed{s}.json"))["results"]["HE/HE"]
        return j["curve"][-1]["gain"]

    y = [gain_he(s) for s in seeds]

    def pear(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        return 0.0 if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])

    r_true = pear(true_cos["HE"], y)
    # null correlation: shuffle the delta pairing across seeds
    def pval(r0, xs, n):
        rngp = random.Random(7)
        c = 0
        for _ in range(n):
            idx = list(range(len(xs)))
            rngp.shuffle(idx)
            if abs(pear([xs[i] for i in idx], y)) >= abs(r0):
                c += 1
        return (c + 1) / (n + 1)

    summary = {
        "true_cos_HE": true_cos["HE"], "true_cos_EH": true_cos["EH"],
        "shuf_cos_HE": shuf_cos["HE"], "cross_cos_HE": cross_cos["HE"],
        "mean_true_HE": float(np.mean(true_cos["HE"])),
        "mean_shuf_HE": float(np.mean(shuf_cos["HE"])),
        "mean_cross_HE": float(np.mean(cross_cos["HE"])),
        "sd_shuf_HE": float(np.std(shuf_cos["HE"])),
        "r_cos_true_vs_gain": r_true,
        "perm_p_true": pval(r_true, true_cos["HE"], args.nperm),
        "r_cos_shuf_vs_gain": pear(shuf_cos["HE"], y),
        "r_cos_cross_vs_gain": pear(cross_cos["HE"], y),
    }
    with open(os.path.join(args.base, "p1b_grad0_null.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("true mean |cos(g0(HE),Δ)| = %.4f ; shuffle-null %.4f ; cross-seed-null %.4f"
          % (np.mean(np.abs(true_cos["HE"])), np.mean(np.abs(shuf_cos["HE"])), np.mean(np.abs(cross_cos["HE"]))))
    print("r(true_cos, gain40) = %.3f (perm p=%.3f); r(shuf) = %.3f; r(cross) = %.3f"
          % (r_true, summary["perm_p_true"], pear(shuf_cos["HE"], y), pear(cross_cos["HE"], y)))
    print(f"saved {args.base}/p1b_grad0_null.json")


if __name__ == "__main__":
    main()
