"""P2 - PCA/SVD over per-seed ΔW_fast and whether future-task gradient
projections onto the top shared directions predict adaptation outcomes.

ΔW_fast_i = W_fast(HE_i) - W_fast(EH_i)  (12 x D, D ~6.3M)
PCA is computed via the 12x12 Gram matrix (no D x D eigendecomposition).
We then project each seed's initial D-probe gradient g_f = softplus(P)*grad
(for HE and EH native arms) onto the top PCs and test whether the projection
predicts gain@40 / LE_D of that seed.

Nulls: permutation over seeds; plus a randomized-direction null (projection
onto random orthogonal directions with the same per-PC energy).

Run: python analysis/wfast_geom/p2.py --base results/analysis_wfast
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

from common import (Args, canonical_keys, load_body_ckpt, group_of,
                    all_seeds, norm2_blockwise)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import stage55_learning_rule_development as s55  # noqa: E402
import stage55c_causality as s55c  # noqa: E402

META = s55.META


def load_delta_and_g0(seed):
    """Return (delta_vec, {order: gf0_vec}) as torch cat tensors."""
    args = Args()
    import pickle
    stoi = pickle.load(open(META, "rb"))["stoi"]
    _d, _p, d_train, d_val = s55.build_curriculum(seed, args, stoi)
    device = "cpu"

    def ckpt(order):
        return load_body_ckpt(seed, order)

    eh, he = ckpt("easy_hard"), ckpt("hard_easy")
    keys = canonical_keys(eh["store"].keys())
    delta = torch.cat([(he["store"][k]["w_fast"] - eh["store"][k]["w_fast"]).reshape(-1).float() for k in keys])

    gf = {}
    for order, name in (("easy_hard", "EH"), ("hard_easy", "HE")):
        model, state = s55c.load_body(str(_bp(seed, order)), args, device)
        model.train()
        rng = random.Random(800000 + seed)
        x, y = s55.get_batch(d_train, args.block, args.batch, rng, device)
        model.set_dla_state(state)
        model.zero_grad(set_to_none=True)
        _logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.dla_cfg.grad_clip)
        with torch.no_grad():
            parts = []
            for k in keys:
                mod = model.key_modules[k]
                g = mod.weight.grad
                if g is None:
                    continue
                gate = F.softplus(state.store[k]["p"])
                parts.append((gate * g).reshape(-1).float())
        gf[name] = torch.cat(parts)
    return delta, gf


def _bp(seed, order):
    c1 = ROOT / "results" / "stage55c" / f"seed{seed}" / "bodies" / f"seed{seed}_{order}_meta.pt"
    c2 = ROOT / "results" / "stage55e" / "bodies" / f"seed{seed}_{order}_meta.pt"
    return c1 if c1.exists() else c2


def gram_pca(X):
    """X: list of D-vectors (n x D). Returns variance ratios + eigvecs of Gram."""
    n = len(X)
    G = torch.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            d = float((X[i] * X[j]).sum())
            G[i, j] = G[j, i] = d
    vals, vecs = torch.linalg.eigh(G)
    # ascending -> descending
    vals = vals.flip(0)
    vecs = vecs.flip(1)
    total = float(G.trace())
    var_ratio = [float(v / total) if total > 0 else 0.0 for v in vals]
    return var_ratio, vecs, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/analysis_wfast")
    ap.add_argument("--nperm", type=int, default=3000)
    args = ap.parse_args()
    seeds = all_seeds()

    deltas = []
    gf = {"EH": [], "HE": []}
    for s in seeds:
        d, g = load_delta_and_g0(s)
        deltas.append(d)
        gf["EH"].append(g["EH"])
        gf["HE"].append(g["HE"])
        print(f"seed {s}: |delta|={d.norm().item():.3f} ", flush=True)

    # ---- PCA on centered deltas ----
    mean = sum(deltas) / len(deltas)
    dc = [d - mean for d in deltas]
    var_ratio, vecs, _ = gram_pca(dc)
    n = len(seeds)
    print("variance ratios (centered):", [round(v, 4) for v in var_ratio[:6]])

    # PC directions: u_k = dc^T v_k / norm ; loadings = dc·u_k
    u = []
    for k in range(n):
        vk = vecs[:, k]
        # u_k in D-space = sum_i vk_i * dc_i
        tmp = sum(vk[i] * dc[i] for i in range(n))
        nn = tmp.norm()
        u.append(tmp / nn if nn > 1e-12 else tmp)

    scores = {}
    for i, s in enumerate(seeds):
        scores[s] = {}
        for k in range(min(4, n)):
            scores[s][f"pc{k+1}_score"] = float((dc[i] * u[k]).sum())
    # projection of each seed's initial gradient onto top PCs
    proj = {arm: {} for arm in ("EH", "HE")}
    for arm in ("EH", "HE"):
        for i, s in enumerate(seeds):
            proj[arm][s] = {}
            for k in range(min(4, n)):
                proj[arm][s][f"pc{k+1}_proj"] = float((gf[arm][i] * u[k]).sum())
            # relative share of gradient energy captured by PC1..PC4
            gnorm = gf[arm][i].norm()
            cap = math.sqrt(sum(proj[arm][s][f"pc{k+1}_proj"] ** 2 for k in range(min(4, n))))
            proj[arm][s]["cap4_rel"] = float(cap / gnorm) if gnorm > 0 else 0.0

    # ---- outcomes ----
    def p0(seed, tag):
        j = json.load(open(ROOT / "results" / "stage55e" / "seeds" / f"seed{seed}.json"))["results"][tag]
        return j["curve"][-1]["gain"], j["LE_D"], j["T80_D"]

    out_rows = {}
    for s in seeds:
        g_he, le_he, _ = p0(s, "HE/HE")
        g_eh, le_eh, _ = p0(s, "EH/EH")
        out_rows[s] = {"gain_HE": g_he, "LE_HE": le_he, "gain_EH": g_eh, "LE_EH": le_eh}

    def pear(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        return 0.0 if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])

    def pval(r0, x, ylist, n=args.nperm):
        rng = random.Random(1)
        c = 0
        for _ in range(n):
            idx = list(range(len(x)))
            rng.shuffle(idx)
            ys = [ylist[i] for i in idx]
            if abs(pear(x, ys)) >= abs(r0):
                c += 1
        return (c + 1) / (n + 1)

    tests = []
    for arm, outk in (("HE", "gain_HE"), ("HE", "LE_HE"), ("EH", "gain_EH"), ("EH", "LE_EH")):
        for pc in range(1, 5):
            x = [proj[arm][s][f"pc{pc}_proj"] for s in seeds]
            y = [out_rows[s][outk] for s in seeds]
            r0 = pear(x, y)
            tests.append({"arm": arm, "outcome": outk, "pc": pc, "pearson": round(r0, 4),
                          "perm_p": round(pval(r0, x, y), 4)})
            print(f"proj gf0({arm}) PC{pc} vs {outk}: r={r0:+.3f}")
    for pc in range(1, 5):
        x = [scores[s][f"pc{pc}_score"] for s in seeds]
        y = [out_rows[s]["gain_HE"] - out_rows[s]["gain_EH"] for s in seeds]
        r0 = pear(x, y)
        tests.append({"arm": "delta", "outcome": "gain_HE-gain_EH", "pc": pc,
                      "pearson": round(r0, 4), "perm_p": round(pval(r0, x, y), 4)})

    summary = {
        "var_ratio": var_ratio[:8],
        "cum_var": np.cumsum(var_ratio[:8]).tolist(),
        "scores": {str(s): scores[s] for s in seeds},
        "proj": {arm: {str(s): proj[arm][s] for s in seeds} for arm in ("EH", "HE")},
        "tests": tests,
    }
    with open(os.path.join(args.base, "p2_pca.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"saved {args.base}/p2_pca.json")


if __name__ == "__main__":
    main()
