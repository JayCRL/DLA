"""P4 - consolidation-trace geometry: do Q (eligibility) and W_slow also carry
a history signal, and does the future gradient align with / is it predicted by
those history-contrast directions?

For fields f in {w_fast, q, slow}:
  * per-seed delta D_f = f(HE) - f(EH)   (geometry null vs same-history cross-seed)
  * cos(g0, D_f) where g0 = gated D-probe gradient at step 0 (native arms)
  * cross-seed / shuffled nulls for that cos
  * correlation of cos with gain@40 (HE/HE, EH/EH)  [+ permutation p]

slow weights are read from the body ckpt['model'] state dict
(state key path + '.weight'); q from ckpt['store'][key]['q'].

Run: python analysis/wfast_geom/p4_consolidation_geom.py --base results/analysis_wfast
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
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


def flat_of(ckpt, field, keys):
    parts = []
    if field == "slow":
        md = ckpt["model"]
        for k in keys:
            parts.append(md[k + ".weight"].reshape(-1).float())
    else:
        for k in keys:
            parts.append(ckpt["store"][k][field].reshape(-1).float())
    return torch.cat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/analysis_wfast")
    ap.add_argument("--nperm", type=int, default=2000)
    args = ap.parse_args()
    from common import Args
    a = Args()
    device = "cpu"
    stoi = pickle.load(open(s55.META, "rb"))["stoi"]
    seeds = all_seeds()

    # per-seed: delta vectors per field + g0 (native arms)
    keys_of = {}
    D = {f: {} for f in ("wf", "q", "slow")}
    g0 = {"HE": {}, "EH": {}}
    for s in seeds:
        eh = load_body_ckpt(s, "easy_hard")
        he = load_body_ckpt(s, "hard_easy")
        keys = canonical_keys(eh["store"].keys())
        keys_of[s] = keys
        for tag, f in (("wf", "w_fast"), ("q", "q"), ("slow", "slow")):
            D[tag][s] = flat_of(he, f, keys) - flat_of(eh, f, keys)
        for arm, order in (("HE", "hard_easy"), ("EH", "easy_hard")):
            _d, _p, d_train, _dv = s55.build_curriculum(s, a, stoi)
            model, state = s55c.load_body(str(_bp(s, order)), a, device)
            model.train()
            rng = random.Random(800000 + s)
            x, y = s55.get_batch(d_train, a.block, a.batch, rng, device)
            model.set_dla_state(state)
            model.zero_grad(set_to_none=True)
            _lg, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.dla_cfg.grad_clip)
            with torch.no_grad():
                parts = []
                for k in keys:
                    mod = model.key_modules[k]
                    gg = mod.weight.grad
                    if gg is None:
                        continue
                    parts.append((F.softplus(state.store[k]["p"]) * gg).reshape(-1).float())
            g0[arm][s] = torch.cat(parts)
        print(f"seed {s} done", flush=True)

    def cos(u, v):
        return float((u * v).sum() / (u.norm() * v.norm() + 1e-12))

    def gain(s, tag):
        j = json.load(open(ROOT / "results" / "stage55e" / "seeds" / f"seed{s}.json"))["results"][tag]
        return j["curve"][-1]["gain"]

    def pear(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        return 0.0 if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])

    out = {"per_seed": {}}
    summary = {}
    for tag in ("wf", "q", "slow"):
        # geometry: within-seed delta norm & its size vs cross-seed null
        dn = [float(D[tag][s].norm()) for s in seeds]
        # cross-seed null (same-history field distance)
        cross = {"HE": [], "EH": []}
        stores = {"HE": {}, "EH": {}}
        for s in seeds:
            for arm, order in (("HE", "hard_easy"), ("EH", "easy_hard")):
                ck = load_body_ckpt(s, order)
                stores[arm][s] = flat_of(ck, {"wf": "w_fast", "q": "q", "slow": "slow"}[tag],
                                         keys_of[s])
        for arm in ("HE", "EH"):
            for i in range(12):
                for j in range(i + 1, 12):
                    cross[arm].append(float((stores[arm][seeds[i]] - stores[arm][seeds[j]]).norm()))
        # alignment with g0
        al = {arm: [cos(g0[arm][s], D[tag][s]) for s in seeds] for arm in ("HE", "EH")}
        yhe = [gain(s, "HE/HE") for s in seeds]
        yeh = [gain(s, "EH/EH") for s in seeds]
        def pval(r0, x, n):
            rngp = random.Random(3)
            c = 0
            for _ in range(n):
                idx = list(range(len(x))); rngp.shuffle(idx)
                if abs(pear([x[i] for i in idx], yhe)) >= abs(r0):
                    c += 1
            return (c + 1) / (n + 1)
        rh = pear(al["HE"], yhe)
        re_ = pear(al["EH"], yeh)
        summary[tag] = {
            "delta_norm_mean": float(np.mean(dn)),
            "delta_norm_sd": float(np.std(dn)),
            "cross_seed_norm_HE_mean": float(np.mean(cross["HE"])),
            "cross_seed_norm_EH_mean": float(np.mean(cross["EH"])),
            "cos_true_HE_mean": float(np.mean(al["HE"])),
            "cos_true_EH_mean": float(np.mean(al["EH"])),
            "r_cos_HE_gain": rh, "perm_p_HE": pval(rh, al["HE"], args.nperm),
            "r_cos_EH_gain": re_,
        }
        out["per_seed"][tag] = {"delta": {str(s): float(D[tag][s].norm()) for s in seeds},
                                "cos_HE": {str(s): al["HE"][i] for i, s in enumerate(seeds)},
                                "cos_EH": {str(s): al["EH"][i] for i, s in enumerate(seeds)}}
        print(f"[{tag}] |D|={summary[tag]['delta_norm_mean']:.3f} vs cross-seed "
              f"HE {summary[tag]['cross_seed_norm_HE_mean']:.3f} | cos_true_HE mean "
              f"{summary[tag]['cos_true_HE_mean']:+.4f} | r(cos_HE,gain)={rh:+.3f} "
              f"(perm p={summary[tag]['perm_p_HE']:.3f}) | r(cos_EH,gain)={re_:+.3f}")
    out["summary"] = summary
    with open(os.path.join(args.base, "p4_consolidation_geom.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.base}/p4_consolidation_geom.json")


if __name__ == "__main__":
    main()
