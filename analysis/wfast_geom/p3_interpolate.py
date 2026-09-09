"""P3 - counterfactual W_fast interpolation along the per-seed history
direction Δ = W_fast(HE) - W_fast(EH), probed on the unseen domain D.

W_fast'(alpha) = W_fast_base + alpha * (W_fast_opp - W_fast_base)
  base=HE, opp=EH : alpha=0 == archived HE/HE arm; alpha=1 == archived
                    HE_body+EH_fast (destructive) arm  -> dose response
  base=EH, opp=HE : alpha=0 == EH/EH; alpha=1 == EH_body+HE_fast

Only W_fast is moved; W_slow / P / Q / m / v and the probe protocol are the
archived ones, so alpha=0 / alpha=1 endpoints double as parity checks against
the stored p0 JSON. Probe = same D protocol as Stage 5.5e (40 steps, eval/2).

Run (one job per seed/body/alpha, orchestrate in parallel):
  python analysis/wfast_geom/p3_interpolate.py --seed 3 --body HE --alpha 0.5
                                        [--out results/analysis_wfast/p3]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch

from common import Args

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import stage55_learning_rule_development as s55  # noqa: E402
import stage55c_causality as s55c  # noqa: E402


def body_path(seed, order):
    c1 = ROOT / "results" / "stage55c" / f"seed{seed}" / "bodies" / f"seed{seed}_{order}_meta.pt"
    c2 = ROOT / "results" / "stage55e" / "bodies" / f"seed{seed}_{order}_meta.pt"
    return c1 if c1.exists() else c2


def run_one(seed, body, alpha, args, device="cpu", out=None):
    out = out or "results/analysis_wfast/p3"
    torch.set_num_threads(2)
    import pickle
    stoi = pickle.load(open(s55.META, "rb"))["stoi"]
    base_o = "hard_easy" if body == "HE" else "easy_hard"
    opp_o = "easy_hard" if body == "HE" else "hard_easy"
    domains, _ph, d_train, d_val = s55.build_curriculum(seed, args, stoi)
    eb = s55.make_eval_batches(domains, d_val, args, device, seed)
    model, state = s55c.load_body(str(body_path(seed, base_o)), args, device)
    if abs(float(alpha)) > 1e-9:
        _om, opp_state = s55c.load_body(str(body_path(seed, opp_o)), args, device)
        with torch.no_grad():
            for k, ss in state.store.items():
                if k in opp_state.store:
                    d = opp_state.store[k]["w_fast"] - ss["w_fast"]
                    ss["w_fast"].add_(float(alpha) * d)
        del _om, opp_state
    model.train()
    rng = random.Random(800000 + seed)
    args.max_steps = args.d_steps
    d = s55.run_transfer(model, d_train, eb, args, rng, device, state=state, dla=True)
    cs = s55.curve_stats(d["curve"], args.d_steps)
    rec = {"seed": seed, "body": body, "alpha": float(alpha), "pre": d["pre"],
           "LE_D": cs["LE"], "T80_D": cs["T80"], "best_gain": cs["best_gain"],
           "gain40": d["curve"][-1]["gain"], "curve": d["curve"]}
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f"seed{seed}_{body}_a{float(alpha):+.2f}.json")
    with open(fn, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"seed {seed} {body} alpha={alpha:+.2f}: pre={rec['pre']:.2f} "
          f"gain40={rec['gain40']:+.4f} LE={rec['LE_D']:.3f}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--body", type=str, choices=("HE", "EH"), required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", default="results/analysis_wfast/p3")
    args = ap.parse_args()
    a = Args()
    run_one(args.seed, args.body, args.alpha, a, out=args.out)


if __name__ == "__main__":
    main()
