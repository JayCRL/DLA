"""Audit B3 - consolidation-pathway ablation (diagnosis only; no core-algorithm
changes, no tuning). Rebuilds an EH/HE meta-history body with one of:

  variant = qonly  : sleep writes only beta*Q            (direct W_fast->W_slow OFF)
  variant = direct : sleep writes only cfd*W_fast        (Q->W_slow OFF)
  variant = nocons : sleep writes nothing (both OFF; decays & moment resets kept)
  variant = full   : exact original sleep (parity check against saved bodies)

then probes the body on the unseen domain D with the identical Stage 5.5e
protocol (40 steps, eval every 2, rng 800000+seed) and saves probe outcome.

Run: python analysis/wfast_geom/audit_b3.py --seed 0 --order EH --variant full \
         --out results/audit
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

import torch

from common import Args as ProtocolArgs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import stage55_learning_rule_development as s55  # noqa: E402
import stage55c_causality as s55c  # noqa: E402
from dla.transformer_dla import TransformerDLAConfig  # noqa: E402
from dla.transformer_dla import DLAState  # noqa: E402


class Args:
    """Identical to Stage 5.5c/e defaults used to build the existing bodies."""
    cur_steps = 40
    d_steps = 40
    block = 128
    batch = 32
    eval_every = 2
    eval_batch = 16
    eval_batches = 4
    wiki_start = 40_000_000
    train_chars = 300_000
    val_chars = 40_000
    eta_fast = 6e-4
    meta_lr = 0.1
    meta_batch = 4
    meta_unroll_len = 2


@torch.no_grad()
def variant_sleep(model, state, variant):
    """Copy of dla_sleep with the write pathway selected by `variant`."""
    tv = model.tempos.values()
    for key, mod in model.key_modules.items():
        s = state.store[key]
        if variant == "qonly":
            add = tv["consolidate_beta"] * s["q"]
        elif variant == "direct":
            add = tv["consolidate_fast_direct"] * s["w_fast"]
        elif variant == "nocons":
            add = None
        else:  # full == original
            add = tv["consolidate_beta"] * s["q"] + tv["consolidate_fast_direct"] * s["w_fast"]
        if add is not None:
            mod.weight.add_(add)
        s["w_fast"].mul_(tv["consolidate_fast_decay"])
        s["q"].mul_(tv["consolidate_q_decay"])
    state.reset_moments()


def run_history_variant(seed, order_name, phases, eb, d_train, args, device,
                        body_dir, variant, meta=False):
    """Mirror of stage55c.run_meta_history with a sleep variant (phi meta-updates ON)."""
    train_rng = random.Random(600000 + seed * 10 + (0 if order_name == "easy_hard" else 1))
    meta_rng = random.Random(610000 + seed * 10 + (0 if order_name == "easy_hard" else 1))
    cfg = TransformerDLAConfig(eta_fast=args.eta_fast)
    model = s55.load_checkpoint(s55c.DLA_GPT, cfg)
    model.train()
    state = model.make_state(device)
    meta_opt = torch.optim.Adam(model.tempos.parameters(), lr=args.meta_lr)
    trace = []
    for t_idx, (name, train_ids, val_dom) in enumerate(phases):
        if meta and t_idx > 0:
            batches = [s55.get_batch(train_ids, args.block, args.meta_batch, meta_rng, device)
                       for _ in range(args.meta_unroll_len)]
            model.meta_update_phi(state, batches, meta_opt)
        for _step in range(1, args.cur_steps + 1):
            x, y = s55.get_batch(train_ids, args.block, args.batch, train_rng, device)
            model.dla_step(x, y, state)
        variant_sleep(model, state, variant)
        trace.append({"stage": t_idx, "phi": model.tempos.snapshot()})
    path = os.path.join(body_dir, f"seed{seed}_{order_name}_meta.pt")
    s55c.save_body(path, model, state)
    return path, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--order", type=str, choices=("EH", "HE"), required=True)
    ap.add_argument("--variant", type=str, choices=("qonly", "direct", "nocons", "full"), required=True)
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--keep-body", action="store_true")
    args = ap.parse_args()
    order = "easy_hard" if args.order == "EH" else "hard_easy"
    seed = args.seed
    a = Args()
    a.max_steps = a.d_steps  # probe protocol (mirror stage55e probe_trajectory)
    device = "cpu"
    import os as _os
    torch.set_num_threads(int(_os.environ.get("TORCH_THREADS", "4")))
    stoi = pickle.load(open(s55.META, "rb"))["stoi"]

    domains, phases, d_train, d_val = s55.build_curriculum(seed, a, stoi)
    eb = s55.make_eval_batches(domains, d_val, a, device, seed)
    # order domains by pretrained birth PPL exactly like stage55c
    from model import GPT
    probe_gpt = s55.load_checkpoint(GPT)
    pre_all = {dom: s55.eval_ppl(probe_gpt, eb[dom]) for dom in domains}
    order_idx = {d: i for i, d in enumerate(sorted(domains, key=lambda d: pre_all[d]))}
    fwd = sorted(phases, key=lambda ph: order_idx[ph[2]])
    rev = list(reversed(fwd))
    ph = fwd if args.order == "EH" else rev
    del probe_gpt

    body_dir = os.path.join(args.out, args.variant)
    os.makedirs(body_dir, exist_ok=True)
    path, trace = run_history_variant(seed, order, ph, eb, d_train, a, device, body_dir, args.variant, meta=True)

    # ---- probe on D (identical to Stage 5.5e native probe) ----
    model, state = s55c.load_body(path, a, device)
    # pre-probe (post-history) state norms for the diagnostic
    qn = math.sqrt(sum(float(state.store[k]["q"].pow(2).sum()) for k in state.store))
    wn = math.sqrt(sum(float(state.store[k]["w_fast"].pow(2).sum()) for k in state.store))
    sn = math.sqrt(sum(float(mp.pow(2).sum()) for mp in model.parameters() if mp.ndim >= 2))
    model.train()
    rng = random.Random(800000 + seed)
    d = s55.run_transfer(model, d_train, eb, a, rng, device, state=state, dla=True)
    cs = s55.curve_stats(d["curve"], a.d_steps)
    rec = {"seed": seed, "order": args.order, "variant": args.variant, "pre": d["pre"],
           "gain40": d["curve"][-1]["gain"], "LE_D": cs["LE"], "T80_D": cs["T80"],
           "Q_norm": qn, "Wfast_norm": wn, "Wslow_norm": sn}
    rec_dir = os.path.join(args.out, args.variant)
    os.makedirs(rec_dir, exist_ok=True)
    with open(os.path.join(rec_dir, f"probe_seed{seed}_{args.order}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    if not args.keep_body and os.path.exists(path):
        os.remove(path)
    print(f"seed {seed} {args.order} {args.variant}: pre={rec['pre']:.2f} "
          f"gain40={rec['gain40']:+.4f} LE={rec['LE_D']:.3f}", flush=True)


if __name__ == "__main__":
    main()
