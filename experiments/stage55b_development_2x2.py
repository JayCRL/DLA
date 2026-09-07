"""Stage 5.5b - the DEVELOPMENT hypothesis verification experiment (2x2).

Design:
    factor A  curriculum history : easy->hard  |  hard->easy
    factor B  learner             : DLA-static (phi frozen) | DLA-meta (phi develops)

5 seeds x 4 conditions.  Every individual is finally probed on the same held-out
unseen domain D (science slice never used in the curriculum).

Headline quantities:
    LE_D, T80_D            (difficulty-normalised learning efficiency on D)
    DG    = LE_D_meta - LE_D_static          (development gain, per history)
    DG_T  = T80_D_static - T80_D_meta        (positive = meta learns D faster)
    interaction_T = DG_T(hard->easy) - DG_T(easy->hard)
    interaction_LE = DG(hard->easy) - DG(easy->hard)
    phi distance     = ||phi_final(easy->hard) - phi_final(hard->easy)||

The question is NOT "which training trick is better" but:
    Is History x Development a stable, repeatable, transferable effect?

Run as two parallel halves to use the CPU:
    OMP_NUM_THREADS=3 python ... --seeds 0,1,2 --out results/stage55b/part_012.json
    OMP_NUM_THREADS=3 python ... --seeds 3,4   --out results/stage55b/part_34.json
then combine with stage55b_combine.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NANO_DIR = os.path.expanduser("~/llm-lab/nanoGPT")
sys.path.insert(0, NANO_DIR)

# reuse Stage 5.5 data/runner machinery
_STAGE55 = ROOT / "experiments" / "stage55_learning_rule_development.py"
_spec = importlib.util.spec_from_file_location("stage55", _STAGE55)
_stage55 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stage55)

from dla.transformer_dla import TransformerDLAConfig, make_dla_gpt_class  # noqa: E402
from model import GPT  # noqa: E402

DLA_GPT = make_dla_gpt_class(GPT)


def run_condition(seed, order_name, learner, phases, eb, d_train, args, device):
    """One individual through one curriculum history + transfer probe D."""
    code = seed * 10 + (0 if order_name == "easy_hard" else 1) * 2 + (0 if learner == "static" else 1)
    train_rng = random.Random(500000 + code)
    meta_rng = random.Random(510000 + code)
    cfg = TransformerDLAConfig(eta_fast=args.eta_fast)
    model = _stage55.load_checkpoint(DLA_GPT, cfg)
    model.train()
    state = model.make_state(device)
    meta_opt = torch.optim.Adam(model.tempos.parameters(), lr=args.meta_lr) if learner == "meta" else None

    rec = {"stage_pre": [], "curve": [], "phi_trace": []}
    for t_idx, (name, train_ids, val_dom) in enumerate(phases):
        if meta_opt is not None and t_idx > 0:
            before = model.tempos.snapshot()
            batches = [_stage55.get_batch(train_ids, args.block, args.meta_batch, meta_rng, device)
                       for _ in range(args.meta_unroll_len)]
            model.meta_update_phi(state, batches, meta_opt)
            after = model.tempos.snapshot()
            rec["phi_trace"].append({"stage": t_idx, "before": before, "after": after,
                                     "norm": _stage55.tempo_delta_norm(before, after)})
        pre = _stage55.eval_ppl(model, eb[val_dom], state=state)
        rec["stage_pre"].append(pre)
        curve = []
        for step in range(1, args.max_steps + 1):
            x, y = _stage55.get_batch(train_ids, args.block, args.batch, train_rng, device)
            model.dla_step(x, y, state)
            if step % args.eval_every == 0 or step == args.max_steps:
                ppl = _stage55.eval_ppl(model, eb[val_dom], state=state)
                curve.append({"step": step, "ppl": ppl, "gain": (pre - ppl) / pre})
        rec["curve"].append(curve)
        model.dla_sleep(state)

    d = _stage55.run_transfer(model, d_train, eb, args, random.Random(520000 + seed), device,
                              state=state, dla=True)
    cs = _stage55.curve_stats(d["curve"], args.max_steps)
    return {
        "seed": seed, "order": order_name, "learner": learner,
        "LE_D": cs["LE"], "T80_D": cs["T80"], "raw_lambda_D": _stage55.raw_lambda(d["curve"], args.target_gain),
        "best_gain_D": cs["best_gain"], "final_gain_D": cs["final_gain"],
        "phi_norms": [tr["norm"] for tr in rec["phi_trace"]],
        "phi_final": model.tempos.snapshot() if learner == "meta" else None,
        "stage_pre": rec["stage_pre"],
        "D_pre": d["pre"],
        "D_curve": d["curve"],
    }


def mean_std(vals):
    n = len(vals)
    mu = sum(vals) / n
    std = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5
    return mu, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--target-gain", type=float, default=0.04)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--wiki-start", type=int, default=40_000_000)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--eta-fast", type=float, default=6e-4)
    ap.add_argument("--meta-lr", type=float, default=0.1)
    ap.add_argument("--meta-batch", type=int, default=4)
    ap.add_argument("--meta-unroll-len", type=int, default=2)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--out", type=str, default="results/stage55b/part.json")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_num_threads(3)
    device = "cpu"
    stoi = pickle.load(open(_stage55.META, "rb"))["stoi"]

    rows = []
    for seed in args.seeds:
        domains, phases, d_train, d_val = _stage55.build_curriculum(seed, args, stoi)
        eb = _stage55.make_eval_batches(domains, d_val, args, device, seed)
        probe = _stage55.load_checkpoint(GPT)
        pre_all = {dom: _stage55.eval_ppl(probe, eb[dom]) for dom in domains}
        order = sorted(domains, key=lambda d: pre_all[d])
        fwd = sorted(phases, key=lambda ph: order.index(ph[2]))
        rev = list(reversed(fwd))
        for order_name, ph in (("easy_hard", fwd), ("hard_easy", rev)):
            for learner in ("static", "meta"):
                t0 = time.time()
                print(f"[seed {seed}] {order_name} / {learner} ...", flush=True)
                row = run_condition(seed, order_name, learner, ph, eb, d_train, args, device)
                rows.append(row)
                print(f"  LE_D={row['LE_D']:.3f} T80_D={row['T80_D']} phi_norms={[round(n,4) for n in row['phi_norms']]} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- summary ----------------
    summary = {"seeds": args.seeds}
    def cell(order, learner):
        return [r for r in rows if r["order"] == order and r["learner"] == learner]

    for order in ("easy_hard", "hard_easy"):
        for learner in ("static", "meta"):
            rs = cell(order, learner)
            if not rs:
                continue
            led = [r["LE_D"] for r in rs]
            t80 = [r["T80_D"] for r in rs]
            mu_le, sd_le = mean_std(led)
            mu_t, sd_t = mean_std(t80)
            summary[f"{order}_{learner}"] = {
                "LE_D": led, "LE_D_mean": mu_le, "LE_D_std": sd_le,
                "T80_D": t80, "T80_D_mean": mu_t, "T80_D_std": sd_t,
            }
        # development gains per history
        mets = cell(order, "meta")
        sts = cell(order, "static")
        if mets and sts:
            dg = [m["LE_D"] - s["LE_D"] for m, s in zip(mets, sts)]
            dgt = [s["T80_D"] - m["T80_D"] for m, s in zip(mets, sts)]
            summary[f"{order}_DG"] = {"per_seed": dg, "mean": sum(dg) / len(dg)}
            summary[f"{order}_DG_T"] = {"per_seed": dgt, "mean": sum(dgt) / len(dgt)}
    # history x development interaction
    if "easy_hard_DG" in summary and "hard_easy_DG" in summary:
        summary["interaction_LE"] = summary["hard_easy_DG"]["mean"] - summary["easy_hard_DG"]["mean"]
    if "easy_hard_DG_T" in summary and "hard_easy_DG_T" in summary:
        summary["interaction_T"] = summary["hard_easy_DG_T"]["mean"] - summary["easy_hard_DG_T"]["mean"]
    # phi distance between histories (meta only)
    eh_meta = cell("easy_hard", "meta")
    he_meta = cell("hard_easy", "meta")
    if eh_meta and he_meta:
        dists = []
        for a, b in zip(eh_meta, he_meta):
            if a["phi_final"] and b["phi_final"]:
                dists.append(_stage55.tempo_delta_norm(a["phi_final"], b["phi_final"]))
        summary["phi_distance_histories"] = {"per_seed": dists, "mean": sum(dists) / len(dists) if dists else None}

    print("\n===== Stage 5.5b summary =====", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
