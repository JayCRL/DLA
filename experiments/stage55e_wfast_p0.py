"""Stage 5.5e P0 - Does W_fast encode memory or change future learning dynamics?

Design (fixed protocol, no model/phi changes):
  * seeds: 0..n-1 (default 5)
  * bodies: easy_hard(meta) and hard_easy(meta), reused from Stage 5.5c where
    available; missing bodies are created by the same run_meta_history function.
  * four combinations:
        EH/EH, HE/HE, EH_body+HE_Wfast, HE_body+EH_Wfast
  * probe: same unseen D, same update rule (DLA dla_step), same data/budget,
    same eval batches.  Full trajectory is recorded at eval_every==2.

Interpretation:
  if the HE_Wfast advantage already exists at t=0 and the learning slopes are
  parallel  -> W_fast carries MEMORY (content advantage)
  if initial PPL is similar but the slope with HE_Wfast is steeper / more stable
  -> W_fast changes FUTURE LEARNING DYNAMICS

Run:
    ~/llm-lab/venv/bin/python experiments/stage55e_wfast_p0.py --seeds 0,1,2,3,4 \
        --cur-steps 40 --d-steps 40 --out results/stage55e
(per-seed parallel is fine)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_STAGE55C = ROOT / "experiments" / "stage55c_causality.py"
_spec = importlib.util.spec_from_file_location("stage55c", _STAGE55C)
_s55c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s55c)

_STAGE55 = ROOT / "experiments" / "stage55_learning_rule_development.py"
_spec55 = importlib.util.spec_from_file_location("stage55", _STAGE55)
_s55 = importlib.util.module_from_spec(_spec55)
_spec55.loader.exec_module(_s55)

import pickle  # noqa: E402


def copy_wfast(src_state, dst_state):
    for key, ss in src_state.store.items():
        if key in dst_state.store:
            dst_state.store[key]["w_fast"].copy_(ss["w_fast"])


def probe_trajectory(model, state, d_train, eb, args, device, seed):
    rng = random.Random(800000 + seed)
    args.max_steps = args.d_steps
    args.eval_every = 2
    d = _s55.run_transfer(model, d_train, eb, args, rng, device, state=state, dla=True)
    cs = _s55.curve_stats(d["curve"], args.d_steps)
    d["LE_D"] = cs["LE"]
    d["T80_D"] = cs["T80"]
    return d


def run_seed(seed, args, device):
    stoi = pickle.load(open(_s55.META, "rb"))["stoi"]
    domains, phases, d_train, d_val = _s55.build_curriculum(seed, args, stoi)
    eb = _s55.make_eval_batches(domains, d_val, args, device, seed)
    body_dir = os.path.join(args.out, "bodies")
    os.makedirs(body_dir, exist_ok=True)
    body_paths = {}
    for order in ("easy_hard", "hard_easy"):
        alt = os.path.join(_STAGE55C.parent.parent, "results", "stage55c", f"seed{seed}", "bodies",
                           f"seed{seed}_{order}_meta.pt")
        path = alt if os.path.exists(alt) else os.path.join(body_dir, f"seed{seed}_{order}_meta.pt")
        body_paths[order] = path
        if not os.path.exists(path):
            # create missing body using 5.5c curriculum runner (meta=True)
            from model import GPT as _GPT
            p = _s55.load_checkpoint(_GPT)
            pre = {dom: _s55.eval_ppl(p, eb[dom]) for dom in domains}
            order = sorted(domains, key=lambda d: pre[d])
            fwd = sorted(phases, key=lambda ph_: order.index(ph_[2]))
            rev = list(reversed(fwd))
            ph = fwd if order == "easy_hard" else rev
            print(f"[seed {seed}] creating missing body {order} ...", flush=True)
            body_paths[order] = _s55c.run_meta_history(seed, order, ph, eb, d_train, args, device, body_dir)[0]
        else:
            print(f"[seed {seed}] using existing body {order}", flush=True)

    EH = "easy_hard"
    HE = "hard_easy"
    results = {}
    combos = [
        ("EH/EH", EH, None),
        ("HE/HE", HE, None),
        ("EH_body+HE_fast", EH, HE),
        ("HE_body+EH_fast", HE, EH),
    ]
    for tag, base_o, src_o in combos:
        model, state = _s55c.load_body(body_paths[base_o], args, device)
        if src_o is not None:
            src_model, src_state = _s55c.load_body(body_paths[src_o], args, device)
            copy_wfast(src_state, state)
            del src_model, src_state
        d = probe_trajectory(model, state, d_train, eb, args, device, seed)
        results[tag] = d
        print(f"[seed {seed}] {tag}: pre={d['pre']:.2f} LE_D={d['LE_D']:.3f} T80={d['T80_D']}", flush=True)
        del model, state
    os.makedirs(os.path.join(args.out, "seeds"), exist_ok=True)
    with open(os.path.join(args.out, "seeds", f"seed{seed}.json"), "w") as f:
        json.dump({"seed": seed, "results": results}, f, indent=2, ensure_ascii=False)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--cur-steps", type=int, default=40)
    ap.add_argument("--d-steps", type=int, default=40)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--wiki-start", type=int, default=40_000_000)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--eta-fast", type=float, default=6e-4)
    ap.add_argument("--meta-lr", type=float, default=0.1)
    ap.add_argument("--meta-batch", type=int, default=4)
    ap.add_argument("--meta-unroll-len", type=int, default=2)
    ap.add_argument("--out", type=str, default="results/stage55e")
    args = ap.parse_args()
    args.seeds = [int(x) for x in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(2)
    device = "cpu"
    all_out = {}
    for seed in args.seeds:
        all_out[f"seed{seed}"] = run_seed(seed, args, device)
    with open(os.path.join(args.out, "p0.json"), "w") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}/p0.json", flush=True)


if __name__ == "__main__":
    main()
