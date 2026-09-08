#!/usr/bin/env python3
"""Fair baselines for the history/future-adaptation protocol.

Compares DLA (existing Stage 5.5e P0 results) with standard continual learners:

  AdamW        - direct fine-tuning
  AdamW+replay - AdamW with a small replay buffer of previous tasks
  EWC          - AdamW with elastic weight consolidation

Each method experiences the same EH/HE curriculum (3 tasks x 40 steps) using
the same slices as Stage 5.5c/e, then adapts to the same unseen D for 40 steps.
We record full D trajectories and gain@40.

Run (per seed or via a small manager):
  python experiments/validation_baselines.py --seed 0 --out results/validation_baselines
"""

import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NANO_DIR = os.path.expanduser("~/llm-lab/nanoGPT")
sys.path.insert(0, NANO_DIR)

from model import GPT  # noqa: E402

_STAGE55 = ROOT / "experiments" / "stage55_learning_rule_development.py"
_spec55 = importlib.util.spec_from_file_location("stage55", _STAGE55)
_s55 = importlib.util.module_from_spec(_spec55)
_spec55.loader.exec_module(_s55)


def unique_parameters(model):
    seen, out = set(), []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def make_batch(ids, block, batch, rng, device):
    n = len(ids) - block - 1
    ix = [rng.randrange(n) for _ in range(batch)]
    x = torch.tensor([[ids[i + j] for j in range(block)] for i in ix], dtype=torch.long, device=device)
    y = torch.tensor([[ids[i + j + 1] for j in range(block)] for i in ix], dtype=torch.long, device=device)
    return x, y


def eval_loss_batches(model, eb):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in eb:
            _, l = model(x, y)
            losses.append(l.item())
    model.train()
    return math.exp(sum(losses) / len(losses))


def train_history(model, method, phases, d_train, eb, args, device, seed):
    """Train on EH/HE curriculum then D using one baseline method."""
    stoi = pickle.load(open(_s55.META, "rb"))["stoi"]
    opt = torch.optim.AdamW(unique_parameters(model), lr=args.lr, weight_decay=0.01)
    rng = random.Random(1_400_000 + seed + ({"adamw":0,"replay":1,"ewc":2}[method] * 10))
    replay_buf = []
    fisher = None
    star_params = None

    def compute_fisher(sample_ids):
        nonlocal fisher, star_params
        star_params = [p.detach().clone() for p in unique_parameters(model)]
        fisher = [torch.zeros_like(p) for p in unique_parameters(model)]
        grads_sq = [torch.zeros_like(p) for p in unique_parameters(model)]
        n_steps = 4
        for _ in range(n_steps):
            x, y = make_batch(sample_ids, args.block, args.batch, rng, device)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            for i, p in enumerate(unique_parameters(model)):
                if p.grad is not None:
                    grads_sq[i].add_(p.grad.detach() ** 2)
        for i in range(len(fisher)):
            fisher[i] = grads_sq[i] / n_steps

    # --- curriculum history ---
    for t_idx, (name, train_ids, val_dom) in enumerate(phases):
        # fixed eval batches for this task (not used for training loss)
        for step in range(args.cur_steps):
            x, y = make_batch(train_ids, args.block, args.batch, rng, device)
            if method == "replay" and replay_buf:
                xr, yr = replay_buf[rng.randrange(len(replay_buf))]
                x = torch.cat([x, xr], dim=0)
                y = torch.cat([y, yr], dim=0)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if method == "ewc" and fisher is not None:
                ewc_loss = loss
                for i, p in enumerate(unique_parameters(model)):
                    if fisher[i] is not None:
                        ewc_loss = ewc_loss + (args.ewc_lambda * fisher[i] * (p - star_params[i]) ** 2).sum()
                loss = ewc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unique_parameters(model), 1.0)
            opt.step()
        # store replay
        if method == "replay":
            # keep 16 random batches from this task
            for _ in range(16):
                replay_buf.append(make_batch(train_ids, args.block, args.batch, rng, device))
            if len(replay_buf) > 48:
                replay_buf = replay_buf[-48:]
        if method == "ewc":
            compute_fisher(train_ids)

    # --- D adaptation ---
    d_eb = eb["D"]
    pre = eval_loss_batches(model, d_eb)
    curve = []
    for step in range(1, args.d_steps + 1):
        if method == "replay" and replay_buf:
            # mix current D batch with replay memory
            xd, yd = make_batch(d_train, args.block, args.batch, rng, device)
            xr, yr = replay_buf[rng.randrange(len(replay_buf))]
            x = torch.cat([xd, xr], dim=0)
            y = torch.cat([yd, yr], dim=0)
        else:
            x, y = make_batch(d_train, args.block, args.batch, rng, device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if method == "ewc" and fisher is not None:
            for i, p in enumerate(unique_parameters(model)):
                if fisher[i] is not None:
                    loss = loss + (args.ewc_lambda * fisher[i] * (p - star_params[i]) ** 2).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unique_parameters(model), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.d_steps:
            ppl = eval_loss_batches(model, d_eb)
            curve.append({"step": step, "ppl": ppl, "gain": (pre - ppl) / pre})
    cs = _s55.curve_stats(curve, args.d_steps)
    return {"method": method, "pre": pre, "LE_D": cs["LE"], "T80_D": cs["T80"], "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cur-steps", type=int, default=40)
    ap.add_argument("--d-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ewc-lambda", type=float, default=1e3)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--wiki-start", type=int, default=40_000_000)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--out", type=str, default="results/validation_baselines")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "seeds"), exist_ok=True)
    torch.set_num_threads(2)
    device = "cpu"
    stoi = pickle.load(open(_s55.META, "rb"))["stoi"]
    domains, phases, d_train, d_val = _s55.build_curriculum(args.seed, args, stoi)
    eb = _s55.make_eval_batches(domains, d_val, args, device, args.seed)
    probe = _s55.load_checkpoint(GPT)
    pre_all = {dom: _s55.eval_ppl(probe, eb[dom]) for dom in domains}
    order = sorted(domains, key=lambda d: pre_all[d])
    fwd = sorted(phases, key=lambda ph: order.index(ph[2]))
    rev = list(reversed(fwd))
    rows = []
    for method in ["adamw", "replay", "ewc"]:
        for order_name, ph in [("easy_hard", fwd), ("hard_easy", rev)]:
            model = _s55.load_checkpoint(GPT)
            res = train_history(model, method, ph, d_train, eb, args, device, args.seed)
            res["order"] = order_name
            rows.append(res)
            print(f"seed {args.seed} {method} {order_name}: LE={res['LE_D']:.3f}", flush=True)
    with open(os.path.join(args.out, "seeds", f"seed{args.seed}.json"), "w") as f:
        json.dump({"seed": args.seed, "rows": rows}, f, indent=2)
    print(f"saved seed {args.seed}")


if __name__ == "__main__":
    main()
