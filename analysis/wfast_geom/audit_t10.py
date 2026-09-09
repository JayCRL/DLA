"""Audit T10 - ten-task cross-domain continual-learning benchmark with sleep
variants, on the Chinese char-GPT. Forward adaptation (per-task LE/gain on the
task's own slice) + backward retention (end-of-history ppl on every earlier task
slice) are recorded per seed/variant, mirroring stage7_cross_domain but with the
audit sleep variants (direct / nocons / nosleep / full) and per-task state saved
for analysis. This is the A2/B2 backbone.

Usage (one job per seed+variant, orchestrate in parallel like audit_b3):
  python analysis/wfast_geom/audit_t10.py --seed 0 --variant direct --out ~/llm-lab/dla_audit_t10
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

from common import Args as ProtocolArgs
from dla.transformer_dla import TransformerDLAConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
NANO = os.path.expanduser("~/llm-lab/nanoGPT")
sys.path.insert(0, NANO)
sys.path.insert(0, str(ROOT / "experiments"))
import stage55_learning_rule_development as s55  # noqa: E402
import stage55c_causality as s55c  # noqa: E402
from model import GPT  # noqa: E402
from audit_b3 import variant_sleep  # noqa: E402  (nosleep handled by caller)


DOMAIN_DIR = ROOT / "data" / "domains"
RAW_WIKI = os.path.expanduser("~/llm-lab/corpus/raw_full.txt")
TASKS = [
    ("wiki_general", str(RAW_WIKI)),
    ("synthetic_qa", str(DOMAIN_DIR / "qa.txt")),
    ("physics", str(DOMAIN_DIR / "physics.txt")),
    ("math", str(DOMAIN_DIR / "math.txt")),
    ("computer", str(DOMAIN_DIR / "computer.txt")),
    ("biology", str(DOMAIN_DIR / "biology.txt")),
    ("economics", str(DOMAIN_DIR / "economics.txt")),
    ("law", str(DOMAIN_DIR / "law.txt")),
    ("history", str(DOMAIN_DIR / "history.txt")),
    ("literature", str(DOMAIN_DIR / "literature.txt")),
]


def make_batch(ids, block, batch, rng, device):
    n = len(ids) - block - 1
    ix = [rng.randrange(n) for _ in range(batch)]
    x = torch.tensor([[ids[i + j] for j in range(block)] for i in ix], dtype=torch.long, device=device)
    y = torch.tensor([[ids[i + j + 1] for j in range(block)] for i in ix], dtype=torch.long, device=device)
    return x, y


def eval_ppl_ids(model, state, ids, block, batch, nb, rng, device):
    model.eval()
    model.set_dla_state(state)
    n = len(ids) - block - 1
    ls = []
    for _ in range(nb):
        ix = [rng.randrange(n) for _ in range(batch)]
        x = torch.stack([torch.tensor(ids[i:i + block], dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(ids[i + 1:i + 1 + block], dtype=torch.long) for i in ix])
        _, l = model(x, y)
        ls.append(l.item())
    model.train()
    return float(np.mean(ls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=("full", "direct", "nocons", "nosleep"), required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--out", default="~/llm-lab/dla_audit_t10")
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    device = "cpu"
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "3")))
    stoi = pickle.load(open(s55.META, "rb"))["stoi"]
    enc = lambda s: [stoi[c] for c in s if c in stoi]
    span = args.train_chars + args.val_chars

    slices = []
    for name, path in TASKS:
        text = Path(path).read_text(encoding="utf-8")
        start = args.seed * span
        seg = text[start:start + span]
        if len(seg) < span:
            raise ValueError(f"seed {args.seed}: {name} pool too short ({len(text)})")
        slices.append((name, enc(seg[:args.train_chars]), enc(seg[args.train_chars:])))

    model = s55.load_checkpoint(s55c.DLA_GPT, TransformerDLAConfig(eta_fast=6e-4))
    model.train()
    state = model.make_state(device)
    train_rng = random.Random(710000 + args.seed)

    rows = []
    for t_idx, (name, train_ids, val_ids) in enumerate(slices):
        eval_rng = random.Random(720000 + args.seed * 100 + t_idx)
        pre = math.exp(eval_ppl_ids(model, state, val_ids, args.block, args.eval_batch,
                                    args.eval_batches, eval_rng, device))
        losses = []
        for step in range(1, args.steps + 1):
            x, y = make_batch(train_ids, args.block, args.batch, train_rng, device)
            info = model.dla_step(x, y, state)
            losses.append(info["loss"])
        post = math.exp(eval_ppl_ids(model, state, val_ids, args.block, args.eval_batch,
                                     args.eval_batches, random.Random(730000 + args.seed + t_idx), device))
        first = losses[:10]
        raw_slope = float(np.polyfit(range(len(first)), first, 1)[0]) if len(first) >= 2 else 0.0
        norm = [first[i] / first[0] for i in range(len(first))] if first and first[0] > 0 else []
        norm_slope = float(np.polyfit(range(len(norm)), norm, 1)[0]) if len(norm) >= 2 else 0.0
        # forward-adaptation proxy: ppl drop over the task (lower better)
        rows.append({"task": t_idx + 1, "domain": name, "pre_ppl": pre, "post_ppl": post,
                     "gain": (pre - post) / pre, "raw_slope10": raw_slope,
                     "norm_slope10": norm_slope, "min_loss": float(min(losses))})
        if args.variant != "nosleep":
            variant_sleep(model, state, args.variant, perm_seed=args.seed * 1000 + t_idx)
        print(f"seed {args.seed} {args.variant} task{t_idx+1} {name}: gain={rows[-1]['gain']:+.4f}", flush=True)

    # backward retention: ppl on each task slice at the very end
    end_ppl = {}
    for t_idx, (name, _tr, val_ids) in enumerate(slices):
        end_ppl[name] = math.exp(eval_ppl_ids(model, state, val_ids, args.block, args.eval_batch,
                                              args.eval_batches, random.Random(740000 + args.seed + t_idx), device))
    with open(os.path.join(out, f"t10_seed{args.seed}_{args.variant}.json"), "w") as f:
        json.dump({"seed": args.seed, "variant": args.variant, "rows": rows,
                   "end_ppl": end_ppl}, f, indent=1, ensure_ascii=False)
    print(f"saved t10_seed{args.seed}_{args.variant}.json", flush=True)


if __name__ == "__main__":
    import math
    main()
