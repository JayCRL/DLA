#!/usr/bin/env python3
"""Stage 8 - physical near-transfer longitudinal (5 seeds, 4 difficulty tiers).

Tasks are four physics files (data/physics_*.txt) sorted by physics-keyword
density: basic -> intermediate -> advanced -> frontier. Each seed uses a
different non-overlapping slice per task. One DLA individual learns them
sequentially (40 steps each), recording norm_slope over the first 10 steps.

Run:
    python experiments/stage8_physics.py --seeds 0,1,2,3,4 --max-steps 40 --out results/stage8_physics_test
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NANO_DIR = os.path.expanduser("~/llm-lab/nanoGPT")
sys.path.insert(0, NANO_DIR)

from dla.transformer_dla import TransformerDLAConfig, make_dla_gpt_class  # noqa: E402
from model import GPT  # noqa: E402

_STAGE55 = ROOT / "experiments" / "stage55_learning_rule_development.py"
_spec55 = importlib.util.spec_from_file_location("stage55", _STAGE55)
_s55 = importlib.util.module_from_spec(_spec55)
_spec55.loader.exec_module(_s55)

DLA_GPT = make_dla_gpt_class(GPT)
META = os.path.expanduser("~/llm-lab/nanoGPT/data/zh_char/meta.pkl")
DATA = ROOT / "data"
TASKS = [
    ("physics_basic", str(DATA / "physics_basic.txt")),
    ("physics_intermediate", str(DATA / "physics_intermediate.txt")),
    ("physics_advanced", str(DATA / "physics_advanced.txt")),
    ("physics_frontier", str(DATA / "physics_frontier.txt")),
]


def load_pool(path):
    return Path(path).read_text(encoding="utf-8")


def make_batch(ids, block, batch, rng, device):
    n = len(ids) - block - 1
    ix = [rng.randrange(n) for _ in range(batch)]
    x = torch.tensor([[ids[i + j] for j in range(block)] for i in ix], dtype=torch.long, device=device)
    y = torch.tensor([[ids[i + j + 1] for j in range(block)] for i in ix], dtype=torch.long, device=device)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--eta-fast", type=float, default=6e-4)
    ap.add_argument("--out", type=str, default="results/stage8_physics_test")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    span = args.train_chars + args.val_chars
    os.makedirs(os.path.join(args.out, "seeds"), exist_ok=True)
    torch.set_num_threads(2)
    device = "cpu"
    stoi = pickle.load(open(META, "rb"))["stoi"]
    cfg = TransformerDLAConfig(eta_fast=args.eta_fast)

    pools = {name: load_pool(path) for name, path in TASKS}
    all_results = {}
    for seed in seeds:
        model = _s55.load_checkpoint(DLA_GPT, cfg)
        model.train()
        state = model.make_state(device)
        train_rng = random.Random(1_000_000 + seed)
        rows = []
        for t_idx, (name, _) in enumerate(TASKS):
            text = pools[name][seed * span : (seed + 1) * span]
            if len(text) < span:
                raise ValueError(f"seed {seed} {name} slice too short: {len(text)}")
            train_ids = [stoi[c] for c in text[: args.train_chars] if c in stoi]
            val_ids = [stoi[c] for c in text[args.train_chars :] if c in stoi]

            eval_rng = random.Random(1_010_000 + seed * 10 + t_idx)
            eb = []
            for _ in range(args.eval_batches):
                n = len(val_ids) - args.block - 1
                ix = [eval_rng.randrange(n) for _ in range(args.eval_batch)]
                x = torch.tensor([[val_ids[i + j] for j in range(args.block)] for i in ix], dtype=torch.long)
                y = torch.tensor([[val_ids[i + j + 1] for j in range(args.block)] for i in ix], dtype=torch.long)
                eb.append((x, y))

            model.eval()
            model.set_dla_state(state)
            ev = []
            for x, y in eb:
                _, l = model(x, y)
                ev.append(l.item())
            model.train()
            pre = math.exp(sum(ev) / len(ev))

            losses = []
            curve = []
            for step in range(1, args.max_steps + 1):
                x, y = make_batch(train_ids, args.block, args.batch, train_rng, device)
                info = model.dla_step(x, y, state)
                losses.append(info["loss"])
                if step % args.eval_every == 0 or step == args.max_steps:
                    model.eval()
                    model.set_dla_state(state)
                    ev = []
                    for xv, yv in eb:
                        _, lv = model(xv, yv)
                        ev.append(lv.item())
                    model.train()
                    ppl = math.exp(sum(ev) / len(ev))
                    curve.append({"step": step, "ppl": ppl, "gain": (pre - ppl) / pre})
            model.dla_sleep(state)

            first = losses[:10]
            raw_slope = float(np.polyfit(range(len(first)), first, 1)[0]) if len(first) >= 2 else 0.0
            norm = [first[i] / first[0] for i in range(len(first))] if first and first[0] > 0 else []
            norm_slope = float(np.polyfit(range(len(norm)), norm, 1)[0]) if len(norm) >= 2 else 0.0
            cs = _s55.curve_stats(curve, args.max_steps)
            rows.append({
                "task": t_idx + 1,
                "domain": name,
                "pre": pre,
                "init_loss": losses[0],
                "opt_loss": min(losses),
                "raw_slope10": raw_slope,
                "norm_slope10": norm_slope,
                "LE": cs["LE"],
                "T80": cs["T80"],
            })
            print(f"seed {seed} task{t_idx+1} {name}: norm_slope={norm_slope:+.4f}", flush=True)
        all_results[f"seed{seed}"] = rows
        with open(os.path.join(args.out, "seeds", f"seed{seed}.json"), "w") as f:
            json.dump({"seed": seed, "rows": rows}, f, indent=2)

    with open(os.path.join(args.out, "stage8.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}/stage8.json", flush=True)


if __name__ == "__main__":
    main()
