"""Stage 7 - cross-domain 10-task longitudinal tracking (B strengthened).

One DLA individual per seed sequentially learns 10 tasks from 10 different
domain pools (fixed order). Each seed uses a different non-overlapping slice of
every domain pool, so the 20-seed run does not reuse data within a domain.

Data substitution note:
  Task 2 uses data/domains/qa.txt (synthetic "Q:/A:" formatted Wikipedia)
  because the original SFT raw text (~4.28M chars) cannot provide 20 unique
  340k slices. The label is recorded as synthetic_qa.

Metrics per task:
  init_loss, opt_loss, raw_slope10, norm_slope10 (loss/loss0 linear slope),
  birth PPL of the slice (probed with the pretrained base model), LE, T80.

Run: see stage7_run_all.sh for 20 seeds, 5 concurrent workers.
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
RAW_WIKI = os.path.expanduser("~/llm-lab/corpus/raw_full.txt")
DOMAIN_DIR = ROOT / "data" / "domains"

# Fixed task order; task2 is synthetic QA due SFT data budget.
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


def load_pool(path):
    return Path(path).read_text(encoding="utf-8")


def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]


def make_batch(ids, block, batch, rng, device):
    n = len(ids) - block - 1
    ix = [rng.randrange(n) for _ in range(batch)]
    x = torch.tensor([[ids[i + j] for j in range(block)] for i in ix], dtype=torch.long, device=device)
    y = torch.tensor([[ids[i + j + 1] for j in range(block)] for i in ix], dtype=torch.long, device=device)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--eta-fast", type=float, default=6e-4)
    ap.add_argument("--out", type=str, default="results/stage7")
    args = ap.parse_args()
    seed = args.seed
    span = args.train_chars + args.val_chars
    os.makedirs(os.path.join(args.out, "seeds"), exist_ok=True)
    torch.set_num_threads(2)
    device = "cpu"
    stoi = pickle.load(open(META, "rb"))["stoi"]
    cfg = TransformerDLAConfig(eta_fast=args.eta_fast)

    # Load domain pools lazily (each up to 10M chars).
    pools = {}
    for name, path in TASKS:
        pools[name] = load_pool(path)

    # Build slices for this seed.
    slices = []
    for name, _ in TASKS:
        pool = pools[name]
        start = seed * span
        if start + span > len(pool):
            raise ValueError(f"seed {seed}: {name} pool too short ({len(pool)} chars, need {start+span})")
        text = pool[start : start + span]
        train_ids = encode(text[: args.train_chars], stoi)
        val_ids = encode(text[args.train_chars :], stoi)
        slices.append((name, train_ids, val_ids))

    # Birth PPL of every task slice using the pretrained base model.
    probe = _s55.load_checkpoint(GPT)
    probe.eval()
    birth_ppl = []
    for name, _, val_ids in slices:
        rng = random.Random(700000 + seed * 1000 + len(birth_ppl))
        losses = []
        for _ in range(args.eval_batches):
            n = len(val_ids) - args.block - 1
            ix = [rng.randrange(n) for _ in range(args.eval_batch)]
            x = torch.tensor([[val_ids[i + j] for j in range(args.block)] for i in ix], dtype=torch.long)
            y = torch.tensor([[val_ids[i + j + 1] for j in range(args.block)] for i in ix], dtype=torch.long)
            _, l = probe(x, y)
            losses.append(l.item())
        birth_ppl.append(math.exp(sum(losses) / len(losses)))
    del probe
    print(f"[seed {seed}] birth PPL {[round(p,2) for p in birth_ppl]}", flush=True)

    model = _s55.load_checkpoint(DLA_GPT, cfg)
    model.train()
    state = model.make_state(device)
    train_rng = random.Random(710000 + seed)
    rows = []
    for t_idx, (name, train_ids, val_ids) in enumerate(slices):
        # fixed eval batches for this stage
        eval_rng = random.Random(720000 + seed * 100 + t_idx)
        eb = []
        for _ in range(args.eval_batches):
            n = len(val_ids) - args.block - 1
            ix = [eval_rng.randrange(n) for _ in range(args.eval_batch)]
            x = torch.tensor([[val_ids[i + j] for j in range(args.block)] for i in ix], dtype=torch.long)
            y = torch.tensor([[val_ids[i + j + 1] for j in range(args.block)] for i in ix], dtype=torch.long)
            eb.append((x, y))
        # pre PPL on current model
        model.eval(); model.set_dla_state(state)
        ev = []
        for x, y in eb:
            _, l = model(x, y); ev.append(l.item())
        model.train()
        pre = math.exp(sum(ev) / len(ev))

        losses = []
        curve = []
        for step in range(1, args.max_steps + 1):
            x, y = make_batch(train_ids, args.block, args.batch, train_rng, device)
            info = model.dla_step(x, y, state)
            losses.append(info["loss"])
            if step % args.eval_every == 0 or step == args.max_steps:
                model.eval(); model.set_dla_state(state)
                ev = []
                for xv, yv in eb:
                    _, lv = model(xv, yv); ev.append(lv.item())
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
            "task": t_idx + 1, "domain": name, "birth_ppl": birth_ppl[t_idx],
            "pre": pre, "init_loss": losses[0], "opt_loss": min(losses),
            "raw_slope10": raw_slope, "norm_slope10": norm_slope,
            "LE": cs["LE"], "T80": cs["T80"], "curve": curve,
        })
        print(f"  seed {seed} task{t_idx+1} {name}: birth={birth_ppl[t_idx]:.2f} "
              f"norm_slope={norm_slope:+.4f} LE={cs['LE']:.2f}", flush=True)

    with open(os.path.join(args.out, "seeds", f"seed{seed}.json"), "w") as f:
        json.dump({"seed": seed, "tasks": TASKS, "rows": rows}, f, indent=2)
    print(f"saved seed {seed}", flush=True)


if __name__ == "__main__":
    main()
