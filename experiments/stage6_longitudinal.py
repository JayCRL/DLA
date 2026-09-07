"""Stage 6 - within-lifetime B test: more experience -> faster learning on new tasks.

Design (avoids the Stage 5 difficulty confound):
  * Four tasks are four DISJOINT slices of the same Science-Wikipedia pool, with
    comparable birth perplexity (we select the 4 slices closest to the median
    among a random candidate set).
  * A single DLA individual (same seed/model/state) learns them sequentially.
  * For every stage we record:
      - per-step training loss (raw)  -> raw slope over first 10 steps
      - normalised loss slope         -> loss(t)/loss(0) slope (difficulty-normalised)
      - test-PPL gain, LE, T80
  * If learning efficiency increases over development, later stages should have
    MORE NEGATIVE normalised slopes / faster LE.

Run:
    ~/llm-lab/venv/bin/python experiments/stage6_longitudinal.py \
        --seeds 0,1,2,3,4,5,6,7,8,9 --max-steps 40 --out results/stage6
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
SCIENCE_POOL = ROOT / "science_pool.txt"
META = os.path.expanduser("~/llm-lab/nanoGPT/data/zh_char/meta.pkl")


def load_pool_text():
    return SCIENCE_POOL.read_text(encoding="utf-8")


def choose_slices(seed, pool, args, stoi):
    """Choose 4 disjoint, comparable-difficulty slices for this seed."""
    span = args.train_chars + args.val_chars
    max_start = max(1, len(pool) // span - 1)
    rng = random.Random(900000 + seed)
    slots = sorted(rng.sample(range(max_start), min(args.candidates, max_start)))
    # birth PPL for each candidate (only val part)
    probe = _s55.load_checkpoint(GPT)
    ppls = []
    for start in slots:
        text = pool[start * span : (start + 1) * span]
        val_ids = [stoi[c] for c in text[args.train_chars :] if c in stoi]
        # evaluate with a quick fixed batch
        r = random.Random(1000 + start)
        losses = []
        for _ in range(4):
            n = len(val_ids) - args.block - 1
            if n <= 0:
                break
            ix = [r.randrange(n) for _ in range(args.eval_batch)]
            x = torch.tensor([[val_ids[i + j] for j in range(args.block)] for i in ix], dtype=torch.long)
            y = torch.tensor([[val_ids[i + j + 1] for j in range(args.block)] for i in ix], dtype=torch.long)
            _, loss = probe(x, y)
            losses.append(loss.item())
        ppl = math.exp(sum(losses) / len(losses)) if losses else float("inf")
        ppls.append((start, ppl))
    ppls.sort(key=lambda z: z[1])
    med = ppls[len(ppls) // 2][1]
    chosen = sorted(ppls, key=lambda z: abs(z[1] - med))[: args.n_tasks]
    chosen.sort()
    return chosen, ppls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--eta-fast", type=float, default=6e-4)
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--n-tasks", type=int, default=4)
    ap.add_argument("--out", type=str, default="results/stage6")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.join(args.out, "seeds"), exist_ok=True)
    torch.set_num_threads(2)
    device = "cpu"
    stoi = pickle.load(open(META, "rb"))["stoi"]
    pool = load_pool_text()
    cfg = TransformerDLAConfig(eta_fast=args.eta_fast)

    results = {}
    for seed in args.seeds:
        chosen, all_ppl = choose_slices(seed, pool, args, stoi)
        print(f"[seed {seed}] chosen PPL {[round(p,2) for _,p in chosen]}", flush=True)
        model = _s55.load_checkpoint(DLA_GPT, cfg)
        model.train()
        state = model.make_state(device)
        train_rng = random.Random(950000 + seed)
        stage_rows = []
        for t_idx, (start, pre_ppl_slice) in enumerate(chosen):
            text = pool[start * (args.train_chars + args.val_chars) : (start + 1) * (args.train_chars + args.val_chars)]
            train_ids = [stoi[c] for c in text[: args.train_chars] if c in stoi]
            val_ids = [stoi[c] for c in text[args.train_chars :] if c in stoi]
            # fixed eval batches for this stage
            rng = random.Random(960000 + seed * 100 + t_idx)
            eb = []
            for _ in range(args.eval_batches):
                n = len(val_ids) - args.block - 1
                ix = [rng.randrange(n) for _ in range(args.eval_batch)]
                x = torch.tensor([[val_ids[i + j] for j in range(args.block)] for i in ix], dtype=torch.long)
                y = torch.tensor([[val_ids[i + j + 1] for j in range(args.block)] for i in ix], dtype=torch.long)
                eb.append((x, y))
            pre_ppl = math.exp(sum((lambda _l: _l.item())(_m[1]) for _m in []) ) if False else None
            # actual pre measured on full state
            model.eval(); model.set_dla_state(state)
            losses_eval=[]
            for x,y in eb:
                _,l=model(x,y); losses_eval.append(l.item())
            model.train()
            pre = math.exp(sum(losses_eval)/len(losses_eval))

            losses = []
            curve = []
            for step in range(1, args.max_steps + 1):
                n = len(train_ids) - args.block - 1
                ix = [train_rng.randrange(n) for _ in range(args.batch)]
                x = torch.tensor([[train_ids[i + j] for j in range(args.block)] for i in ix], dtype=torch.long)
                y = torch.tensor([[train_ids[i + j + 1] for j in range(args.block)] for i in ix], dtype=torch.long)
                info = model.dla_step(x, y, state)
                losses.append(info["loss"])
                if step % args.eval_every == 0 or step == args.max_steps:
                    model.eval(); model.set_dla_state(state)
                    ev=[]
                    for xv,yv in eb:
                        _,lv=model(xv,yv); ev.append(lv.item())
                    model.train()
                    ppl=math.exp(sum(ev)/len(ev))
                    curve.append({"step":step,"ppl":ppl,"gain":(pre-ppl)/pre})
            model.dla_sleep(state)
            # slopes
            first = [losses[i] for i in range(min(10, len(losses)))]
            raw_slope = float(np.polyfit(range(len(first)), first, 1)[0])
            norm = [first[0] / first[0] if i==0 else first[i]/first[0] for i in range(len(first))] if first else []
            norm_slope = float(np.polyfit(range(len(norm)), norm, 1)[0]) if len(norm)>=2 else 0.0
            cs = _s55.curve_stats(curve, args.max_steps)
            stage_rows.append({
                "task": t_idx+1, "start": start, "pre": pre,
                "raw_slope10": raw_slope, "norm_slope10": norm_slope,
                "loss_first": first, "LE": cs["LE"], "T80": cs["T80"], "curve": curve,
            })
            print(f"  seed {seed} task{t_idx+1}: pre={pre:.2f} raw_slope={raw_slope:+.4f} norm_slope={norm_slope:+.4f} LE={cs['LE']:.3f}", flush=True)
        results[f"seed{seed}"] = stage_rows
        with open(os.path.join(args.out, "seeds", f"seed{seed}.json"), "w") as f:
            json.dump({"seed": seed, "chosen": chosen, "rows": stage_rows}, f, indent=2)

    with open(os.path.join(args.out, "stage6.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}/stage6.json", flush=True)


if __name__ == "__main__":
    main()
