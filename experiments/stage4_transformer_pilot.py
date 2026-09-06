"""Stage 4 pilot - DLA mechanism on a real (small) Transformer.

Backbone : ~/llm-lab/nanoGPT/out-chinese/ckpt.pt
           6-layer / 8-head / 256-dim / vocab 7280 / block 256, char-level Chinese

Lifetime  : A (wikipedia) -> B (SFT-style Q&A) -> A again (relearning)

Arms:
    Baseline  standard AdamW fine-tuning of W_slow (learning rule outside model)
    DLA       per-parameter P-gated fast weights + Adam moments + sleep consolidation
              (learning rule inside the model)

Metrics (relative to the individual's own starting point):
    adaptation gain : (ppl_pre - ppl_post) / ppl_pre  on the current domain
    forgetting      : (ppl_A_after_B - ppl_A_after_A) / ppl_A_after_A
    relearning      : fraction of the A gain recovered after re-visiting A
    plasticity      : mean softplus(P) per parameter matrix across the lifetime

Run (pilot):
    ~/llm-lab/venv/bin/python experiments/stage4_transformer_pilot.py --steps 30
"""

from __future__ import annotations

import argparse
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

from dla.transformer_dla import TransformerDLAConfig, make_dla_gpt_class  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

CKPT = os.path.join(NANO_DIR, "out-chinese", "ckpt.pt")
META = os.path.join(NANO_DIR, "data", "zh_char", "meta.pkl")
WIKI = os.path.expanduser("~/llm-lab/corpus/raw_full.txt")
SFT = os.path.join(NANO_DIR, "data", "zh_sft", "raw_sft.txt")


# ---------------------------------------------------------------------- data
def load_meta():
    with open(META, "rb") as f:
        meta = pickle.load(f)
    return meta["stoi"]


def read_chars(path: str, start_chars: int, budget: int) -> str:
    out = []
    total = 0
    with open(path, encoding="utf-8") as f:
        if start_chars:
            f.read(start_chars)
        while total < budget:
            chunk = f.read(min(1 << 20, budget - total))
            if not chunk:
                break
            out.append(chunk)
            total += len(chunk)
    return "".join(out)


def encode(text: str, stoi: dict) -> list:
    return [stoi[c] for c in text if c in stoi]


def get_batch(ids: list, block_size: int, batch_size: int, rng: random.Random, device):
    n = len(ids) - block_size - 1
    ix = [rng.randrange(n) for _ in range(batch_size)]
    x = torch.stack([torch.tensor(ids[i : i + block_size], dtype=torch.long, device=device) for i in ix])
    y = torch.stack([torch.tensor(ids[i + 1 : i + 1 + block_size], dtype=torch.long, device=device) for i in ix])
    return x, y


def unique_parameters(model):
    seen = set()
    out = []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


# ----------------------------------------------------------------- model io
def load_checkpoint(GPTClass, *class_args):
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["model_args"])
    model = GPTClass(config, *class_args)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        print(f"WARNING missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    model.to("cpu")
    model.eval()
    return model, ckpt["model_args"]


# ----------------------------------------------------------------- evaluation
def make_eval_batches(domains, args, device, seed=999):
    """Fixed eval batches: every measurement uses exactly the same tokens."""
    rng = random.Random(seed)
    out = {}
    for dom, ids in domains.items():
        batches = []
        for _ in range(args.eval_batches):
            x, y = get_batch(ids, args.block, args.eval_batch, rng, device)
            batches.append((x, y))
        out[dom] = batches
    return out


@torch.no_grad()
def eval_ppl(model, batches, state=None):
    """Char-level PPL over fixed batches. state=None -> slow-only for DLA models."""
    was_training = model.training
    model.eval()
    if hasattr(model, "set_dla_state"):
        model.set_dla_state(state)
    losses = []
    for x, y in batches:
        logits, loss = model(x, y)
        losses.append(loss.item())
    model.train(was_training)
    return math.exp(sum(losses) / len(losses))


# -------------------------------------------------------------------- runners
def run_baseline(model, domains, phases, args, rng, device, eval_batches):
    """Standard continual fine-tuning: AdamW directly on W_slow."""
    model.train()
    opt = torch.optim.AdamW(unique_parameters(model), lr=args.lr_baseline, weight_decay=0.01)
    log = {"pre": {}, "curve": {}, "matrix": []}
    for dom in domains:
        log["pre"][dom] = eval_ppl(model, eval_batches[dom])

    for phase_name, (train_ids, val_dom, steps) in phases.items():
        log["curve"][phase_name] = []
        for step in range(1, steps + 1):
            t0 = time.time()
            x, y = get_batch(train_ids, args.block, args.batch, rng, device)
            opt.zero_grad(set_to_none=True)
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unique_parameters(model), 1.0)
            opt.step()
            if step % args.eval_every == 0 or step == steps:
                ppl = eval_ppl(model, eval_batches[val_dom])
                log["curve"][phase_name].append({"step": step, "ppl": ppl, "sec": time.time() - t0})
                print(f"  baseline {phase_name} step {step}/{steps}: ppl {ppl:.3f}", flush=True)
        row = {dom: eval_ppl(model, eval_batches[dom]) for dom in domains}
        log["matrix"].append(row)
        print(f"  baseline after {phase_name}: " + "  ".join(f"{d}={v:.3f}" for d, v in row.items()), flush=True)
    return log


def run_dla(model, domains, phases, args, rng, device, dla_cfg, eval_batches):
    """Developmental lifetime: wake (dla_step) + sleep (dla_sleep)."""
    model.train()
    state = model.make_state(device)
    log = {"pre": {}, "curve": {}, "matrix": [], "matrix_slow": [], "plasticity": {}}
    for dom in domains:
        log["pre"][dom] = eval_ppl(model, eval_batches[dom], state=state)

    for phase_name, (train_ids, val_dom, steps) in phases.items():
        log["curve"][phase_name] = []
        for step in range(1, steps + 1):
            t0 = time.time()
            x, y = get_batch(train_ids, args.block, args.batch, rng, device)
            info = model.dla_step(x, y, state)
            if step % args.eval_every == 0 or step == steps:
                ppl_full = eval_ppl(model, eval_batches[val_dom], state=state)
                ppl_slow = eval_ppl(model, eval_batches[val_dom], state=None)
                log["curve"][phase_name].append(
                    {"step": step, "ppl_full": ppl_full, "ppl_slow": ppl_slow, "sec": time.time() - t0,
                     "loss": info["loss"], "progress": info["progress"], "life_step": info["life_step"]}
                )
                print(
                    f"  DLA {phase_name} step {step}/{steps}: ppl_full {ppl_full:.3f} "
                    f"ppl_slow {ppl_slow:.3f} (progress {info['progress']:+.3f})",
                    flush=True,
                )
        row_full = {dom: eval_ppl(model, eval_batches[dom], state=state) for dom in domains}
        row_slow = {dom: eval_ppl(model, eval_batches[dom], state=None) for dom in domains}
        log["matrix"].append(row_full)
        log["matrix_slow"].append(row_slow)
        log["plasticity"][phase_name] = model.plasticity_by_layer(state)
        print(f"  DLA after {phase_name} full: " + "  ".join(f"{d}={v:.3f}" for d, v in row_full.items()), flush=True)
        print(f"  DLA after {phase_name} slow: " + "  ".join(f"{d}={v:.3f}" for d, v in row_slow.items()), flush=True)
        model.dla_sleep(state)
    log["plasticity"]["final"] = model.plasticity_by_layer(state)
    return log


def metrics(log, domains, dla=False):
    A, B = domains
    pre = log["pre"]
    if dla:
        m = log["matrix"]
        ms = log["matrix_slow"]
        out = {
            "A_pre": pre[A], "B_pre": pre[B],
            "A_postA": m[0][A], "B_postB": m[1][B], "A_afterB": m[1][A], "A_afterRel": m[2][A],
            "A_slow_postA": ms[0][A], "B_slow_postB": ms[1][B], "A_slow_afterB": ms[1][A], "A_slow_afterRel": ms[2][A],
            "gain_B_full": (pre[B] - m[1][B]) / pre[B],
            "gain_B_slow": (pre[B] - ms[1][B]) / pre[B],
            "forget_A_full": (m[1][A] - m[0][A]) / m[0][A],
            "forget_A_slow": (ms[1][A] - ms[0][A]) / ms[0][A],
            "relearn_A_full": (m[1][A] - m[2][A]) / m[1][A],
            "relearn_A_slow": (ms[1][A] - ms[2][A]) / ms[1][A],
        }
    else:
        m = log["matrix"]
        out = {
            "A_pre": pre[A], "B_pre": pre[B],
            "A_postA": m[0][A], "B_postB": m[1][B], "A_afterB": m[1][A], "A_afterRel": m[2][A],
            "gain_B": (pre[B] - m[1][B]) / pre[B],
            "forget_A": (m[1][A] - m[0][A]) / m[0][A],
            "relearn_A": (m[1][A] - m[2][A]) / m[1][A],
        }
    return out


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30, help="adaptation steps per phase")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=6)
    ap.add_argument("--wiki-start", type=int, default=40_000_000)
    ap.add_argument("--train-chars", type=int, default=250_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--lr-baseline", type=float, default=6e-4)
    ap.add_argument("--eta-fast", type=float, default=6e-4)
    ap.add_argument("--consolidate-beta", type=float, default=0.3)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-dla", action="store_true")
    ap.add_argument("--out", type=str, default="results/stage4")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    device = "cpu"
    rng = random.Random(2026)

    print("loading vocab + domain texts ...", flush=True)
    stoi = load_meta()
    wiki_train = encode(read_chars(WIKI, args.wiki_start, args.train_chars), stoi)
    wiki_val = encode(read_chars(WIKI, args.wiki_start + args.train_chars, args.val_chars), stoi)
    sft_train = encode(read_chars(SFT, 0, args.train_chars), stoi)
    sft_val = encode(read_chars(SFT, args.train_chars, args.val_chars), stoi)
    print(f"wiki train {len(wiki_train)} / val {len(wiki_val)} tokens", flush=True)
    print(f"sft  train {len(sft_train)} / val {len(sft_val)} tokens", flush=True)

    domains = {"wiki": wiki_val, "sft": sft_val}
    phases = {
        "wiki": (wiki_train, "wiki", args.steps),
        "sft": (sft_train, "sft", args.steps),
        "wiki_relearn": (wiki_train, "wiki", args.steps),
    }
    eval_batches = make_eval_batches(domains, args, device)

    DLA_GPT = make_dla_gpt_class(GPT)
    results = {}
    if not args.skip_baseline:
        print("\n===== baseline: AdamW fine-tune =====", flush=True)
        model, _ = load_checkpoint(GPT)
        t0 = time.time()
        results["baseline"] = run_baseline(model, domains, phases, args, rng, device, eval_batches)
        print(f"baseline done in {time.time()-t0:.1f}s", flush=True)

    if not args.skip_dla:
        print("\n===== DLA-Transformer =====", flush=True)
        dla_cfg = TransformerDLAConfig(eta_fast=args.eta_fast, consolidate_beta=args.consolidate_beta)
        model, _ = load_checkpoint(DLA_GPT, dla_cfg)
        t0 = time.time()
        results["dla"] = run_dla(model, domains, phases, args, rng, device, dla_cfg, eval_batches)
        print(f"DLA done in {time.time()-t0:.1f}s", flush=True)

    domain_names = ["wiki", "sft"]
    summary = {arm: metrics(log, domain_names, dla=(arm == "dla")) for arm, log in results.items()}

    print("\n===== Stage 4 pilot summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(os.path.join(args.out, "pilot.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}/pilot.json")


if __name__ == "__main__":
    main()
