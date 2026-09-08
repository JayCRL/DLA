#!/usr/bin/env python3
"""Second-setting small replication: Shakespeare character GPT.

Re-runs the core W_fast causal check on a different backbone/corpus:
  * backbone: ~7M Shakespeare char GPT (vocab 65)
  * corpus: data/shakespeare_char/input.txt
  * protocol: 3-task EH/HE history (DLA, no meta) -> unseen D -> W_fast
    cross-injection (HE/HE vs HE+EH_fast)

Small scale: 2 seeds by default.
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

from dla.transformer_dla import TransformerDLAConfig, make_dla_gpt_class  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

DLA_GPT = make_dla_gpt_class(GPT)
CKPT = os.path.join(NANO_DIR, "out-shakespeare-char", "ckpt.pt")
RAW = os.path.join(NANO_DIR, "data", "shakespeare_char", "input.txt")
META = os.path.join(NANO_DIR, "data", "shakespeare_char", "meta.pkl")


class Args:
    eta_fast = 6e-4


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    config = GPTConfig(**ck["model_args"])
    model = DLA_GPT(config, TransformerDLAConfig(eta_fast=6e-4))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [k for k in missing if not k.startswith("tempos.")]
    assert not missing and not unexpected, (missing, unexpected)
    model.to("cpu").eval()
    return model


def get_slices(seed):
    text = open(RAW, encoding="utf-8").read()
    stoi = pickle.load(open(META, "rb"))["stoi"]
    span = 100_000  # enough slices for two seeds in ~1.1M chars
    chunks = []
    for i in range(5):
        s = text[(seed * 5 + i) * span : (seed * 5 + i + 1) * span]
        enc = [stoi[c] for c in s if c in stoi]
        chunks.append(enc)
    return chunks


def make_batch(ids, block=128, batch=16, rng=None, device="cpu"):
    n = len(ids) - block - 1
    ix = [rng.randrange(n) for _ in range(batch)]
    x = torch.tensor([[ids[i + j] for j in range(block)] for i in ix], dtype=torch.long)
    y = torch.tensor([[ids[i + j + 1] for j in range(block)] for i in ix], dtype=torch.long)
    return x, y


def eval_ppl(model, ids, block=128, batches=4, seed=0, device="cpu"):
    rng = random.Random(seed)
    model.eval()
    losses = []
    for _ in range(batches):
        n = len(ids) - block - 1
        ix = [rng.randrange(n) for _ in range(8)]
        x = torch.tensor([[ids[i + j] for j in range(block)] for i in ix], dtype=torch.long)
        y = torch.tensor([[ids[i + j + 1] for j in range(block)] for i in ix], dtype=torch.long)
        with torch.no_grad():
            _, l = model(x, y)
        losses.append(l.item())
    model.train()
    return math.exp(sum(losses) / len(losses))


def train_history(seed, order, chunks):
    """Train a DLA body on 3 chunks in the given order (EH or HE)."""
    model = load_model()
    model.train()
    state = model.make_state("cpu")
    train_rng = random.Random(1_000_000 + seed * 10 + (0 if order == "easy_hard" else 1))
    for train_ids in chunks:
        for _ in range(40):
            x, y = make_batch(train_ids, rng=train_rng)
            model.dla_step(x, y, state)
        model.dla_sleep(state)
    return model, state


def run_d_probe(model, state, d_ids, seed):
    rng = random.Random(2_000_000 + seed)
    pre = eval_ppl(model, d_ids, seed=seed)
    curve = []
    for step in range(1, 41):
        x, y = make_batch(d_ids, rng=rng)
        model.dla_step(x, y, state)
        if step % 2 == 0 or step == 40:
            ppl = eval_ppl(model, d_ids, seed=seed)
            curve.append({"step": step, "ppl": ppl, "gain": (pre - ppl) / pre})
    best = max(p["gain"] for p in curve)
    gmax = max(best, 0.0)
    le = max(0.0, curve[-1]["gain"] / gmax) if gmax > 0 else 0.0
    t80 = 40
    if gmax > 0:
        for p in curve:
            if p["gain"] >= 0.8 * gmax:
                t80 = p["step"]
                break
    return {"pre": pre, "curve": curve, "LE_D": le, "T80_D": t80}


def copy_fast(src_state, dst_state):
    for k, ss in src_state.store.items():
        if k in dst_state.store:
            dst_state.store[k]["w_fast"].copy_(ss["w_fast"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--out", type=str, default="results/validation_second")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    os.makedirs(os.path.join(args.out, "seeds"), exist_ok=True)
    torch.set_num_threads(2)
    all_rows = []
    for seed in seeds:
        chunks = get_slices(seed)
        c0, c1, c2, d = chunks[0], chunks[1], chunks[2], chunks[3]
        # birth PPL ordering of curriculum chunks
        probe = load_model()
        ppls = [(c0, eval_ppl(probe, c0, seed=seed)), (c1, eval_ppl(probe, c1, seed=seed)), (c2, eval_ppl(probe, c2, seed=seed))]
        ppls.sort(key=lambda z: z[1])
        eh_chunks = [z[0] for z in ppls]
        he_chunks = list(reversed(eh_chunks))
        # histories
        eh_model, eh_state = train_history(seed, "easy_hard", eh_chunks)
        he_model, he_state = train_history(seed, "hard_easy", he_chunks)
        # probes
        res_native = run_d_probe(he_model, he_state, d, seed)
        # cross: HE body + EH fast (fresh deep-copied HE body)
        import copy
        cross_model = copy.deepcopy(he_model)
        cross_state = copy.deepcopy(he_state)
        copy_fast(eh_state, cross_state)
        res_cross = run_d_probe(cross_model, cross_state, d, seed)
        row = {"seed": seed, "HE_LE": res_native["LE_D"], "HE_gain40": res_native["curve"][-1]["gain"],
               "cross_LE": res_cross["LE_D"], "cross_gain40": res_cross["curve"][-1]["gain"],
               "curves": {"native": res_native["curve"], "cross": res_cross["curve"]}}
        all_rows.append(row)
        print(f"seed {seed}: HE LE={row['HE_LE']:.3f} gain40={row['HE_gain40']:+.4f} | "
              f"cross LE={row['cross_LE']:.3f} gain40={row['cross_gain40']:+.4f}", flush=True)
        with open(os.path.join(args.out, "seeds", f"seed{seed}.json"), "w") as f:
            json.dump(row, f, indent=2)
    with open(os.path.join(args.out, "second_setting.json"), "w") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}/second_setting.json")


if __name__ == "__main__":
    main()
