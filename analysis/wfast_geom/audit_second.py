"""A5 - second-backbone (Shakespeare char GPT) core causal replication.

Port of experiments/validation_second_setting.py with (i) configurable slice span
so N>2 disjoint seeds fit the corpus, (ii) extra arm: HE + energy-matched shuffled
EH W_fast (allocation destroys structure), and (iii) per-seed paired output.

Corpus length limits N: slices used per seed = 5*span; disjoint seeds need
(seed+1)*5*span <= n_chars.

Run (one job per seed): python analysis/wfast_geom/audit_second.py --seed 0 --span 26000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "wfast_geom"))
NANO = os.path.expanduser("~/llm-lab/nanoGPT")
sys.path.insert(0, NANO)
sys.path.insert(0, str(ROOT / "experiments"))

from dla.transformer_dla import TransformerDLAConfig, make_dla_gpt_class  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

DLA_GPT = make_dla_gpt_class(GPT)
CKPT = os.path.join(NANO, "out-shakespeare-char", "ckpt.pt")
RAW = os.path.join(NANO, "data", "shakespeare_char", "input.txt")
META = os.path.join(NANO, "data", "shakespeare_char", "meta.pkl")


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    config = GPTConfig(**ck["model_args"])
    model = DLA_GPT(config, TransformerDLAConfig(eta_fast=6e-4))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [k for k in missing if not k.startswith("tempos.")]
    assert not missing and not unexpected, (missing, list(unexpected)[:3])
    model.to("cpu").eval()
    return model


def n_chars():
    return len(open(RAW, encoding="utf-8").read())


def get_slices(seed, span):
    text = open(RAW, encoding="utf-8").read()
    stoi = pickle.load(open(META, "rb"))["stoi"]
    chunks = []
    for i in range(4):
        s = text[(seed * 4 + i) * span:(seed * 4 + i + 1) * span]
        chunks.append([stoi[c] for c in s if c in stoi])
    return chunks  # 0..2 = curriculum chunks, 3 = unseen D


def make_batch(ids, block=128, batch=16, rng=None, device="cpu"):
    n = len(ids) - block - 1
    ix = [rng.randrange(n) for _ in range(batch)]
    x = torch.tensor([[ids[i + j] for j in range(block)] for i in ix], dtype=torch.long)
    y = torch.tensor([[ids[i + j + 1] for j in range(block)] for i in ix], dtype=torch.long)
    return x, y


def eval_ppl(model, ids, block=128, batches=4, seed=0):
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
    return {"pre": pre, "gain40": curve[-1]["gain"]}


def copy_fast(src_state, dst_state):
    for k, ss in src_state.store.items():
        if k in dst_state.store:
            dst_state.store[k]["w_fast"].copy_(ss["w_fast"])


def copy_shuf_fast(src_state, dst_state, seed):
    rng = random.Random(3_000_000 + seed)
    for k, ss in src_state.store.items():
        if k in dst_state.store:
            flat = ss["w_fast"].reshape(-1)
            perm = torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(rng.randint(0, 10 ** 9)))
            dst_state.store[k]["w_fast"].copy_(flat[perm].reshape(ss["w_fast"].shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--span", type=int, default=26000)
    ap.add_argument("--out", default="~/llm-lab/dla_audit_second")
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "3")))
    seed = args.seed
    chunks = get_slices(seed, args.span)
    c0, c1, c2, d = chunks
    probe = load_model()
    ppls = sorted([(c0, eval_ppl(probe, c0, seed=seed)), (c1, eval_ppl(probe, c1, seed=seed)),
                   (c2, eval_ppl(probe, c2, seed=seed))], key=lambda z: z[1])
    eh_chunks = [z[0] for z in ppls]
    he_chunks = list(reversed(eh_chunks))
    eh_model, eh_state = train_history(seed, "easy_hard", eh_chunks)
    he_model, he_state = train_history(seed, "hard_easy", he_chunks)

    native = run_d_probe(he_model, he_state, d, seed)
    cross = copy.deepcopy(he_model), copy.deepcopy(he_state)
    copy_fast(eh_state, cross[1])
    cross_res = run_d_probe(*cross, d, seed)
    shuf = copy.deepcopy(he_model), copy.deepcopy(he_state)
    copy_shuf_fast(eh_state, shuf[1], seed)
    shuf_res = run_d_probe(*shuf, d, seed)

    row = {"seed": seed, "span": args.span, "HE_gain40": native["gain40"],
           "cross_gain40": cross_res["gain40"], "shuf_gain40": shuf_res["gain40"],
           "HE_pre": native["pre"], "cross_pre": cross_res["pre"]}
    with open(os.path.join(out, f"second_seed{seed}.json"), "w") as f:
        json.dump(row, f, indent=1)
    print(f"seed {seed}: HE {row['HE_gain40']:+.4f} | HE+EH {row['cross_gain40']:+.4f} "
          f"| HE+shufEH {row['shuf_gain40']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
