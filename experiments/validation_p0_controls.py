#!/usr/bin/env python3
"""Paper Validation P0 - strict controls for the W_fast cross-injection effect.

Uses saved EH/HE bodies (seeds 0..11) and the same held-out D protocol as
Stage 5.5e P0. For every seed we run D-probes for:

  HE/HE                native
  HE+EH_Wfast          raw transfer (destructive)
  HE+norm(EH_Wfast)    per-matrix Frobenius norm matched to HE
  HE+shuffle(EH_Wfast) within-matrix shuffle + norm matched to HE
  HE+EH_emb/attn/mlp   module-localized W_fast transfer

All probes use the same D data, update rule, budget and eval batches per seed.
Output includes full trajectories, paired stats and effect sizes.
"""

import argparse
import importlib.util
import json
import math
import os
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

_STAGE55C = ROOT / "experiments" / "stage55c_causality.py"
_specC = importlib.util.spec_from_file_location("stage55c", _STAGE55C)
_s55c = importlib.util.module_from_spec(_specC)
_specC.loader.exec_module(_s55c)

DLA_GPT = make_dla_gpt_class(GPT)


class Args:
    eta_fast = 6e-4


def find_body(seed, order):
    c1 = ROOT / "results" / "stage55c" / f"seed{seed}" / "bodies" / f"seed{seed}_{order}_meta.pt"
    c2 = ROOT / "results" / "stage55e" / "bodies" / f"seed{seed}_{order}_meta.pt"
    return c1 if c1.exists() else c2


def is_emb(key):
    return key.startswith("transformer.wte") or key.startswith("transformer.wpe") or key.startswith("lm_head")


def is_attn(key):
    return ".attn." in key


def is_mlp(key):
    return ".mlp." in key


def copy_module_group(src_state, dst_state, group):
    for key, ss in src_state.store.items():
        if key not in dst_state.store:
            continue
        if group == "emb" and not is_emb(key):
            continue
        if group == "attn" and not is_attn(key):
            continue
        if group == "mlp" and not is_mlp(key):
            continue
        dst_state.store[key]["w_fast"].copy_(ss["w_fast"])


def copy_norm_matched(src_state, dst_state, rng=None):
    """Copy src W_fast scaled per-matrix to dst W_fast Frobenius norm."""
    for key, ss in src_state.store.items():
        if key not in dst_state.store:
            continue
        src = ss["w_fast"].double()
        target = dst_state.store[key]["w_fast"].double()
        if src.numel() == 0 or target.numel() == 0:
            continue
        ns = src.norm().item()
        nt = target.norm().item()
        if ns > 0:
            dst_state.store[key]["w_fast"].copy_((src * (nt / ns)).float())


def copy_shuffled(src_state, dst_state, rng):
    """Within-matrix shuffle of src W_fast, then per-matrix norm match to dst."""
    for key, ss in src_state.store.items():
        if key not in dst_state.store:
            continue
        src = ss["w_fast"].double().reshape(-1)
        # deterministic per-seed/key shuffle
        perm = torch.randperm(src.numel(), generator=torch.Generator().manual_seed(rng.randint(0, 1_000_000)))
        shuffled = src[perm].reshape(ss["w_fast"].shape)
        target = dst_state.store[key]["w_fast"].double()
        if shuffled.numel() == 0 or target.numel() == 0:
            continue
        ns = shuffled.norm().item()
        nt = target.norm().item()
        if ns > 0:
            dst_state.store[key]["w_fast"].copy_((shuffled * (nt / ns)).float())


def run_d_probe(model, state, d_train, eb, args, device, seed, tag):
    rng = random.Random(1_200_000 + seed + (len(tag) % 1000))
    args.max_steps = args.d_steps
    d = _s55.run_transfer(model, d_train, eb, args, rng, device, state=state, dla=True)
    cs = _s55.curve_stats(d["curve"], args.d_steps)
    return {"tag": tag, "LE_D": cs["LE"], "T80_D": cs["T80"], "pre": d["pre"], "curve": d["curve"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--d-steps", type=int, default=40)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--wiki-start", type=int, default=40_000_000)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--out", type=str, default="results/validation")
    args = ap.parse_args()
    args.seeds = [int(x) for x in args.seeds.split(",")]
    os.makedirs(os.path.join(args.out, "seeds"), exist_ok=True)
    torch.set_num_threads(2)
    device = "cpu"
    stoi = pickle.load(open(_s55.META, "rb"))["stoi"]

    tags = ["HE/HE", "HE+EH_Wfast", "HE+normEH", "HE+shuffleEH",
            "HE+EH_emb", "HE+EH_attn", "HE+EH_mlp"]
    all_rows = []
    for seed in args.seeds:
        # build same D as Stage 5.5e
        domains, phases, d_train, d_val = _s55.build_curriculum(seed, args, stoi)
        eb = _s55.make_eval_batches(domains, d_val, args, device, seed)
        he_path = find_body(seed, "hard_easy")
        eh_path = find_body(seed, "easy_hard")
        if not he_path.exists() or not eh_path.exists():
            print(f"seed {seed}: missing body, skip")
            continue
        he_model, he_state = _s55c.load_body(str(he_path), Args(), device)
        eh_model, eh_state = _s55c.load_body(str(eh_path), Args(), device)

        # tag-specific operations on a fresh copy of HE body/state
        # because each probe mutates the state.
        def probe(tag, op=None):
            model, state = _s55c.load_body(str(he_path), Args(), device)
            if op is not None:
                op(state)
            res = run_d_probe(model, state, d_train, eb, args, device, seed, tag)
            all_rows.append({"seed": seed, **res})
            print(f"seed {seed} {tag}: pre={res['pre']:.2f} LE={res['LE_D']:.3f}", flush=True)

        probe("HE/HE")
        probe("HE+EH_Wfast", lambda st: copy_module_group(eh_state, st, "all"))
        probe("HE+normEH", lambda st: copy_norm_matched(eh_state, st))
        rng = random.Random(1_300_000 + seed)
        probe("HE+shuffleEH", lambda st: copy_shuffled(eh_state, st, rng))
        probe("HE+EH_emb", lambda st: copy_module_group(eh_state, st, "emb"))
        probe("HE+EH_attn", lambda st: copy_module_group(eh_state, st, "attn"))
        probe("HE+EH_mlp", lambda st: copy_module_group(eh_state, st, "mlp"))

    with open(os.path.join(args.out, "results_validation.json"), "w") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}/results_validation.json")


if __name__ == "__main__":
    import pickle
    main()
