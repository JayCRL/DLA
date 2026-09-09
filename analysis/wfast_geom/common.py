"""Shared helpers for the W_fast geometry analysis (P1/P2 on Stage 5.5e bodies).

Everything here is derived from EXISTING artifacts only:
  * saved EH/HE bodies (results/stage55c|stage55e/bodies/seed*_*_meta.pt)
  * per-seed P0 D-probe JSONs (results/stage55e/seeds/seed*.json)

No model training is performed; D-probe *replays* (replay.py) reproduce the
exact stored protocol deterministically so per-step gradient trajectories can
be summarised without re-training anything new.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
NANO_DIR = os.path.expanduser("~/llm-lab/nanoGPT")
sys.path.insert(0, NANO_DIR)

from model import GPT, GPTConfig  # noqa: E402  (registers unpickle modules)
from dla.transformer_dla import DLAState, TransformerDLAConfig, make_dla_gpt_class  # noqa: E402

EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import stage55_learning_rule_development as s55  # noqa: E402
import stage55c_causality as s55c  # noqa: E402

META = s55.META

# ----------------------------------------------------------------- groups
def group_of(key: str) -> str:
    if key.startswith("transformer.wte") or key.startswith("transformer.wpe") or key.startswith("lm_head"):
        return "emb"
    if ".attn." in key:
        return "attn"
    if ".mlp." in key:
        return "mlp"
    raise KeyError(f"no group for {key}")


GROUPS = ("emb", "attn", "mlp")


def group_keys(store_keys):
    out = {g: [] for g in GROUPS}
    for k in sorted(store_keys):
        out[group_of(k)].append(k)
    return out


def canonical_keys(store_keys):
    return sorted(store_keys)


# ----------------------------------------------------------------- bodies
def body_path(seed: int, order: str) -> Path:
    c1 = ROOT / "results" / "stage55c" / f"seed{seed}" / "bodies" / f"seed{seed}_{order}_meta.pt"
    c2 = ROOT / "results" / "stage55e" / "bodies" / f"seed{seed}_{order}_meta.pt"
    if c1.exists():
        return c1
    return c2


def load_body_ckpt(seed: int, order: str):
    """torch.load a body checkpoint (store + model + phi), no model construction."""
    p = body_path(seed, order)
    if not p.exists():
        raise FileNotFoundError(p)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    return ckpt


def store_of(ckpt) -> dict:
    return ckpt["store"]


def all_seeds():
    return list(range(12))


# ----------------------------------------------------------------- vector ops (matrix-blockwise, no giant copies)
def norm2_blockwise(store, keys) -> float:
    return sum(float(store[k]["w_fast"].pow(2).sum()) for k in keys)


def dot_blockwise(store_a, store_b, keys, field="w_fast") -> float:
    return sum(float((store_a[k][field] * store_b[k][field]).sum()) for k in keys)


def cosine(store_a, store_b, keys, field="w_fast") -> float:
    na = norm2_blockwise(store_a, keys)
    nb = norm2_blockwise(store_b, keys)
    if na <= 0 or nb <= 0:
        return 0.0
    return dot_blockwise(store_a, store_b, keys, field) / math.sqrt(na * nb)


# ----------------------------------------------------------------- args (identical to Stage 5.5e P0 protocol)
class Args:
    eta_fast = 6e-4
    d_steps = 40
    block = 128
    batch = 32
    eval_every = 2
    eval_batch = 16
    eval_batches = 4
    wiki_start = 40_000_000
    train_chars = 300_000
    val_chars = 40_000


def build_D(seed: int):
    """Rebuild the identical unseen-domain D data + eval batches for a seed."""
    args = Args()
    stoi = _load_stoi()
    _domains, _phases, d_train, d_val = s55.build_curriculum(seed, args, stoi)
    return args, d_train


def _load_stoi():
    import pickle
    return pickle.load(open(META, "rb"))["stoi"]


def arm_tags():
    return ["EH/EH", "HE/HE", "EH_body+HE_fast", "HE_body+EH_fast"]


def slug(tag: str) -> str:
    return tag.replace("/", "_").replace("+", "plus")
