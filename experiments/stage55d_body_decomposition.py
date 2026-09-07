"""Stage 5.5d - Body decomposition.

Uses the saved 5.5c bodies (seed{0..2}, easy_hard/hard_easy meta) and asks:
WHICH component of the Body carries the History effect?

Components tested by cross-swapping between EH and HE bodies:
    state components : W_slow (all), W_fast, P, Q
    W_slow groups    : embeddings (wte/wpe/lm_head), attention, MLP
    native references: EH/EH, HE/HE

Each combination is probed on the SAME unseen D domain with the same budget.

Run per seed in parallel:
    ~/llm-lab/venv/bin/python experiments/stage55d_body_decomposition.py \
        --seeds 0 --d-steps 30 --out results/stage55d
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_STAGE55C = ROOT / "experiments" / "stage55c_causality.py"
_spec = importlib.util.spec_from_file_location("stage55c", _STAGE55C)
_s55c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s55c)

_STAGE55 = ROOT / "experiments" / "stage55_learning_rule_development.py"
_spec55 = importlib.util.spec_from_file_location("stage55", _STAGE55)
_s55 = importlib.util.module_from_spec(_spec55)
_spec55.loader.exec_module(_s55)

import pickle  # noqa: E402

# group classifiers by wrapped module path
def _is_emb(name):
    return name.startswith("transformer.wte") or name.startswith("transformer.wpe") or name.startswith("lm_head")

def _is_attn(name):
    return ".attn." in name

def _is_mlp(name):
    return ".mlp." in name


def copy_slow_group(src_model, dst_model, group):
    """Copy selected slow weight matrices from src_model into dst_model."""
    with torch.no_grad():
        for name, mod in src_model.wrapped:
            if group == "emb" and not _is_emb(name):
                continue
            if group == "attn" and not _is_attn(name):
                continue
            if group == "mlp" and not _is_mlp(name):
                continue
            if group == "all_slow":
                pass
            else:
                continue
            dst_name_mod = None
            for dn, dm in dst_model.wrapped:
                if dn == name:
                    dst_name_mod = dm
                    break
            if dst_name_mod is not None and dst_name_mod.weight.shape == mod.weight.shape:
                dst_name_mod.weight.copy_(mod.weight)


def copy_state_component(src_state, dst_state, component):
    """Copy W_fast/P/Q from src_state into dst_state."""
    for key, ss in src_state.store.items():
        if key not in dst_state.store:
            continue
        if component in ("fast", "all_state"):
            dst_state.store[key]["w_fast"].copy_(ss["w_fast"])
        if component in ("p", "all_state"):
            dst_state.store[key]["p"].copy_(ss["p"])
        if component in ("q", "all_state"):
            dst_state.store[key]["q"].copy_(ss["q"])


def run_probe(model, state, d_train, eb, args, device, seed, tag):
    rng = random.Random(700000 + seed + (len(tag) % 1000))
    args.max_steps = args.d_steps
    d = _s55.run_transfer(model, d_train, eb, args, rng, device, state=state, dla=True)
    cs = _s55.curve_stats(d["curve"], args.d_steps)
    return {"tag": tag, "LE_D": cs["LE"], "T80_D": cs["T80"], "best_gain_D": cs["best_gain"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--d-steps", type=int, default=30)
    ap.add_argument("--target-gain", type=float, default=0.04)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--eta-fast", type=float, default=6e-4)
    ap.add_argument("--wiki-start", type=int, default=40_000_000)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--out", type=str, default="results/stage55d")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(3)
    device = "cpu"

    combos = [
        ("native", "EH", "EH", "all"),
        ("native", "HE", "HE", "all"),
        ("swap", "EH_body+HE_slow", "EH", "HE", "all_slow"),
        ("swap", "EH_body+HE_fast", "EH", "HE", "fast"),
        ("swap", "EH_body+HE_P", "EH", "HE", "p"),
        ("swap", "EH_body+HE_Q", "EH", "HE", "q"),
        ("swap", "EH_body+HE_attn_slow", "EH", "HE", "attn"),
        ("swap", "EH_body+HE_mlp_slow", "EH", "HE", "mlp"),
        ("swap", "EH_body+HE_emb_slow", "EH", "HE", "emb"),
        ("swap", "HE_body+EH_slow", "HE", "EH", "all_slow"),
        ("swap", "HE_body+EH_fast", "HE", "EH", "fast"),
        ("swap", "HE_body+EH_P", "HE", "EH", "p"),
        ("swap", "HE_body+EH_Q", "HE", "EH", "q"),
        ("swap", "HE_body+EH_attn_slow", "HE", "EH", "attn"),
        ("swap", "HE_body+EH_mlp_slow", "HE", "EH", "mlp"),
        ("swap", "HE_body+EH_emb_slow", "HE", "EH", "emb"),
    ]

    all_out = {}
    stoi = pickle.load(open(_s55.META, "rb"))["stoi"]
    for seed in args.seeds:
        body_dir = os.path.join(_s55c.ROOT, "results", "stage55c", f"seed{seed}", "bodies")
        paths = {o: os.path.join(body_dir, f"seed{seed}_{o}_meta.pt") for o in ("easy_hard", "hard_easy")}
        domains, phases, d_train, d_val = _s55.build_curriculum(seed, args, stoi)
        eb = _s55.make_eval_batches(domains, d_val, args, device, seed)
        order_short = {"easy_hard": "EH", "hard_easy": "HE"}
        models = {}
        states = {}
        for o, path in paths.items():
            m, s = _s55c.load_body(path, args, device)
            models[order_short[o]], states[order_short[o]] = m, s

        print(f"\n===== seed {seed}: body decomposition =====", flush=True)
        for combo in combos:
            if combo[0] == "native":
                body_o = combo[1]
                model, state = _s55c.load_body(paths["easy_hard" if body_o == "EH" else "hard_easy"], args, device)
                tag = f"{body_o}/{body_o}"
            else:
                # combo: swap src component into base body
                _, tag, base_o, src_o, component = combo
                base_path = paths["easy_hard" if base_o == "EH" else "hard_easy"]
                src_path = paths["easy_hard" if src_o == "EH" else "hard_easy"]
                model, state = _s55c.load_body(base_path, args, device)
                src_model, src_state = _s55c.load_body(src_path, args, device)
                if component == "all_slow" or component in ("attn", "mlp", "emb"):
                    copy_slow_group(src_model, model, component)
                else:
                    copy_state_component(src_state, state, component)
                del src_model, src_state
            res = run_probe(model, state, d_train, eb, args, device, seed, tag)
            all_out.setdefault(f"seed{seed}", {})[tag] = res
            print(f"  {tag}: LE_D={res['LE_D']:.3f} T80={res['T80_D']}", flush=True)
            del model, state

    with open(os.path.join(args.out, "decomposition.json"), "w") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}/decomposition.json", flush=True)


if __name__ == "__main__":
    main()
