"""Stage 5.5 - does the LEARNING RULE itself develop over a lifetime?

Causal test designed after the Stage 5 negative result.

Arms (same curriculum, per seed):
    AdamW        W dynamic, no P, phi = external optimizer (fixed rule)
    DLA-static   W dynamic, P dynamic, phi(t) = phi_0   (learning rule frozen)
    DLA-meta     W dynamic, P dynamic, phi(t+1) = phi(t) - beta * grad_phi L_future

phi is ``TransformerTempoParams``: eta_fast, eta_plast, fast_decay,
stability, alpha_q, consolidation rates.  In DLA-meta, before entering each
NEW stage, phi takes one meta-gradient step on a short differentiable unroll of
batches from that FUTURE stage (future learning, not current loss).

Every arm is finally probed on a held-out UNSEEN domain D (science slice never
used in the curriculum): LE_D measures transfer of the learning system itself.

Metrics reported:
    LE_t, T80_t          (difficulty-normalised, from Stage 5 analysis)
    Delta_phi_t          (||phi_{t+1}-phi_t|| for DLA-meta; 0 for DLA-static)
    LE_D                 (learning efficiency on unseen D)
    reversed curriculum  (phi and LE_D for hard->easy vs easy->hard histories)

Run:
    ~/llm-lab/venv/bin/python experiments/stage55_learning_rule_development.py \
        --seeds 0,1,2 --max-steps 60 --out results/stage55
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NANO_DIR = os.path.expanduser("~/llm-lab/nanoGPT")
sys.path.insert(0, NANO_DIR)

from dla.transformer_dla import TransformerDLAConfig, make_dla_gpt_class  # noqa: E402
from dla.plotting import plot_trajectories  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

CKPT = os.path.join(NANO_DIR, "out-chinese", "ckpt.pt")
META = os.path.join(NANO_DIR, "data", "zh_char", "meta.pkl")
WIKI = os.path.expanduser("~/llm-lab/corpus/raw_full.txt")
SFT = os.path.join(NANO_DIR, "data", "zh_sft", "raw_sft.txt")
WIKI_JSON = os.path.expanduser("~/llm-lab/datasets/wikipedia-cn-20230720-filtered.json")

SCI_KEYWORDS = ["科学", "数学", "物理", "化学", "生物", "定理", "公式", "分子", "原子",
                "方程", "函数", "细胞", "基因", "量子", "力学", "实验", "理论"]


# ---------------------------------------------------------------------- data
def read_chars(path, start_chars, budget):
    out, total = [], 0
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


def build_science_text():
    docs = json.load(open(WIKI_JSON, encoding="utf-8"))
    picked = []
    for d in docs:
        t = (d.get("completion") or "").strip()
        if t and any(k in t for k in SCI_KEYWORDS):
            picked.append(t)
            if sum(len(x) for x in picked) >= 12_000_000:
                break
    return "\n\n".join(picked)


_SCIENCE_TEXT = None


def science_slice(start, n_chars):
    global _SCIENCE_TEXT
    if _SCIENCE_TEXT is None:
        _SCIENCE_TEXT = build_science_text()
    return _SCIENCE_TEXT[start : start + n_chars]


def build_curriculum(seed, args, stoi):
    enc = lambda s: [stoi[c] for c in s if c in stoi]
    span = args.train_chars + args.val_chars
    wiki = read_chars(WIKI, args.wiki_start + seed * 1_500_000, span)
    sft = read_chars(SFT, seed * span, span)
    sci = science_slice(seed * span, span)
    domains = {
        "wiki": enc(wiki[args.train_chars:]),
        "sft": enc(sft[args.train_chars:]),
        "science": enc(sci[args.train_chars:]),
    }
    phases = [
        ("wiki", enc(wiki[: args.train_chars]), "wiki"),
        ("sft", enc(sft[: args.train_chars]), "sft"),
        ("science", enc(sci[: args.train_chars]), "science"),
    ]
    # held-out transfer domain D: unseen science slice
    d_all = science_slice(1_800_000, span)
    d_train = enc(d_all[: args.train_chars])
    d_val = enc(d_all[args.train_chars:])
    return domains, phases, d_train, d_val


def get_batch(ids, block_size, batch_size, rng, device):
    n = len(ids) - block_size - 1
    ix = [rng.randrange(n) for _ in range(batch_size)]
    x = torch.stack([torch.tensor(ids[i : i + block_size], dtype=torch.long, device=device) for i in ix])
    y = torch.stack([torch.tensor(ids[i + 1 : i + 1 + block_size], dtype=torch.long, device=device) for i in ix])
    return x, y


def make_eval_batches(domains, d_val, args, device, seed):
    rng = random.Random(seed * 7919 + 13)
    out = {}
    for dom, ids in domains.items():
        out[dom] = [get_batch(ids, args.block, args.eval_batch, rng, device) for _ in range(args.eval_batches)]
    out["D"] = [get_batch(d_val, args.block, args.eval_batch, rng, device) for _ in range(args.eval_batches)]
    return out


def unique_parameters(model):
    seen, out = set(), []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def load_checkpoint(GPTClass, *class_args):
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["model_args"])
    model = GPTClass(config, *class_args)
    # DLA variants add tempo parameters (phi) that are NOT in the pretrained
    # checkpoint; they are initialised from the DNA config instead.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    missing = [k for k in missing if not k.startswith("tempos.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch missing={missing[:3]} unexpected={list(unexpected)[:3]}")
    model.to("cpu").eval()
    return model


@torch.no_grad()
def eval_ppl(model, batches, state=None):
    was_training = model.training
    model.eval()
    if hasattr(model, "set_dla_state"):
        model.set_dla_state(state)
    losses = []
    for x, y in batches:
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train(was_training)
    return math.exp(sum(losses) / len(losses))


# ------------------------------------------------------------- metric helpers
def curve_stats(curve, max_steps):
    if not curve:
        return {"LE": 0.0, "T80": max_steps, "best_gain": 0.0, "final_gain": 0.0}
    best = max(p["gain"] for p in curve)
    g_max = max(best, 0.0)
    le = max(0.0, curve[-1]["gain"] / g_max) if g_max > 0 else 0.0
    t80 = max_steps
    if g_max > 0:
        for p in curve:
            if p["gain"] >= 0.8 * g_max:
                t80 = p["step"]
                break
    return {"LE": le, "T80": t80, "best_gain": best, "final_gain": curve[-1]["gain"]}


def raw_lambda(curve, target_gain):
    for p in curve:
        if p["gain"] >= target_gain:
            return 1.0 / p["step"]
    return 0.0


def tempo_norm(snap):
    return math.sqrt(sum(v * v for v in snap.values()))


def tempo_delta_norm(before, after):
    return math.sqrt(sum((after[k] - before[k]) ** 2 for k in before))


# -------------------------------------------------------------------- runners
def run_baseline(model, phases, args, rng, device, eb):
    model.train()
    opt = torch.optim.AdamW(unique_parameters(model), lr=args.lr, weight_decay=0.01)
    rec = {"stage_pre": [], "curve": []}
    for name, train_ids, val_dom in phases:
        pre = eval_ppl(model, eb[val_dom])
        rec["stage_pre"].append(pre)
        curve = []
        for step in range(1, args.max_steps + 1):
            x, y = get_batch(train_ids, args.block, args.batch, rng, device)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unique_parameters(model), 1.0)
            opt.step()
            if step % args.eval_every == 0 or step == args.max_steps:
                ppl = eval_ppl(model, eb[val_dom])
                curve.append({"step": step, "ppl": ppl, "gain": (pre - ppl) / pre})
        rec["curve"].append(curve)
    return rec


def run_dla(model, phases, args, rng, device, eb, meta=False):
    model.train()
    state = model.make_state(device)
    meta_opt = torch.optim.Adam(model.tempos.parameters(), lr=args.meta_lr) if meta else None
    rec = {"stage_pre": [], "curve": [], "curve_slow": [], "plasticity": [], "phi_trace": [], "meta_loss": []}
    for t_idx, (name, train_ids, val_dom) in enumerate(phases):
        if meta and t_idx > 0:
            before = model.tempos.snapshot()
            batches = [get_batch(train_ids, args.block, args.meta_batch, rng, device) for _ in range(args.meta_unroll_len)]
            mloss = model.meta_update_phi(state, batches, meta_opt)
            after = model.tempos.snapshot()
            rec["phi_trace"].append({"stage": t_idx, "before": before, "after": after,
                                     "norm": tempo_delta_norm(before, after)})
            rec["meta_loss"].append(mloss)
        pre_full = eval_ppl(model, eb[val_dom], state=state)
        pre_slow = eval_ppl(model, eb[val_dom], state=None)
        rec["stage_pre"].append(pre_full)
        curve, curve_slow = [], []
        for step in range(1, args.max_steps + 1):
            x, y = get_batch(train_ids, args.block, args.batch, rng, device)
            info = model.dla_step(x, y, state)
            if step % args.eval_every == 0 or step == args.max_steps:
                ppl_full = eval_ppl(model, eb[val_dom], state=state)
                ppl_slow = eval_ppl(model, eb[val_dom], state=None)
                curve.append({"step": step, "ppl": ppl_full, "gain": (pre_full - ppl_full) / pre_full})
                curve_slow.append({"step": step, "ppl": ppl_slow, "gain": (pre_slow - ppl_slow) / pre_slow})
        rec["curve"].append(curve)
        rec["curve_slow"].append(curve_slow)
        rec["plasticity"].append(model.plasticity_by_layer(state))
        model.dla_sleep(state)
    return rec, state


def run_transfer(model, d_train, eb, args, rng, device, state=None, dla=False):
    """Probe on unseen domain D after the curriculum."""
    model.train()
    if not dla:
        opt = torch.optim.AdamW(unique_parameters(model), lr=args.lr, weight_decay=0.01)
    pre = eval_ppl(model, eb["D"], state=state)
    curve = []
    for step in range(1, args.max_steps + 1):
        x, y = get_batch(d_train, args.block, args.batch, rng, device)
        if dla:
            model.dla_step(x, y, state)
        else:
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unique_parameters(model), 1.0)
            opt.step()
        if step % args.eval_every == 0 or step == args.max_steps:
            ppl = eval_ppl(model, eb["D"], state=state)
            curve.append({"step": step, "ppl": ppl, "gain": (pre - ppl) / pre})
    return {"pre": pre, "curve": curve}


def stage_stats(curves, max_steps, target_gain):
    return {"LE": [curve_stats(c, max_steps)["LE"] for c in curves],
            "T80": [curve_stats(c, max_steps)["T80"] for c in curves],
            "raw_lambda": [raw_lambda(c, target_gain) for c in curves]}


def mean_list(list_of_lists):
    n = len(list_of_lists[0])
    return [sum(row[i] for row in list_of_lists) / len(list_of_lists) for i in range(n)]


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--target-gain", type=float, default=0.04)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--wiki-start", type=int, default=40_000_000)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eta-fast", type=float, default=6e-4)
    ap.add_argument("--meta-lr", type=float, default=0.1)
    ap.add_argument("--meta-batch", type=int, default=4)
    ap.add_argument("--meta-unroll-len", type=int, default=2)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--reversed-seeds", type=str, default="0,1")
    ap.add_argument("--skip-adamw", action="store_true")
    ap.add_argument("--skip-static", action="store_true")
    ap.add_argument("--skip-meta", action="store_true")
    ap.add_argument("--out", type=str, default="results/stage55")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.reversed_seeds = [int(s) for s in args.reversed_seeds.split(",")] if args.reversed_seeds else []
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    device = "cpu"
    stoi = pickle.load(open(META, "rb"))["stoi"]
    DLA_GPT = make_dla_gpt_class(GPT)

    results = {k: [] for k in ["adamw", "dla_static", "dla_meta"]}
    d_results = {k: [] for k in ["adamw", "dla_static", "dla_meta"]}
    order_log = {}

    for seed in args.seeds:
        domains, phases, d_train, d_val = build_curriculum(seed, args, stoi)
        eb = make_eval_batches(domains, d_val, args, device, seed)
        probe = load_checkpoint(GPT)
        pre_all = {dom: eval_ppl(probe, eb[dom]) for dom in domains}
        order = sorted(domains, key=lambda d: pre_all[d])
        order_log[seed] = {"forward": order, "pre": {d: pre_all[d] for d in domains}}
        phases = sorted(phases, key=lambda ph: order.index(ph[2]))
        rng = random.Random(400000 + seed)

        if not args.skip_adamw:
            print(f"\n[seed {seed}] AdamW forward curriculum ...", flush=True)
            model = load_checkpoint(GPT)
            rec = run_baseline(model, phases, args, rng, device, eb)
            results["adamw"].append(rec)
            d = run_transfer(model, d_train, eb, args, rng, device, dla=False)
            d_results["adamw"].append(d)
            print(f"  AdamW D transfer LE={curve_stats(d['curve'], args.max_steps)['LE']:.3f}", flush=True)

        for arm, meta in (("dla_static", False), ("dla_meta", True)):
            if (arm == "dla_static" and args.skip_static) or (arm == "dla_meta" and args.skip_meta):
                continue
            print(f"[seed {seed}] {arm} forward curriculum ...", flush=True)
            cfg = TransformerDLAConfig(eta_fast=args.eta_fast)
            model = load_checkpoint(DLA_GPT, cfg)
            rec, state = run_dla(model, phases, args, rng, device, eb, meta=meta)
            results[arm].append(rec)
            d = run_transfer(model, d_train, eb, args, rng, device, state=state, dla=True)
            d_results[arm].append(d)
            print(f"  {arm} D transfer LE={curve_stats(d['curve'], args.max_steps)['LE']:.3f}", flush=True)

    # ------------------------------------------------------------------ summary
    summary = {"order": order_log, "target_gain": args.target_gain, "arms": {}}
    for arm, recs in results.items():
        if not recs:
            continue
        s = {}
        for key, curves in (("LE", [r["curve"] for r in recs]),):
            pass
        les = [[curve_stats(c, args.max_steps)["LE"] for c in r["curve"]] for r in recs]
        t80s = [[curve_stats(c, args.max_steps)["T80"] for c in r["curve"]] for r in recs]
        raw = [[raw_lambda(c, args.target_gain) for c in r["curve"]] for r in recs]
        s["LE_mean"] = mean_list(les)
        s["T80_mean"] = mean_list(t80s)
        s["raw_lambda_mean"] = mean_list(raw)
        s["LE_slope"] = slope([0, 1, 2], s["LE_mean"])
        if arm == "dla_meta" and all(r["phi_trace"] for r in recs):
            norms = [tr["norm"] for r in recs for tr in r["phi_trace"]]
            s["delta_phi_norms"] = norms
            s["delta_phi_mean"] = sum(norms) / len(norms) if norms else 0.0
        if d_results[arm]:
            dles = [curve_stats(d["curve"], args.max_steps)["LE"] for d in d_results[arm]]
            dt80 = [curve_stats(d["curve"], args.max_steps)["T80"] for d in d_results[arm]]
            s["D_LE_mean"] = sum(dles) / len(dles)
            s["D_T80_mean"] = sum(dt80) / len(dt80)
            s["D_LE_all"] = dles
        summary["arms"][arm] = s

    # ------------------------------------------------------- reversed curriculum
    rev_summary = {}
    if args.reversed_seeds and results.get("dla_meta"):
        rev_summary = {"seeds": args.reversed_seeds, "forward": {}, "reversed": {}}
        for seed in args.reversed_seeds:
            domains, phases, d_train, d_val = build_curriculum(seed, args, stoi)
            eb = make_eval_batches(domains, d_val, args, device, seed)
            probe = load_checkpoint(GPT)
            pre_all = {dom: eval_ppl(probe, eb[dom]) for dom in domains}
            order = sorted(domains, key=lambda d: pre_all[d])
            fwd = sorted(phases, key=lambda ph: order.index(ph[2]))
            rev = list(reversed(fwd))
            for label, ph in (("forward", fwd), ("reversed", rev)):
                for arm, meta in (("dla_static", False), ("dla_meta", True)):
                    rng = random.Random(410000 + seed)
                    cfg = TransformerDLAConfig(eta_fast=args.eta_fast)
                    model = load_checkpoint(DLA_GPT, cfg)
                    rec, state = run_dla(model, ph, args, rng, device, eb, meta=meta)
                    d = run_transfer(model, d_train, eb, args, rng, device, state=state, dla=True)
                    key = f"{label}_{arm}"
                    rev_summary.setdefault(key, []).append(
                        {"LE_D": curve_stats(d["curve"], args.max_steps)["LE"],
                         "T80_D": curve_stats(d["curve"], args.max_steps)["T80"],
                         "phi_norms": [tr["norm"] for tr in rec["phi_trace"]] if rec["phi_trace"] else []})
            print(f"[seed {seed}] reversed curriculum done", flush=True)

    print("\n===== Stage 5.5 summary =====", flush=True)
    out = {"args": vars(args), "summary": summary, "reversed": rev_summary,
           "results": results, "d_results": d_results}
    print(json.dumps({"summary": summary, "reversed": rev_summary}, indent=2, ensure_ascii=False), flush=True)
    with open(os.path.join(args.out, "stage55.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    series = {}
    for arm in ("adamw", "dla_static", "dla_meta"):
        if arm in summary["arms"]:
            series[arm] = summary["arms"][arm]["LE_mean"]
    plot_trajectories(series, os.path.join(args.out, "LE_development.png"),
                      xlabel="curriculum stage", ylabel="normalized LE",
                      title="Stage 5.5: learning efficiency across development")
    print(f"\nsaved -> {args.out}/stage55.json + LE_development.png", flush=True)


if __name__ == "__main__":
    main()
