"""Stage 5 - progressive curriculum and Lambda_t (learning efficiency across development).

Curriculum (difficulty validated by birth-model PPL on linghang1, 2026-09-07):
    stage 0  wiki general   (pre PPL ~ 24.8, in-distribution warm-up)
    stage 1  SFT Q&A        (pre PPL ~ 34.6, format shift)
    stage 2  wiki science   (pre PPL ~ 36.9, technical vocabulary shift)

Protocol per seed:
    * fixed eval batches per domain
    * each stage: adapt up to ``max_steps``, eval every ``eval_every``
    * target: relative adaptation gain >= ``target_gain`` on the current domain
    * Lambda_t = 1 / steps_to_target   (0 if never reached)
    * DLA sleeps between stages (consolidation); P and cognitive state persist
      -> the individual *develops* through the curriculum
    * baseline: best AdamW lr from Stage 4 formal (1e-4)

Question: does Lambda_t rise over the curriculum for the developmental individual,
while the static baseline stays flat / declines?

Run:
    ~/llm-lab/venv/bin/python experiments/stage5_progressive_curriculum.py \
        --seeds 0,1,2,3,4 --max-steps 80 --out results/stage5
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
            if sum(len(x) for x in picked) >= 2_000_000:
                break
    return "\n\n".join(picked)


_SCIENCE_TEXT = None


def get_science_slice(seed, train_chars, val_chars):
    global _SCIENCE_TEXT
    if _SCIENCE_TEXT is None:
        _SCIENCE_TEXT = build_science_text()
    start = seed * (train_chars + val_chars)
    return _SCIENCE_TEXT[start : start + train_chars + val_chars]


def build_curriculum(seed, args, stoi):
    enc = lambda s: [stoi[c] for c in s if c in stoi]
    wiki_start = args.wiki_start + seed * 1_500_000
    wiki_all = read_chars(WIKI, wiki_start, args.train_chars + args.val_chars)
    sft_start = seed * (args.train_chars + args.val_chars)
    sft_all = read_chars(SFT, sft_start, args.train_chars + args.val_chars)
    sci_all = get_science_slice(seed, args.train_chars, args.val_chars)
    domains = {
        "wiki": enc(wiki_all[args.train_chars : args.train_chars + args.val_chars]),
        "sft": enc(sft_all[args.train_chars : args.train_chars + args.val_chars]),
        "science": enc(sci_all[args.train_chars : args.train_chars + args.val_chars]),
    }
    phases = [
        ("wiki", enc(wiki_all[: args.train_chars]), "wiki"),
        ("sft", enc(sft_all[: args.train_chars]), "sft"),
        ("science", enc(sci_all[: args.train_chars]), "science"),
    ]
    return domains, phases


def get_batch(ids, block_size, batch_size, rng, device):
    n = len(ids) - block_size - 1
    ix = [rng.randrange(n) for _ in range(batch_size)]
    x = torch.stack([torch.tensor(ids[i : i + block_size], dtype=torch.long, device=device) for i in ix])
    y = torch.stack([torch.tensor(ids[i + 1 : i + 1 + block_size], dtype=torch.long, device=device) for i in ix])
    return x, y


def make_eval_batches(domains, args, device, seed):
    rng = random.Random(seed * 7919 + 13)
    out = {}
    for dom, ids in domains.items():
        batches = []
        for _ in range(args.eval_batches):
            batches.append(get_batch(ids, args.block, args.eval_batch, rng, device))
        out[dom] = batches
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
    model.load_state_dict(ckpt["model"], strict=True)
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


# --------------------------------------------------------------------- runners
def run_baseline(model, domains, phases, args, rng, device, eb, lr):
    model.train()
    opt = torch.optim.AdamW(unique_parameters(model), lr=lr, weight_decay=0.01)
    out = {"stage_pre": [], "curve": []}
    for phase_name, train_ids, val_dom in phases:
        pre = eval_ppl(model, eb[val_dom])
        out["stage_pre"].append(pre)
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
        out["curve"].append(curve)
    return out


def run_dla(model, domains, phases, args, rng, device, eb, dla_cfg):
    model.train()
    state = model.make_state(device)
    out = {"stage_pre": [], "curve": [], "curve_slow": [], "plasticity": []}
    for phase_name, train_ids, val_dom in phases:
        pre_full = eval_ppl(model, eb[val_dom], state=state)
        pre_slow = eval_ppl(model, eb[val_dom], state=None)
        out["stage_pre"].append(pre_full)
        curve, curve_slow = [], []
        for step in range(1, args.max_steps + 1):
            x, y = get_batch(train_ids, args.block, args.batch, rng, device)
            info = model.dla_step(x, y, state)
            if step % args.eval_every == 0 or step == args.max_steps:
                ppl_full = eval_ppl(model, eb[val_dom], state=state)
                ppl_slow = eval_ppl(model, eb[val_dom], state=None)
                curve.append({"step": step, "ppl": ppl_full, "gain": (pre_full - ppl_full) / pre_full})
                curve_slow.append({"step": step, "ppl": ppl_slow, "gain": (pre_slow - ppl_slow) / pre_slow})
        out["curve"].append(curve)
        out["curve_slow"].append(curve_slow)
        out["plasticity"].append(model.plasticity_by_layer(state))
        model.dla_sleep(state)
    return out


def lambda_from_curve(curve, target_gain):
    for point in curve:
        if point["gain"] >= target_gain:
            return 1.0 / point["step"]
    return 0.0


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=80)
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
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-dla", action="store_true")
    ap.add_argument("--out", type=str, default="results/stage5")
    args = ap.parse_args()
    args.seeds = [int(x) for x in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    device = "cpu"
    stoi = pickle.load(open(META, "rb"))["stoi"]
    DLA_GPT = make_dla_gpt_class(GPT)

    base_runs, dla_runs = [], []
    curriculum_order = {}
    for seed in args.seeds:
        domains, phases = build_curriculum(seed, args, stoi)
        eb = make_eval_batches(domains, args, device, seed)
        # difficulty = birth-model PPL on the domain; order stages easy -> hard
        probe = load_checkpoint(GPT)
        pre_all = {dom: eval_ppl(probe, eb[dom]) for dom in domains}
        order = sorted(domains, key=lambda d: pre_all[d])
        phases = sorted(phases, key=lambda ph: order.index(ph[2]))
        curriculum_order[seed] = order
        print(f"seed {seed} curriculum (easy->hard): {order} "
              f"pre_ppl={[round(pre_all[d],2) for d in order]}", flush=True)
        rng = random.Random(300000 + seed)

        if not args.skip_baseline:
            print(f"\n===== seed {seed} / AdamW lr={args.lr} =====", flush=True)
            t0 = time.time()
            model = load_checkpoint(GPT)
            rec = run_baseline(model, domains, phases, args, rng, device, eb, args.lr)
            base_runs.append(rec)
            lams = [lambda_from_curve(c, args.target_gain) for c in rec["curve"]]
            print(f"  stage_pre={[round(p,2) for p in rec['stage_pre']]} Lambda={[round(l,4) for l in lams]} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if not args.skip_dla:
            print(f"===== seed {seed} / DLA eta={args.eta_fast} =====", flush=True)
            t0 = time.time()
            dla_cfg = TransformerDLAConfig(eta_fast=args.eta_fast)
            model = load_checkpoint(DLA_GPT, dla_cfg)
            rec = run_dla(model, domains, phases, args, rng, device, eb, dla_cfg)
            dla_runs.append(rec)
            lams = [lambda_from_curve(c, args.target_gain) for c in rec["curve"]]
            print(f"  stage_pre={[round(p,2) for p in rec['stage_pre']]} Lambda_full={[round(l,4) for l in lams]} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    xs = [0, 1, 2]
    stage_names = ["wiki(general)", "sft(Q&A)", "wiki(science)"]
    summary = {"target_gain": args.target_gain, "curriculum_order": curriculum_order}
    if base_runs:
        lam = [[lambda_from_curve(c, args.target_gain) for c in r["curve"]] for r in base_runs]
        lam_mean = [sum(l[i] for l in lam) / len(lam) for i in xs]
        summary["baseline"] = {
            "lambda_per_seed": lam, "lambda_mean": lam_mean,
            "lambda_slope": slope(xs, lam_mean),
            "stage_pre_mean": [sum(r["stage_pre"][i] for r in base_runs) / len(base_runs) for i in xs],
        }
    if dla_runs:
        lam = [[lambda_from_curve(c, args.target_gain) for c in r["curve"]] for r in dla_runs]
        lam_slow = [[lambda_from_curve(c, args.target_gain) for c in r["curve_slow"]] for r in dla_runs]
        lam_mean = [sum(l[i] for l in lam) / len(lam) for i in xs]
        lam_slow_mean = [sum(l[i] for l in lam_slow) / len(lam_slow) for i in xs]
        summary["dla"] = {
            "lambda_per_seed": lam, "lambda_mean": lam_mean, "lambda_slope": slope(xs, lam_mean),
            "lambda_slow_mean": lam_slow_mean, "lambda_slow_slope": slope(xs, lam_slow_mean),
            "stage_pre_mean": [sum(r["stage_pre"][i] for r in dla_runs) / len(dla_runs) for i in xs],
        }

    print("\n===== Stage 5 summary =====", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    series = {}
    if base_runs:
        series["AdamW"] = summary["baseline"]["lambda_mean"]
    if dla_runs:
        series["DLA fast"] = summary["dla"]["lambda_mean"]
        series["DLA slow"] = summary["dla"]["lambda_slow_mean"]
    plot_trajectories(
        series,
        os.path.join(args.out, "lambda_curriculum.png"),
        xlabel="curriculum stage", ylabel=f"Lambda_t (target gain {args.target_gain:.0%})",
        title="Stage 5: learning efficiency across the progressive curriculum",
    )
    with open(os.path.join(args.out, "stage5.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary, "baseline_runs": base_runs, "dla_runs": dla_runs},
                  f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}/stage5.json + lambda_curriculum.png", flush=True)


if __name__ == "__main__":
    main()
