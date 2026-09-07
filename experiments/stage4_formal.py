"""Stage 4 FORMAL - DLA mechanism on nanoGPT, with honest hyperparameter tuning.

Protocol
--------
1. Hyperparameter sweep on two dedicated sweep lifetimes (not used later):
   * baseline: AdamW lr in {1e-4, 3e-4, 6e-4, 1e-3}
   * DLA     : eta_fast in {6e-4, 1.2e-3, 2.4e-3}
   selection criterion: score = gain_B - 2 * forgetting_A
   (adapt to B well while preserving A; DLA uses its slow-memory forgetting)
2. Formal run: 5 seeds. Each seed gets its own wiki/SFT slices and fixed eval
   batches. Lifetime: A(wiki) -> B(sft) -> A relearn.
3. Output: per-seed + mean metrics, complete fast/slow curves, relearning curve,
   plasticity trajectories; PNG plots + JSON.

Run:
    ~/llm-lab/venv/bin/python experiments/stage4_formal.py --steps 100 \
        --sweep-steps 60 --seeds 0,1,2,3,4 --out results/stage4_formal
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


def build_lifetime(seed, args, stoi):
    """Each seed lives on a different slice of the two domains."""
    wiki_start = args.wiki_start + seed * 1_500_000
    sft_start = seed * 400_000
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    wiki_train = encode(read_chars(WIKI, wiki_start, args.train_chars))
    wiki_val = encode(read_chars(WIKI, wiki_start + args.train_chars, args.val_chars))
    sft_train = encode(read_chars(SFT, sft_start, args.train_chars))
    sft_val = encode(read_chars(SFT, sft_start + args.train_chars, args.val_chars))
    domains = {"wiki": wiki_val, "sft": sft_val}
    phases = {
        "wiki": (wiki_train, "wiki"),
        "sft": (sft_train, "sft"),
        "wiki_relearn": (wiki_train, "wiki"),
    }
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
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (list(missing)[:3], list(unexpected)[:3])
    model.to("cpu").eval()
    return model, ckpt["model_args"]


# ------------------------------------------------------------------ evaluation
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
def run_baseline(model, domains, phases, steps, args, rng, device, eval_batches, lr):
    model.train()
    opt = torch.optim.AdamW(unique_parameters(model), lr=lr, weight_decay=0.01)
    log = {"pre": {}, "curve": {}, "matrix": []}
    for dom in domains:
        log["pre"][dom] = eval_ppl(model, eval_batches[dom])
    for phase_name, (train_ids, val_dom) in phases.items():
        log["curve"][phase_name] = []
        for step in range(1, steps + 1):
            x, y = get_batch(train_ids, args.block, args.batch, rng, device)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unique_parameters(model), 1.0)
            opt.step()
            if step % args.eval_every == 0 or step == steps:
                log["curve"][phase_name].append({"step": step, "ppl": eval_ppl(model, eval_batches[val_dom])})
        log["matrix"].append({dom: eval_ppl(model, eval_batches[dom]) for dom in domains})
    return log


def run_dla(model, domains, phases, steps, args, rng, device, eval_batches, dla_cfg):
    model.train()
    state = model.make_state(device)
    log = {"pre": {}, "curve": {}, "matrix": [], "matrix_slow": [], "plasticity": {}}
    for dom in domains:
        log["pre"][dom] = eval_ppl(model, eval_batches[dom], state=state)
    for phase_name, (train_ids, val_dom) in phases.items():
        log["curve"][phase_name] = []
        for step in range(1, steps + 1):
            x, y = get_batch(train_ids, args.block, args.batch, rng, device)
            info = model.dla_step(x, y, state)
            if step % args.eval_every == 0 or step == steps:
                log["curve"][phase_name].append(
                    {"step": step,
                     "ppl_full": eval_ppl(model, eval_batches[val_dom], state=state),
                     "ppl_slow": eval_ppl(model, eval_batches[val_dom], state=None),
                     "loss": info["loss"], "progress": info["progress"], "success": info["success"]}
                )
        log["matrix"].append({dom: eval_ppl(model, eval_batches[dom], state=state) for dom in domains})
        log["matrix_slow"].append({dom: eval_ppl(model, eval_batches[dom], state=None) for dom in domains})
        log["plasticity"][phase_name] = model.plasticity_by_layer(state)
        model.dla_sleep(state)
    log["plasticity"]["final"] = model.plasticity_by_layer(state)
    return log


def metrics_from(log, dla=False):
    pre = log["pre"]
    m = log["matrix"]
    if dla:
        ms = log["matrix_slow"]
        out = {
            "A_pre": pre["wiki"], "B_pre": pre["sft"],
            "A_postA": m[0]["wiki"], "B_postB": m[1]["sft"],
            "A_afterB": m[1]["wiki"], "A_afterRel": m[2]["wiki"],
            "A_slow_postA": ms[0]["wiki"], "B_slow_postB": ms[1]["sft"],
            "A_slow_afterB": ms[1]["wiki"], "A_slow_afterRel": ms[2]["wiki"],
            "gain_B_full": (pre["sft"] - m[1]["sft"]) / pre["sft"],
            "gain_B_slow": (pre["sft"] - ms[1]["sft"]) / pre["sft"],
            "forget_A_full": (m[1]["wiki"] - m[0]["wiki"]) / m[0]["wiki"],
            "forget_A_slow": (ms[1]["wiki"] - ms[0]["wiki"]) / ms[0]["wiki"],
            "relearn_A_full": (m[1]["wiki"] - m[2]["wiki"]) / m[1]["wiki"],
            "relearn_A_slow": (ms[1]["wiki"] - ms[2]["wiki"]) / ms[1]["wiki"],
        }
    else:
        out = {
            "A_pre": pre["wiki"], "B_pre": pre["sft"],
            "A_postA": m[0]["wiki"], "B_postB": m[1]["sft"],
            "A_afterB": m[1]["wiki"], "A_afterRel": m[2]["wiki"],
            "gain_B": (pre["sft"] - m[1]["sft"]) / pre["sft"],
            "forget_A": (m[1]["wiki"] - m[0]["wiki"]) / m[0]["wiki"],
            "relearn_A": (m[1]["wiki"] - m[2]["wiki"]) / m[1]["wiki"],
        }
    return out


def score_metrics(met, dla=False):
    if dla:
        return met["gain_B_full"] - 2.0 * max(0.0, met["forget_A_slow"])
    return met["gain_B"] - 2.0 * max(0.0, met["forget_A"])


# ----------------------------------------------------------------------- sweep
def sweep(args, device):
    stoi = pickle.load(open(META, "rb"))["stoi"]
    DLA_GPT = make_dla_gpt_class(GPT)
    print("\n===== AdamW lr sweep (2 sweep lifetimes) =====", flush=True)
    best_lr, best_lr_score = None, -1e9
    for lr in args.lr_candidates:
        scores = []
        for sw_seed in (6, 7):
            domains, phases = build_lifetime(sw_seed, args, stoi)
            eb = make_eval_batches(domains, args, device, sw_seed)
            rng = random.Random(100000 + sw_seed)
            model, _ = load_checkpoint(GPT)
            log = run_baseline(model, domains, phases, args.sweep_steps, args, rng, device, eb, lr)
            scores.append(score_metrics(metrics_from(log)))
        mean_score = sum(scores) / len(scores)
        print(f"  lr {lr}: score {mean_score:+.4f}  ({scores[0]:+.3f}, {scores[1]:+.3f})", flush=True)
        if mean_score > best_lr_score:
            best_lr, best_lr_score = lr, mean_score
    print(f"  -> best baseline lr = {best_lr} (score {best_lr_score:+.4f})", flush=True)

    print("\n===== DLA eta_fast sweep =====", flush=True)
    best_eta, best_eta_score = None, -1e9
    for eta in args.eta_candidates:
        scores = []
        for sw_seed in (6, 7):
            domains, phases = build_lifetime(sw_seed, args, stoi)
            eb = make_eval_batches(domains, args, device, sw_seed)
            rng = random.Random(100000 + sw_seed)
            dla_cfg = TransformerDLAConfig(eta_fast=eta, consolidate_beta=args.consolidate_beta,
                                            consolidate_fast_direct=args.consolidate_fast_direct, alpha_q=args.alpha_q)
            model, _ = load_checkpoint(DLA_GPT, dla_cfg)
            log = run_dla(model, domains, phases, args.sweep_steps, args, rng, device, eb, dla_cfg)
            scores.append(score_metrics(metrics_from(log, dla=True), dla=True))
        mean_score = sum(scores) / len(scores)
        print(f"  eta {eta}: score {mean_score:+.4f}  ({scores[0]:+.3f}, {scores[1]:+.3f})", flush=True)
        if mean_score > best_eta_score:
            best_eta, best_eta_score = eta, mean_score
    print(f"  -> best DLA eta_fast = {best_eta} (score {best_eta_score:+.4f})", flush=True)
    return best_lr, best_eta


# ---------------------------------------------------------------------- formal
def formal(args, device, best_lr, best_eta):
    stoi = pickle.load(open(META, "rb"))["stoi"]
    DLA_GPT = make_dla_gpt_class(GPT)
    base_logs, dla_logs = [], []
    base_mets, dla_mets = [], []
    for seed in args.seeds:
        domains, phases = build_lifetime(seed, args, stoi)
        eb = make_eval_batches(domains, args, device, seed)
        rng = random.Random(200000 + seed)

        print(f"\n===== formal seed {seed} / baseline lr={best_lr} =====", flush=True)
        t0 = time.time()
        model, _ = load_checkpoint(GPT)
        log = run_baseline(model, domains, phases, args.steps, args, rng, device, eb, best_lr)
        base_logs.append(log)
        met = metrics_from(log)
        base_mets.append(met)
        print(f"  baseline {time.time()-t0:.0f}s  " + _fmt(met), flush=True)

        print(f"===== formal seed {seed} / DLA eta={best_eta} =====", flush=True)
        t0 = time.time()
        dla_cfg = TransformerDLAConfig(eta_fast=best_eta, consolidate_beta=args.consolidate_beta,
                                        consolidate_fast_direct=args.consolidate_fast_direct, alpha_q=args.alpha_q)
        model, _ = load_checkpoint(DLA_GPT, dla_cfg)
        log = run_dla(model, domains, phases, args.steps, args, rng, device, eb, dla_cfg)
        dla_logs.append(log)
        met = metrics_from(log, dla=True)
        dla_mets.append(met)
        print(f"  DLA {time.time()-t0:.0f}s  " + _fmt(met), flush=True)

    return base_logs, base_mets, dla_logs, dla_mets


def _fmt(met):
    if "gain_B" in met:
        return f"gainB={met['gain_B']:+.3f} forgetA={met['forget_A']:+.3f} relearn={met['relearn_A']:+.3f}"
    return (f"gainB_full={met['gain_B_full']:+.3f} forgetA_slow={met['forget_A_slow']:+.3f} "
            f"relearnA_full={met['relearn_A_full']:+.3f}")


def mean_metrics(mets):
    keys = list(mets[0].keys())
    out = {}
    for k in keys:
        vals = [m[k] for m in mets]
        mu = sum(vals) / len(vals)
        out[k] = {"mean": mu, "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    return out


def curve_mean(logs, phase, key):
    """Average a phase curve across seeds, aligned by step index."""
    n = min(len(log["curve"][phase]) for log in logs)
    steps = [logs[0]["curve"][phase][i]["step"] for i in range(n)]
    means = []
    for i in range(n):
        vals = [log["curve"][phase][i][key] for log in logs]
        means.append(sum(vals) / len(vals))
    return steps, means


def make_plots(args, base_logs, dla_logs):
    os.makedirs(args.out, exist_ok=True)
    # A across lifetime (fast/slow + baseline)
    series = {}
    series["baseline_A"] = [base_logs[i]["matrix"][j]["wiki"] for j in range(3) for i in [0]]
    # average across seeds at each phase
    for name, key in [("baseline", None)]:
        pass
    xs = ["A_pre", "postA", "afterB", "afterRel"]
    def mean_across_seeds(get):
        return [sum(get(i, j) for i in range(len(base_logs))) / len(base_logs) for j in range(len(xs))]
    # baseline
    base_series = []
    for j, lab in enumerate(xs):
        vals = []
        for log in base_logs:
            if j == 0:
                vals.append(log["pre"]["wiki"])
            else:
                vals.append(log["matrix"][j - 1]["wiki"])
        base_series.append(sum(vals) / len(vals))
    dla_full, dla_slow = [], []
    for j, lab in enumerate(xs):
        fv, sv = [], []
        for log in dla_logs:
            if j == 0:
                fv.append(log["pre"]["wiki"]); sv.append(log["pre"]["wiki"])
            else:
                fv.append(log["matrix"][j - 1]["wiki"]); sv.append(log["matrix_slow"][j - 1]["wiki"])
        dla_full.append(sum(fv) / len(fv)); dla_slow.append(sum(sv) / len(sv))
    plot_trajectories(
        {"AdamW": base_series, "DLA fast": dla_full, "DLA slow": dla_slow},
        os.path.join(args.out, "A_across_lifetime.png"),
        xlabel="lifetime phase", ylabel="wiki PPL",
        title="Stage 4 formal: A domain across the lifetime",
    )
    # relearning curve for A
    rsteps, rbase = curve_mean(base_logs, "wiki_relearn", "ppl")
    _, rdla_full = curve_mean(dla_logs, "wiki_relearn", "ppl_full")
    _, rdla_slow = curve_mean(dla_logs, "wiki_relearn", "ppl_slow")
    plot_trajectories(
        {"AdamW": rbase, "DLA fast": rdla_full, "DLA slow": rdla_slow},
        os.path.join(args.out, "A_relearning_curve.png"),
        xlabel="relearning step", ylabel="wiki PPL",
        title="Stage 4 formal: relearning curve (A revisited)",
    )
    # B adaptation curve
    bsteps, bbase = curve_mean(base_logs, "sft", "ppl")
    _, bdla_full = curve_mean(dla_logs, "sft", "ppl_full")
    _, bdla_slow = curve_mean(dla_logs, "sft", "ppl_slow")
    plot_trajectories(
        {"AdamW": bbase, "DLA fast": bdla_full, "DLA slow": bdla_slow},
        os.path.join(args.out, "B_adaptation_curve.png"),
        xlabel="adaptation step", ylabel="SFT PPL",
        title="Stage 4 formal: adaptation curve (B domain)",
    )


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--sweep-steps", type=int, default=60)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--wiki-start", type=int, default=40_000_000)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--lr-candidates", type=str, default="1e-4,3e-4,6e-4,1e-3")
    ap.add_argument("--eta-candidates", type=str, default="6e-4,1.2e-3,2.4e-3")
    ap.add_argument("--consolidate-beta", type=float, default=1.0)
    ap.add_argument("--consolidate-fast-direct", type=float, default=0.15)
    ap.add_argument("--alpha-q", type=float, default=0.3)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--best-lr", type=float, default=None)
    ap.add_argument("--best-eta", type=float, default=None)
    ap.add_argument("--out", type=str, default="results/stage4_formal")
    args = ap.parse_args()
    args.lr_candidates = [float(x) for x in args.lr_candidates.split(",")]
    args.eta_candidates = [float(x) for x in args.eta_candidates.split(",")]
    args.seeds = [int(x) for x in args.seeds.split(",")]

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    device = "cpu"

    if args.skip_sweep:
        best_lr, best_eta = args.best_lr, args.best_eta
        assert best_lr and best_eta
    else:
        best_lr, best_eta = sweep(args, device)

    base_logs, base_mets, dla_logs, dla_mets = formal(args, device, best_lr, best_eta)
    make_plots(args, base_logs, dla_logs)

    summary = {
        "best_lr": best_lr, "best_eta": best_eta,
        "baseline": {"per_seed": base_mets, "mean": mean_metrics(base_mets)},
        "dla": {"per_seed": dla_mets, "mean": mean_metrics(dla_mets)},
    }
    print("\n===== Stage 4 FORMAL summary =====", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    with open(os.path.join(args.out, "formal.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary, "baseline_logs": base_logs, "dla_logs": dla_logs},
                  f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}/formal.json + plots", flush=True)


if __name__ == "__main__":
    main()
