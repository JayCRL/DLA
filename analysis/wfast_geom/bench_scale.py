"""Measure DLA step time and memory on a real backbone — run this FIRST on the cloud GPU.

Purpose
-------
The cost of the scale experiment is dominated by one unknown: how long a single
DLA wake step takes on the target GPU.  Guessing it from the Mac's MPS numbers
is wrong (different backend, different kernels, different memory hierarchy), and
guessing it from FLOPs is unreliable for this workload because the DLA rule does
many small elementwise ops on tensors shaped like the backbone weights.

So: measure it.  This script reports seconds/step, peak VRAM, and a projected
wall-clock + cost for a full multi-seed run, so the experiment can be sized
before committing hours of GPU rental.

Usage
-----
    python analysis/wfast_geom/bench_scale.py --model gpt2-large --device cuda
    python analysis/wfast_geom/bench_scale.py --model gpt2-xl --device cuda --state-dtype bf16

It runs a short warmup, then times N steps of ``dla_step`` plus the eval loop,
and prints a table.  No files are written except an optional --json report.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "wfast_geom"))
sys.path.insert(0, os.path.expanduser("~/llm-lab/nanoGPT"))

import audit_scale as S  # noqa: E402

# AutoDL RTX 4090 reference price (CNY/hour); override with --rate
DEFAULT_RATE = 1.88


def human(n: int) -> str:
    return f"{n/1e9:.2f}G"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-large", choices=list(S.MODELS))
    ap.add_argument("--state-dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10, help="timed wake steps")
    ap.add_argument("--evals", type=int, default=2, help="timed eval passes")
    ap.add_argument("--eval-batches", type=int, default=2)
    ap.add_argument("--history-steps", type=int, default=40)
    ap.add_argument("--probe-steps", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=4, help="seeds in the target run")
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE, help="CNY per GPU-hour")
    ap.add_argument("--corpus", default=os.path.expanduser("~/llm-lab/hf_gpt2/shakespeare_bpe.npy"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    dev = a.device
    print("=" * 72)
    print(f"Backbone benchmark: {a.model}  state_dtype={a.state_dtype}  device={dev}")
    print("=" * 72)

    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    t0 = time.time()
    model = S.build(a.model, a.state_dtype, dev)
    print(f"  build: {time.time()-t0:.1f}s  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    ids = S.get_corpus(a.corpus, model.config.vocab_size)
    print(f"  corpus: {len(ids):,} tokens")

    state = model.make_state(dev)
    model.set_dla_state(state)

    # ---- warmup (also materialises allocator pools) ----------------------
    rng = torch.Generator().manual_seed(0)
    for _ in range(a.warmup):
        x, y = S.make_batch(ids, a.block, a.batch, rng)
        model.dla_step(x.to(dev), y.to(dev), state)
    if dev == "cuda":
        torch.cuda.synchronize()

    # ---- time wake steps -------------------------------------------------
    step_times = []
    for _ in range(a.steps):
        x, y = S.make_batch(ids, a.block, a.batch, rng)
        x, y = x.to(dev), y.to(dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        model.dla_step(x, y, state)
        if dev == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - t)

    sec_step = statistics.median(step_times)
    print(f"\n  wake step: median {sec_step*1000:.1f} ms  "
          f"(min {min(step_times)*1000:.1f}, max {max(step_times)*1000:.1f})")

    # ---- time eval -------------------------------------------------------
    eval_times = []
    for i in range(a.evals):
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        S.eval_ppl(model, ids[: a.block * 64], dev, a.block, a.eval_batches, seed=1 + i)
        if dev == "cuda":
            torch.cuda.synchronize()
        eval_times.append(time.time() - t)
    sec_eval = statistics.median(eval_times)
    print(f"  eval pass ({a.eval_batches} batches): median {sec_eval*1000:.1f} ms")

    peak = torch.cuda.max_memory_allocated() if dev == "cuda" else 0
    if dev == "cuda":
        print(f"\n  peak VRAM: {human(peak)}  "
              f"(reserved {human(torch.cuda.max_memory_reserved())})")

    # ---- project a full run ---------------------------------------------
    n_eval_per_probe = max(1, a.probe_steps // 2)
    per_seed = (2 * a.history_steps * sec_step          # EH and HE histories
                + 2 * a.probe_steps * sec_step          # two probes
                + (2 * (n_eval_per_probe + 1) + 4) * sec_eval)  # evals + birth ppl
    total = per_seed * a.seeds

    print("\n" + "=" * 72)
    print("Projected cost")
    print("=" * 72)
    print(f"  per seed : {per_seed/60:6.1f} min")
    print(f"  {a.seeds} seeds  : {total/60:6.1f} min  = {total/3600:.2f} h")
    print(f"  at ¥{a.rate}/h : ¥{total/3600*a.rate:.2f}")
    print(f"  + setup/upload/debug margin (×1.8): ¥{total/3600*a.rate*1.8:.2f}")
    print()
    if peak and peak > 0.9 * torch.cuda.get_device_properties(0).total_memory:
        print("  ⚠️  peak VRAM is near the card limit — reduce --batch or use fp32-free state")
    print("=" * 72)

    if a.json:
        json.dump({
            "model": a.model, "state_dtype": a.state_dtype, "device": dev,
            "sec_per_step": sec_step, "sec_per_eval": sec_eval,
            "peak_vram_bytes": peak, "per_seed_sec": per_seed,
            "total_sec": total, "seeds": a.seeds, "rate": a.rate,
        }, open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}")


if __name__ == "__main__":
    main()
