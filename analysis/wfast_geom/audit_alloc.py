"""Allocation-level selectivity at scale: the paper's core causal test.

Question
--------
During sleep, consolidation writes the fast trace into the slow store.  Does it
matter WHERE it writes, holding the written ENERGY fixed?

    direct     : W_slow += gamma * W_fast          (the rule's own allocation)
    shufwrite  : the same energy, coordinates randomly permuted  (energy-matched)
    nocons     : write nothing at all

If selective write-back is real, `direct` must beat `shufwrite` even though the two
write exactly the same total energy into exactly the same tensors.  `shufwrite`
should in turn sit at `nocons`.  This is the contrast the paper calls allocation
level selectivity; it is NOT the W_fast carrier test (that one is settled).

Protocol
--------
A curriculum of `--stages` history stages, each `--history-steps` wake steps
followed by one sleep under the chosen variant, then a 40-step D-probe whose
perplexity is measured on a HELD-OUT slice.  Within a seed every variant sees the
same history data, the same stage boundaries and the same probe seed, so the only
thing that differs is the sleep write.  Variants are paired per seed.

The backbone is restored between variants, because sleep mutates it in place.
"""
import argparse
import json
import math
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import audit_scale as S  # noqa: E402

VARIANTS = ("direct", "shufwrite", "nocons", "qonly", "uniformwrite", "topwrite",
            "nosleep", "full")


def stable_hash(text: str) -> int:
    """Process-independent hash.

    NOTE: audit_b3.variant_sleep seeds its permutation with abs(hash(key)), and
    Python salts str hashing per process, so its energy-matched shuffle differs on
    every run.  zlib.crc32 makes the permutation reproducible.
    """
    return zlib.crc32(text.encode("utf-8"))


def variant_sleep(model, state, variant, perm_seed=0, gamma_scale=1.0):
    """dla_sleep with the write pathway selected by ``variant``.

    Mirrors analysis/wfast_geom/audit_b3.py:variant_sleep, ported to the GPT
    wrapper, with a deterministic permutation seed.
    """
    tv = model.tempos.values()
    gam = tv["consolidate_fast_direct"] * gamma_scale
    for key, mod in model.key_modules.items():
        s = state.store[key]
        wf = s["w_fast"]
        if variant == "qonly":
            add = tv["consolidate_beta"] * s["q"]
        elif variant == "direct":
            add = gam * wf
        elif variant == "shufwrite":
            flat = wf.reshape(-1)
            gen = torch.Generator(device=flat.device).manual_seed(
                perm_seed * 7919 + stable_hash(key) % 1000003)
            perm = torch.randperm(flat.numel(), generator=gen)
            add = gam * flat[perm].reshape(wf.shape)
        elif variant == "uniformwrite":
            n = wf.numel()
            c = gam * wf.norm() / math.sqrt(n) if n > 0 else 0.0
            add = torch.full_like(wf, c)
        elif variant == "topwrite":
            n = wf.numel()
            k = int(round(0.8 * n))
            flat_abs = wf.reshape(-1).abs()
            thr = flat_abs.topk(k, largest=True).values.min()
            mask = wf.abs() >= thr
            keep = mask.sum().item()
            c = gam * wf.norm() / math.sqrt(keep) if keep > 0 else 0.0
            add = torch.where(mask, torch.full_like(wf, c), torch.zeros_like(wf))
            del flat_abs
        elif variant in ("nocons", "nosleep"):
            add = None
        else:  # full
            add = tv["consolidate_beta"] * s["q"] + gam * wf
        if add is not None:
            mod.weight.add_(add)
        s["w_fast"].mul_(tv["consolidate_fast_decay"])
        s["q"].mul_(tv["consolidate_q_decay"])
    state.reset_moments()
    return state


def run_arm(model, base_sd, stages, variant, args, device, d_train, d_eval,
            probe_seed):
    """One history+sleeps+probe arm.  Restores the backbone first."""
    model.set_dla_state(None)
    model.load_state_dict(base_sd, strict=False)
    torch.cuda.empty_cache()

    state = model.make_state(device)
    for t, st_ids in enumerate(stages):
        S.train_history(model, state, st_ids, device, args.history_steps,
                        args.block, args.batch, seed=args.seed * 1000 + 100 + t)
        if variant != "nosleep":
            variant_sleep(model, state, variant, perm_seed=args.seed * 1000 + t)
    pre, curve, gain = S.run_probe(model, state, d_train, d_eval, device,
                                   args.probe_steps, args.block, args.batch,
                                   seed=probe_seed, eval_batches=args.eval_batches)
    del state
    model.set_dla_state(None)
    torch.cuda.empty_cache()
    return {"pre": pre, "gain40": gain, "final_ppl": curve[-1]["ppl"],
            "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="gpt2-large", choices=list(S.MODELS))
    ap.add_argument("--variants", default="direct,shufwrite,nocons")
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=60000,
                    help="total history tokens (split evenly over stages)")
    ap.add_argument("--d-train", type=int, default=60000)
    ap.add_argument("--d-eval", type=int, default=20000)
    ap.add_argument("--history-steps", type=int, default=40)
    ap.add_argument("--probe-steps", type=int, default=40)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--state-dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.expanduser("~/dla_alloc"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    model = S.build(a.model, a.state_dtype, a.device)
    n_par = sum(p.numel() for p in model.parameters())
    ids = S.get_corpus(None, model.config.vocab_size)

    stage_len = a.chunk // a.stages
    stages = [ids[i * stage_len:(i + 1) * stage_len] for i in range(a.stages)]
    off = a.stages * stage_len
    d_train = ids[off:off + a.d_train]
    d_eval = ids[off + a.d_train:off + a.d_train + a.d_eval]
    if len(d_eval) < a.d_eval:
        raise ValueError(f"corpus too short: need {off + a.d_train + a.d_eval}")

    print(f"[alloc] backbone={a.model} params={n_par/1e6:.1f}M "
          f"stages={a.stages}x{stage_len} D_train={len(d_train)} "
          f"D_eval={len(d_eval)}", flush=True)

    # keep the pristine backbone (incl. tempos) on CPU so every variant starts equal
    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out = {"seed": a.seed, "model": a.model, "params": n_par,
           "stages": a.stages, "chunk": a.chunk, "arms": {}}
    probe_seed = 800000 + a.seed
    for variant in [v.strip() for v in a.variants.split(",") if v.strip()]:
        r = run_arm(model, base_sd, stages, variant, a, a.device, d_train, d_eval,
                    probe_seed)
        out["arms"][variant] = r
        print(f"[alloc] arm={variant:13s} pre={r['pre']:8.2f} "
              f"gain40={r['gain40']:+.6f} final={r['final_ppl']:8.2f}", flush=True)

    arm = out["arms"]
    if "direct" in arm and "shufwrite" in arm:
        out["tau_shuf"] = arm["shufwrite"]["gain40"] - arm["direct"]["gain40"]
        print(f"[alloc] seed={a.seed} tau_shuf (shuf-direct)={out['tau_shuf']:+.4f}",
              flush=True)
    if "direct" in arm and "nocons" in arm:
        out["tau_nocons"] = arm["nocons"]["gain40"] - arm["direct"]["gain40"]
        print(f"[alloc] seed={a.seed} tau_nocons (nocons-direct)="
              f"{out['tau_nocons']:+.4f}", flush=True)

    p = os.path.join(a.out, f"alloc_{a.model}_s{a.seed}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[alloc] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
