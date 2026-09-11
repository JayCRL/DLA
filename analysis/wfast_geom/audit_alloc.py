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


def variant_sleep(model, state, variant, perm_seed=0, gamma_scale=1.0,
                  fast_decay=None):
    """dla_sleep with the write pathway selected by ``variant``.

    Mirrors analysis/wfast_geom/audit_b3.py:variant_sleep, ported to the GPT
    wrapper, with a deterministic permutation seed.
    """
    tv = model.tempos.values()
    gam = tv["consolidate_fast_direct"] * gamma_scale
    # W_fast RETENTION lambda.  dla_sleep does ``w_fast *= consolidate_fast_decay``,
    # so that tempo IS the retention factor: larger == slower leak (leak_rate =
    # 1 - lambda).  The direction is therefore NOT reversed -- but the tempo is
    # sigmoid-parameterised and _logit clamps p to [1e-4, 1-1e-4], so it can never
    # express an exact 0 or 1.  ``fast_decay`` overrides the value directly for the
    # leak sweep, so lambda=0 clears W_fast exactly and lambda=1 retains it exactly.
    decay = tv["consolidate_fast_decay"] if fast_decay is None else float(fast_decay)
    # audit_scale.build makes every backbone weight require grad (the wake rule
    # consumes backbone grads), so the sleep write must be an explicit no-grad edit.
    ctx = torch.no_grad()
    ctx.__enter__()
    for key, mod in model.key_modules.items():
        s = state.store[key]
        wf = s["w_fast"]
        if variant == "qonly":
            add = tv["consolidate_beta"] * s["q"]
        elif variant == "direct":
            add = gam * wf
        elif variant == "shufwrite":
            flat = wf.reshape(-1)
            # Build the permutation on CPU: torch.randperm wants a CPU generator,
            # and a CPU-side permutation keeps the shuffle identical across devices.
            gen = torch.Generator().manual_seed(
                perm_seed * 7919 + stable_hash(key) % 1000003)
            perm = torch.randperm(flat.numel(), generator=gen).to(flat.device)
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
        s["w_fast"].mul_(decay)
        s["q"].mul_(tv["consolidate_q_decay"])
    ctx.__exit__(None, None, None)
    state.reset_moments()
    return state


def _gini(x):
    """Gini coefficient of a non-negative 1-D tensor (0 = uniform, ->1 = one coord)."""
    n = x.numel()
    tot = x.sum()
    if n == 0 or tot <= 0:
        return float("nan")
    xs, _ = torch.sort(x)
    idx = torch.arange(1, n + 1, dtype=xs.dtype, device=xs.device)
    return float(2 * (idx * xs).sum() / (n * tot) - (n + 1) / n)


def selectivity_metrics(wf_by_key):
    """Parameter-level selectivity of a W_fast state.

    Computed on the pooled |W_fast| over every wrapped layer, so the numbers
    describe the whole fast store rather than one matrix.  Every metric is a
    concentration measure and they are reported together on purpose: a uniform
    |W_fast| sits near cv~0 / top-1% ~1% / normalized_entropy~1 / gini~0, and a
    state that puts its mass on few coordinates moves all four the other way.
    No single one of them is trusted as "the" selectivity index.
    """
    if not wf_by_key:
        return {}
    flat = torch.cat([v.reshape(-1).abs().float() for v in wf_by_key.values()])
    n = flat.numel()
    tot = flat.sum()
    if n == 0 or tot <= 0:
        return {}
    p = flat / tot
    out = {
        "w_fast_norm": float(tot),
        "cv": float(flat.std(unbiased=True) / flat.mean()),
        "normalized_entropy": float(-(p * (p + 1e-12).log()).sum() / math.log(n)),
        "gini": _gini(flat),
        "hhi": float((p ** 2).sum()),
    }
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(round(frac * n)))
        out[f"top{int(frac * 100)}pct_mass"] = float(
            torch.topk(flat, k, largest=True).values.sum() / tot)
    return out


def timescale_metrics(final_wf, stage_wf):
    """How far back the fast store still remembers.

    cos(W_fast_final, W_fast_at_stage_k): with retention lambda the stage-k
    contribution to the final W_fast carries a factor lambda^(K-k), so at
    lambda->0 only the last stage survives (early-stage cosines fall to 0) and
    at lambda->1 every stage contributes.  ``horizon`` is the |cos|-weighted mean
    stage index (1-indexed): a small value means the store is dominated by the
    OLDEST stage, a value near K means it holds only the most recent one.
    """
    if not final_wf or not stage_wf:
        return {}
    keys = sorted(final_wf)
    f = torch.cat([final_wf[k].reshape(-1).float() for k in keys])
    cos = []
    for snap in stage_wf:
        s = torch.cat([snap[k].reshape(-1).float() for k in keys])
        d = float(f.norm() * s.norm())
        cos.append(float((f * s).sum() / d) if d > 0 else float("nan"))
    out = {f"cos_stage{i + 1}": c for i, c in enumerate(cos)}
    a = [abs(c) for c in cos if c == c]
    if a and sum(a) > 0:
        out["horizon"] = float(sum((i + 1) * c for i, c in enumerate(a)) / sum(a))
    return out


def run_arm(model, base_sd, stages, variant, args, device, d_train, d_eval,
            probe_seed, fast_decay=None):
    """One history+sleeps+probe arm.  Restores the backbone first."""
    model.set_dla_state(None)
    model.load_state_dict(base_sd, strict=False)
    torch.cuda.empty_cache()

    state = model.make_state(device)
    stage_wf = []
    for t, st_ids in enumerate(stages):
        S.train_history(model, state, st_ids, device, args.history_steps,
                        args.block, args.batch, seed=args.seed * 1000 + 100 + t)
        if variant != "nosleep":
            variant_sleep(model, state, variant, perm_seed=args.seed * 1000 + t,
                          fast_decay=fast_decay)
        # Snapshot W_fast at each sleep boundary (CPU, float32) so the
        # memory-timescale metrics can ask how much of each stage is still
        # present in the store at the end.  Scalar metrics are kept; the
        # snapshots themselves are dropped before the next arm.
        if getattr(args, "measure_mechanism", False):
            stage_wf.append({k: s["w_fast"].detach().float().cpu().clone()
                             for k, s in state.store.items()})
    pre, curve, gain = S.run_probe(model, state, d_train, d_eval, device,
                                   args.probe_steps, args.block, args.batch,
                                   seed=probe_seed, eval_batches=args.eval_batches)
    del state
    model.set_dla_state(None)
    torch.cuda.empty_cache()
    rec = {"pre": pre, "gain40": gain, "final_ppl": curve[-1]["ppl"],
           "curve": curve}
    if stage_wf:
        final_wf = stage_wf[-1]
        rec["selectivity"] = selectivity_metrics(final_wf)
        # cos against the EARLIER stages only; cos with the last stage is 1 by
        # construction and would only dilute the horizon.
        rec["timescale"] = timescale_metrics(final_wf, stage_wf[:-1])
        rec["w_fast_norm_per_stage"] = [float(
            sum(v.norm() ** 2 for v in snap.values()) ** 0.5) for snap in stage_wf]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="gpt2-large", choices=list(S.MODELS))
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--variants", default="direct,shufwrite,nocons")
    ap.add_argument("--lambdas", default="",
                    help="W_fast retention sweep, comma-separated, e.g. "
                         "'0,0.5,0.9,0.95,0.99,0.995,0.999'.  lambda IS the "
                         "retention factor (W_fast *= lambda each sleep), so a "
                         "larger value means a SLOWER leak.  Empty = use the "
                         "tempo's own value (single run, original output format).")
    ap.add_argument("--no-mechanism", dest="measure_mechanism", action="store_false",
                    help="skip the W_fast selectivity / memory-timescale metrics")
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
    corpus = a.corpus or os.path.expanduser("~/llm-lab/hf_gpt2/shakespeare_bpe.npy")
    ids = S.get_corpus(corpus, model.config.vocab_size)

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
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]

    # Leak sweep: one pass per retention lambda.  Empty --lambdas keeps the
    # original single-run behaviour (tempo's own decay) so existing callers and
    # summary_alloc.py keep working unchanged.
    lambdas = [float(x) for x in a.lambdas.split(",") if x.strip()] if a.lambdas else [None]

    def contrasts(arm, prefix):
        """Three-metric contrasts of every non-direct variant against direct."""
        if "direct" not in arm:
            return
        for other in variants:
            if other == "direct" or other not in arm:
                continue
            tag = "shuf" if other == "shufwrite" else other
            A, B = arm[other], arm["direct"]
            out[f"{prefix}tau_{tag}"] = A["gain40"] - B["gain40"]
            # final_ppl: lower is better, so >0 means DIRECT is better.
            out[f"{prefix}tau_{tag}_ppl"] = A["final_ppl"] - B["final_ppl"]
            out[f"{prefix}tau_{tag}_drop"] = (B["pre"] - B["final_ppl"]) - (A["pre"] - A["final_ppl"])
            out[f"{prefix}tau_{tag}_preimb"] = A["pre"] - B["pre"]
            print(f"[alloc] seed={a.seed} {prefix}tau_{tag}: "
                  f"gain40={out[f'{prefix}tau_{tag}']:+.4f}  "
                  f"final_ppl={out[f'{prefix}tau_{tag}_ppl']:+.4f} (lower better, "
                  f">0 means direct wins)  pre-imbalance={out[f'{prefix}tau_{tag}_preimb']:+.4f}",
                  flush=True)

    for lam in lambdas:
        prefix = "" if len(lambdas) == 1 and lam is None else f"lam{lam:g}."
        scope = {} if prefix == "" else None
        for variant in variants:
            r = run_arm(model, base_sd, stages, variant, a, a.device, d_train,
                        d_eval, probe_seed, fast_decay=lam)
            r["lambda"] = lam
            key = variant if prefix == "" else f"lam{lam:g}|{variant}"
            out["arms"][key] = r
            sel = r.get("selectivity") or {}
            ts = r.get("timescale") or {}
            print(f"[alloc] lam={('tempo' if lam is None else f'{lam:g}'):>6s} "
                  f"arm={variant:13s} pre={r['pre']:8.2f} gain40={r['gain40']:+.6f} "
                  f"final={r['final_ppl']:8.2f} "
                  f"cv={sel.get('cv', float('nan')):.6f} "
                  f"top1={sel.get('top1pct_mass', float('nan')):.6f} "
                  f"gini={sel.get('gini', float('nan')):.6f} "
                  f"h={ts.get('horizon', float('nan')):.3f}", flush=True)
            if prefix == "":
                out["arms"][key] = r
        if prefix:
            contrasts({k.split("|")[1]: v for k, v in out["arms"].items()
                       if k.startswith(prefix)}, prefix)

    if len(lambdas) == 1 and lambdas[0] is None:
        contrasts(out["arms"], "")
        p = os.path.join(a.out, f"alloc_{a.model}_s{a.seed}.json")
    else:
        p = os.path.join(a.out, f"leak_{a.model}_s{a.seed}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[alloc] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
