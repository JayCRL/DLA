"""leak_gain.py -- disentangling the EMERGENCE and EXPRESSION of parameter-level
selective writeback in leaky fast-weight dynamics.

Two distinct control variables
------------------------------
The mechanism is approximated by

    wake :  W_fast[t+1] = (1 - fast_decay) * W_fast[t] + dW[t]
    sleep:  W_fast <- lambda * W_fast
            W_slow <- W_slow + gamma * W_fast          (direct writeback)

    fast_decay  controls how long a past update stays in the fast store
                (the MEMORY TIMESCALE).
    lambda      controls how strongly whatever survives scales the sleep
                writeback (the WRITEBACK INFLUENCE / GAIN).  lambda > 1 is a
                legitimate amplification, NOT a retention factor.

Keeping these two apart is the whole point of this script.  In the earlier
`audit_alloc --lambdas` sweep lambda was varied with fast_decay fixed at 0.02,
and since 40 wake steps already multiply the old content by 0.98^40 = 0.446,
lambda only ever modulated an already-attenuated store -- the sweep could not
tell a timescale effect from a gain effect.

What is measured, and why each one
----------------------------------
magnitude      ||W_fast||, mean|W|, std|W|, max|W|          -- how BIG the store is
normalized     cv, top-1/5/10% mass, normalized entropy,    -- how CONCENTRATED
selectivity      gini, hhi                                     its distribution is

Every normalized metric is computed on |W| / sum|W| and is therefore exactly
scale-invariant.  This is deliberate: lambda > 1 multiplies every coordinate by
the same factor and CANNOT change any of them.  If lambda moves the normalized
metrics it must be doing something beyond scaling; if it only moves ||W_fast||
then the honest conclusion is that lambda controls the STRENGTH of a selective
writeback rather than the EMERGENCE of selectivity.  summary_leakgain.py tests
that directly by correlating the two families.

timescale      cos(W_fast_final, W_fast_after_stage_k), and an empirical
               lag-cosine decay curve cos(W_t, W_{t+lag}) measured from the
               actual history trajectory (not inferred from the formula), with
               the 1/e horizon read off that curve.

causal         every (fast_decay, lambda) cell runs direct / shufwrite / nocons.
               B = final_ppl(shufwrite) - final_ppl(direct) > 0 means the native
               coordinate allocation beats an energy-matched shuffle.
               final_ppl is the primary metric: it does not contain `pre`, so it
               cannot inherit the post-treatment starting-level confound that
               makes a relative gain unusable.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

import audit_scale as S  # noqa: E402
from audit_alloc import variant_sleep  # noqa: E402


# --------------------------------------------------------------------------
# magnitude vs normalized selectivity
# --------------------------------------------------------------------------
def magnitude_metrics(wf_by_key):
    """Scale-dependent descriptions of the fast store (how big it is)."""
    if not wf_by_key:
        return {}
    flat = torch.cat([v.reshape(-1).abs().float() for v in wf_by_key.values()])
    if flat.numel() == 0:
        return {}
    return {
        "w_fast_norm": float(flat.norm()),
        "abs_mean": float(flat.mean()),
        "abs_std": float(flat.std(unbiased=True)),
        "abs_max": float(flat.max()),
        "abs_sum": float(flat.sum()),
    }


def normalized_selectivity(wf_by_key):
    """Scale-INVARIANT concentration of the fast store.

    Every number here is unchanged if W_fast is multiplied by any positive
    constant -- which is exactly what a sleep `lambda` does.  So a change in
    these metrics under a lambda sweep is evidence that lambda is doing more
    than scaling.
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
        "cv": float(flat.std(unbiased=True) / flat.mean()),
        "normalized_entropy": float(-(p * (p + 1e-12).log()).sum() / math.log(n)),
        "hhi": float((p ** 2).sum()),
    }
    xs, _ = torch.sort(flat)
    idx = torch.arange(1, n + 1, dtype=xs.dtype, device=xs.device)
    out["gini"] = float(2 * (idx * xs).sum() / (n * tot) - (n + 1) / n)
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(round(frac * n)))
        out[f"top{int(frac * 100)}pct_mass"] = float(
            torch.topk(flat, k, largest=True).values.sum() / tot)
    return out


# --------------------------------------------------------------------------
# memory timescale
# --------------------------------------------------------------------------
def pooled(wf_by_key, keys):
    return torch.cat([wf_by_key[k].reshape(-1).float() for k in keys])


def cos(a, b):
    d = float(a.norm() * b.norm())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def timescale_metrics(final_wf, stage_wf):
    """Survival of each earlier stage's fast state in the final W_fast."""
    if not final_wf or not stage_wf:
        return {}
    keys = sorted(final_wf)
    f = pooled(final_wf, keys)
    out, cs = {}, []
    for i, snap in enumerate(stage_wf):
        c = cos(f, pooled(snap, keys))
        out[f"cos_stage{i + 1}"] = c
        cs.append(c)
    a = [abs(c) for c in cs if c == c]
    if a and sum(a) > 0:
        out["horizon"] = float(sum((i + 1) * c for i, c in enumerate(a)) / sum(a))
    return out


def lag_decay_metrics(snaps, snap_every):
    """Empirical cos(W_t, W_{t+lag}) from the real trajectory.

    ``snaps`` is the list of pooled W_fast vectors recorded every ``snap_every``
    wake steps.  The curve is measured, never inferred from (1-fast_decay)^k.
    ``horizon_steps`` is the lag (in wake steps) at which the mean curve first
    falls to 1/e, linearly interpolated; it is censored at the trajectory length
    when the store outlives the history -- reported as ``censored=True`` rather
    than silently capped.
    """
    n = len(snaps)
    if n < 3:
        return {}
    out = {"n_snaps": n, "snap_every": snap_every}
    curve = {}
    for lag in range(1, n):
        vals = [cos(snaps[t], snaps[t + lag]) for t in range(n - lag)]
        vals = [v for v in vals if v == v]
        curve[lag] = float(sum(vals) / len(vals)) if vals else float("nan")
    out["lag_cos"] = {str(k * snap_every): v for k, v in curve.items()}
    # 1/e crossing on the lag-steps axis
    thr = 1.0 / math.e
    lag_steps = sorted(curve)
    hz, censored = None, True
    for i, lag in enumerate(lag_steps):
        if curve[lag] <= thr:
            if i == 0:
                hz = float(lag * snap_every)
            else:
                l0, l1 = lag_steps[i - 1], lag
                c0, c1 = curve[l0], curve[l1]
                frac = (c0 - thr) / (c0 - c1) if c0 != c1 else 0.0
                hz = float((l0 + frac * (l1 - l0)) * snap_every)
            censored = False
            break
    out["horizon_steps"] = hz if hz is not None else float(n * snap_every)
    out["censored"] = censored
    out["cos_lag1"] = curve.get(1, float("nan"))
    return out


# --------------------------------------------------------------------------
def set_fast_decay(model, fd):
    """Set the wake leak exactly.

    ``fast_decay`` is a sigmoid-parameterised tempo, and the wake rule reads
    ``sigmoid(logit_fast_decay)``.  Writing the logit directly reproduces the
    requested value to ~1e-18 for every point on the sweep grid, so the sweep
    never fights the _logit clamp.
    """
    p = model.tempos.logit_fast_decay
    p.data.fill_(math.log(fd / (1.0 - fd)))


def run_cell(model, base_sd, stages, variant, args, device, d_train, d_eval,
             probe_seed, fd, lam):
    """One (fast_decay, lambda, variant) cell."""
    model.set_dla_state(None)
    model.load_state_dict(base_sd, strict=False)
    set_fast_decay(model, fd)
    torch.cuda.empty_cache()

    state = model.make_state(device)
    stage_wf, snaps, trace = [], [], []
    nan = False
    for t, st_ids in enumerate(stages):
        # Inlined rather than calling S.train_history once per step: that helper
        # seeds its own rng, so calling it 40 times with 40 seeds would draw a
        # DIFFERENT batch sequence than the single 40-step call used by P1 /
        # audit_alloc.  This loop reproduces train_history exactly -- same seed,
        # same rng, same sequential batches -- and only adds the snapshots.
        rng = torch.Generator().manual_seed(args.seed * 1000 + 100 + t)
        model.set_dla_state(state)
        for step in range(args.history_steps):
            x, y = S.make_batch(st_ids, args.block, args.batch, rng)
            x, y = x.to(device), y.to(device)
            model.dla_step(x, y, state)
            if args.snap_every and (step + 1) % args.snap_every == 0:
                # Concatenate on the GPU and cast to fp16 BEFORE the host
                # copy: the per-layer .float().cpu().half() chain spent most of
                # its time on CPU (observed ~1300% CPU) transferring 4 bytes per
                # parameter instead of 2.
                snaps.append(torch.cat([
                    s["w_fast"].reshape(-1) for _, s in sorted(state.store.items())
                ]).detach().to(torch.float16).cpu())
        if variant != "nosleep":
            variant_sleep(model, state, variant, perm_seed=args.seed * 1000 + t,
                          fast_decay=lam)
        wf = {k: s["w_fast"].detach().float().cpu().clone()
              for k, s in state.store.items()}
        stage_wf.append(wf)
        nm = float(sum(v.norm() ** 2 for v in wf.values()) ** 0.5)
        mx = max(float(v.abs().max()) for v in wf.values())
        trace.append({"stage": t + 1, "w_fast_norm": nm, "w_fast_absmax": mx})
        if not math.isfinite(nm) or not math.isfinite(mx):
            nan = True
            print(f"[lg] ** NON-FINITE W_fast at stage {t+1} fd={fd:g} lam={lam:g} "
                  f"variant={variant} -- cell aborted", flush=True)
            break

    rec = {"fast_decay": fd, "lambda": lam, "variant": variant,
           "w_fast_trace": trace, "nonfinite": nan}
    if nan:
        rec.update({"pre": float("nan"), "final_ppl": float("nan"),
                    "gain40": float("nan"), "curve": []})
        model.set_dla_state(None)
        torch.cuda.empty_cache()
        return rec

    final_wf = stage_wf[-1]
    rec["magnitude"] = magnitude_metrics(final_wf)
    rec["selectivity"] = normalized_selectivity(final_wf)
    rec["timescale"] = timescale_metrics(final_wf, stage_wf[:-1])
    rec["lag_decay"] = lag_decay_metrics(snaps, args.snap_every) if snaps else {}

    pre, curve, gain = S.run_probe(model, state, d_train, d_eval, device,
                                   args.probe_steps, args.block, args.batch,
                                   seed=probe_seed, eval_batches=args.eval_batches)
    rec.update({"pre": pre, "gain40": gain, "final_ppl": curve[-1]["ppl"],
                "drop": pre - curve[-1]["ppl"], "curve": curve})

    del state, stage_wf, snaps
    model.set_dla_state(None)
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="gpt2", choices=list(S.MODELS))
    ap.add_argument("--fast-decays", default="0.02")
    ap.add_argument("--lambdas", default="1.0")
    ap.add_argument("--variants", default="direct,shufwrite,nocons")
    ap.add_argument("--gamma-scale", type=float, default=1.0,
                    help="multiplies the writeback energy gamma; kept at 1.0 so "
                         "only fast_decay and lambda move")
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=60000)
    ap.add_argument("--d-train", type=int, default=60000)
    ap.add_argument("--d-eval", type=int, default=20000)
    ap.add_argument("--history-steps", type=int, default=40)
    ap.add_argument("--probe-steps", type=int, default=40)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--snap-every", type=int, default=10,
                    help="record W_fast every N wake steps for the lag-decay "
                         "curve; 0 disables it")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--state-dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/dla_leakgain"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fds = [float(x) for x in a.fast_decays.split(",") if x.strip()]
    lams = [float(x) for x in a.lambdas.split(",") if x.strip()]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if any(l <= 0 for l in lams):
        raise SystemExit("lambda must be > 0")

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
        raise SystemExit(f"corpus too short: need {off + a.d_train + a.d_eval}")
    print(f"[lg] {a.model} {n_par/1e6:.1f}M stages={a.stages}x{stage_len} "
          f"cells={len(fds)}x{len(lams)}x{len(variants)} "
          f"fast_decay={fds} lambda={lams}", flush=True)

    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    probe_seed = 800000 + a.seed
    out = {"seed": a.seed, "model": a.model, "params": n_par,
           "stages": a.stages, "chunk": a.chunk,
           "gamma_scale": a.gamma_scale, "cells": {}}

    partial_path = os.path.join(a.out, f"leakgain_{a.model}_s{a.seed}.json")

    def flush():
        """Write the JSON after every cell.

        Without this a long grid leaves no trace until the very end, so a run
        that is merely slow looks identical to one that is stuck -- and the
        queue log is additionally block-buffered by the grep in its pipeline,
        which is exactly how a healthy run first looked like a silent failure.
        Flushing per cell makes progress observable from the file itself.
        """
        with open(partial_path, "w") as f:
            json.dump(out, f, indent=1)

    for fd in fds:
        for lam in lams:
            cell = {}
            for variant in variants:
                r = run_cell(model, base_sd, stages, variant, a, a.device,
                             d_train, d_eval, probe_seed, fd, lam)
                cell[variant] = r
                sel = r.get("selectivity") or {}
                mag = r.get("magnitude") or {}
                lg = r.get("lag_decay") or {}
                print(f"[lg] fd={fd:<6g} lam={lam:<5g} {variant:10s} "
                      f"pre={r['pre']:8.2f} final={r['final_ppl']:8.2f} "
                      f"|W|={mag.get('w_fast_norm', float('nan')):8.3f} "
                      f"cv={sel.get('cv', float('nan')):.3f} "
                      f"top1={sel.get('top1pct_mass', float('nan')):.4f} "
                      f"hz={lg.get('horizon_steps', float('nan')):.0f}"
                      + ("  ** NAN" if r.get("nonfinite") else ""), flush=True)
            # causal contrasts: final_ppl is the primary metric (no `pre` in it)
            if "direct" in cell:
                D = cell["direct"]
                for other in variants:
                    if other == "direct" or other not in cell:
                        continue
                    tag = "shuf" if other == "shufwrite" else other
                    A = cell[other]
                    cell[f"B_{tag}"] = A["final_ppl"] - D["final_ppl"]
                    cell[f"B_{tag}_gain"] = A["gain40"] - D["gain40"]
                    cell[f"preimb_{tag}"] = A["pre"] - D["pre"]
                    print(f"[lg] fd={fd:<6g} lam={lam:<5g} B_{tag}="
                          f"{cell[f'B_{tag}']:+.4f} (final_ppl, >0 => direct better)",
                          flush=True)
            out["cells"][f"fd{fd:g}|lam{lam:g}"] = cell
            flush()

    p = os.path.join(a.out, f"leakgain_{a.model}_s{a.seed}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[lg] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
