"""audit_scale_v2 -- history-order carrier test with the starting-level confound removed.

Why v2 exists
-------------
v1 built the two history arms from two chunks and let each arm END on a different
chunk:

    EH = E(hard) -> H(easy)      ends on the EASY chunk   -> good immediate state
    HE = H(easy) -> E(hard)      ends on the HARD chunk   -> worse immediate state

The last chunk dominates the immediate state, so the arms did not start the probe at
the same perplexity: HE began ~2 PPL worse at every scale (12/12, 11/12, 9/12 seeds).
Because the headline metric was a RELATIVE gain (pre-final)/pre, and a relative gain
rewards whichever arm starts worse, the comparison was largely measuring the starting
level.  Measured on the v1 results:

    corr(pre imbalance, gain40 effect) = +0.915 / +0.870 / +0.929   (124M/355M/774M)
    equal-pre corrected effect         = +0.0039 / +0.0012 / -0.0044  (10x, 36x, sign flip)

The same pattern appeared in the 6.59M archive (corr = +0.972) and in the W_fast swap
experiment (corr = +0.992, 92% of the effect removed by the correction).

The fix
-------
Every arm now ends on the SAME final chunk D:

    ABD = A -> B -> D
    BAD = B -> A -> D
               ^ same last stage, same data, same batch order

The manipulation is the order of A and B only.  The "which chunk did I end on"
asymmetry is gone by construction, so the arms should start the probe at a matched
perplexity -- and that match is now CHECKED, not assumed.

Two extra content controls are available with --controls:

    AAD = A -> A -> D      (A content only)
    BBD = B -> B -> D      (B content only)

They separate "the order of two different chunks" from "the content of the chunks".

Metrics
-------
    gain40     (pre - final)/pre          level-confounded; kept for continuity
    drop       pre - final                partly level-dependent
    final_ppl  absolute end-of-probe       level-free
    T80        steps to reach 80% of the arm's own total drop  (convergence speed)

plus the full perplexity curve, so any other speed metric can be derived later.

Acceptance checks (printed and stored, not optional)
---------------------------------------------------
    pre imbalance between arms, and corr(pre imbalance, gain40 effect) across seeds
    flagged LEVEL-CONFOUNDED when |corr| is high -- the failure mode of v1
"""
import argparse
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

import audit_scale as S  # noqa: E402

CHUNK_SEED = {"A": 11, "B": 23, "C": 37, "D": 41}


def t80(curve, frac=0.8):
    """Steps for the arm to capture `frac` of its own total 40-step perplexity drop."""
    ppls = [c["ppl"] for c in curve]
    total = ppls[0] - ppls[-1]
    if total <= 0:
        return len(curve)
    target = ppls[0] - frac * total
    for c in curve:
        if c["ppl"] <= target:
            return c["step"]
    return curve[-1]["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="gpt2-large", choices=list(S.MODELS))
    ap.add_argument("--state-dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=40000)
    ap.add_argument("--history-steps", type=int, default=40)
    ap.add_argument("--probe-steps", type=int, default=40)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--d-train", type=int, default=60000)
    ap.add_argument("--d-eval", type=int, default=20000)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--controls", action="store_true",
                    help="also run AAD and BBD (content controls)")
    ap.add_argument("--legacy-layout", action="store_true",
                    help="reproduce v1's two-chunk layout (for the confound demo)")
    ap.add_argument("--out", default=os.path.expanduser("~/dla_scale_v2"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device

    model = S.build(a.model, a.state_dtype, dev)
    n_par = sum(p.numel() for p in model.parameters())
    corpus = a.corpus or os.path.expanduser("~/llm-lab/hf_gpt2/shakespeare_bpe.npy")
    ids = S.get_corpus(corpus, model.config.vocab_size)
    print(f"[v2] {a.model} {n_par/1e6:.1f}M  corpus={len(ids):,} tokens", flush=True)

    A = ids[: a.chunk]
    B = ids[a.chunk: 2 * a.chunk]
    if a.legacy_layout:
        # v1: no common final chunk; the arms end on different chunks.
        arms = ["AB", "BA"]
        chunks = {"A": A, "B": B}
        off = 2 * a.chunk
    else:
        Dfin = ids[2 * a.chunk: 3 * a.chunk]
        arms = ["ABD", "BAD"] + (["AAD", "BBD"] if a.controls else [])
        chunks = {"A": A, "B": B, "D": Dfin}
        off = 3 * a.chunk
    d_train = ids[off: off + a.d_train]
    d_eval = ids[off + a.d_train: off + a.d_train + a.d_eval]
    if len(d_eval) < a.d_eval:
        raise SystemExit(f"corpus too short: need {off + a.d_train + a.d_eval}, "
                         f"have {len(ids)}")
    print(f"[v2] layout off={off} d_train={len(d_train)} d_eval={len(d_eval)} "
          f"arms={arms}", flush=True)

    out = {"seed": a.seed, "model": a.model, "params": n_par,
           "layout": "legacy" if a.legacy_layout else "common-final-chunk",
           "chunk": a.chunk, "history_steps": a.history_steps, "arms": {}}

    probe_kw = dict(seed=800000 + a.seed, eval_batches=a.eval_batches)
    for arm in arms:
        model.set_dla_state(None)
        torch.cuda.empty_cache()
        state = model.make_state(dev)
        for tag in arm:
            S.train_history(model, state, chunks[tag], dev, a.history_steps,
                            a.block, a.batch, seed=a.seed * 1000 + CHUNK_SEED[tag])
        pre, curve, gain = S.run_probe(model, state, d_train, d_eval, dev,
                                       a.probe_steps, a.block, a.batch, **probe_kw)
        fin = curve[-1]["ppl"]
        out["arms"][arm] = {
            "pre": pre, "final_ppl": fin, "gain40": gain,
            "drop": pre - fin, "t80": t80(curve), "curve": curve,
        }
        print(f"[v2] arm={arm:4s} pre={pre:8.2f} final={fin:8.2f} "
              f"gain40={gain:+.6f} drop={pre-fin:7.3f} T80={t80(curve)}", flush=True)
        del state
        model.set_dla_state(None)
        torch.cuda.empty_cache()

    # ---- acceptance check: did the fix actually match the starting points? ----
    if not a.legacy_layout and "ABD" in out["arms"] and "BAD" in out["arms"]:
        x, y = out["arms"]["ABD"], out["arms"]["BAD"]
        imb = y["pre"] - x["pre"]
        out["pre_imbalance"] = imb
        out["delta_gain"] = y["gain40"] - x["gain40"]
        out["delta_final"] = x["final_ppl"] - y["final_ppl"]   # >0 => BAD better
        print(f"[v2] seed={a.seed} pre imbalance(BAD-ABD)={imb:+.3f}  "
              f"d_gain={out['delta_gain']:+.4f}  d_final={out['delta_final']:+.4f}",
              flush=True)
    elif a.legacy_layout:
        x, y = out["arms"]["AB"], out["arms"]["BA"]
        out["pre_imbalance"] = y["pre"] - x["pre"]
        out["delta_gain"] = y["gain40"] - x["gain40"]
        out["delta_final"] = x["final_ppl"] - y["final_ppl"]
        print(f"[v2] seed={a.seed} LEGACY pre imbalance={out['pre_imbalance']:+.3f}  "
              f"d_gain={out['delta_gain']:+.4f}", flush=True)

    p = os.path.join(a.out, f"v2_{a.model}_s{a.seed}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[v2] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
