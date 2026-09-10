"""Scale extension: run the HE/EH history-contrast probe on a large GPT-2 backbone.

Motivation
----------
Every result in the paper so far rides on a 6.59M (Chinese char) or 10.65M
(Shakespeare char) backbone.  This script asks whether the allocation-level
selectivity claim survives at ~1B scale by mounting the SAME DLA mechanism on a
frozen pretrained GPT-2 (355M / 774M / 1.5B) and re-running the causal arms:

  * ``direct`` : HE history's W_fast injected before the D-probe  (carrier)
  * ``nocons`` : no history-carried W_fast at all                 (control)
  * ``shuf``   : energy-matched shuffled W_fast                   (structure null)

Design notes / deviations from audit_second.py
----------------------------------------------
* The backbone is HF ``transformers`` GPT2LMHeadModel weights copied into the
  nanoGPT ``GPT`` (identical module names: wte/wpe/h.N.attn.c_attn/...), so
  ``make_dla_gpt_class`` wrapping works unchanged (all nn.Linear/nn.Embedding).
* Memory: the DLA state is 5 tensors shaped like the backbone weights, so at
  774M the state is ~10GB in fp32.  ``--state-dtype bf16`` halves it.
* This is a *scale* probe, not a new protocol: everything else (birth-PPL
  ordering, 40-step D-probe, gain@40) is inherited verbatim.

Run (GPU expected; CPU only for the smoke test):
    python analysis/wfast_geom/audit_scale.py --model gpt2-large --seed 0 \
        --state-dtype bf16 --out ~/dla_scale
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "wfast_geom"))
NANO = os.path.expanduser("~/llm-lab/nanoGPT")
sys.path.insert(0, NANO)

from dla.transformer_dla import TransformerDLAConfig, make_dla_gpt_class  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

DLA_GPT = make_dla_gpt_class(GPT)

# HF repo -> nanoGPT GPTConfig.  GPT-2 configs are standard (no RoPE/RMSNorm).
MODELS = {
    "gpt2":        GPTConfig(n_layer=12, n_head=12, n_embd=768,  block_size=1024),
    "gpt2-medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024, block_size=1024),
    "gpt2-large":  GPTConfig(n_layer=36, n_head=20, n_embd=1280, block_size=1024),
    "gpt2-xl":     GPTConfig(n_layer=48, n_head=25, n_embd=1600, block_size=1024),
}


LOCAL_WEIGHTS = os.path.expanduser("~/llm-lab/hf_gpt2")


def load_hf_into_gpt(model_name: str, device: str = "cpu"):
    """Materialise a nanoGPT GPT carrying HF GPT-2 weights (backbone frozen)."""
    from transformers import GPT2LMHeadModel

    cfg = MODELS[model_name]
    # Prefer a locally pre-downloaded directory (mirror path); hf_hub's own
    # network stack is unreliable behind this host's proxy setup.
    local = os.path.join(LOCAL_WEIGHTS, model_name)
    src = local if os.path.isdir(local) else model_name
    hf = GPT2LMHeadModel.from_pretrained(src)
    sd = hf.state_dict()

    # HF GPT2LMHeadModel and nanoGPT GPT share key names, but two structural
    # differences have to be reconciled:
    #   1. HF stores Conv1D weights as (in, out); nanoGPT uses nn.Linear (out, in)
    #      -> transpose every 2-D weight whose shape disagrees.
    #   2. nanoGPT pads the vocab (50257 -> 50304) for kernel efficiency.
    base = GPT(cfg)
    target = base.state_dict()
    mapped = {}
    for k, v in sd.items():
        if k == "lm_head.weight":
            continue
        t = target.get(k)
        if t is None:
            continue
        if v.shape != t.shape:
            if v.dim() == 2 and v.shape == t.shape[::-1]:
                v = v.t().contiguous()
            elif k == "transformer.wte.weight" and v.shape[1] == t.shape[1]:
                pad = torch.zeros(t.shape, dtype=v.dtype)
                pad[: v.shape[0]] = v
                v = pad
            else:
                raise ValueError(f"cannot reconcile {k}: {tuple(v.shape)} vs {tuple(t.shape)}")
        mapped[k] = v

    missing, unexpected = base.load_state_dict(mapped, strict=False)
    missing = [m for m in missing if "lm_head" not in m]
    assert not missing, f"unmapped keys: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {list(unexpected)[:5]}"
    del hf, sd
    return base, cfg


def build(model_name: str, state_dtype: str, device: str, eta_fast: float | None = None):
    base, cfg = load_hf_into_gpt(model_name, device="cpu")
    dla_cfg = TransformerDLAConfig(state_dtype=state_dtype)
    if eta_fast is not None:
        dla_cfg.eta_fast = eta_fast
    model = DLA_GPT(cfg, dla_cfg)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    missing = [k for k in missing if not k.startswith("tempos.")]
    assert not missing, f"missing: {missing[:5]}"
    del base
    # The DLA wake rule consumes backbone grads to drive Adam on W_fast, so the
    # wrapped layers must require grad (HF checkpoints load them frozen).
    for p in model.parameters():
        p.requires_grad_(True)
    model.to(device).eval()
    model.to(device).eval()
    return model


def get_corpus(path: str, vocab_size: int):
    """Load a token stream, preserving real token ids for BPE vocabularies.

    ``.npy``/``.bin`` files hold pre-tokenised ids and are used verbatim.  A raw
    ``.txt`` is only safe for character-level backbones whose ids are < vocab;
    for a BPE backbone a .txt would produce meaningless noise (``ord(c) % V``),
    so callers should pass a tokenised file instead.
    """
    if path.endswith(".npy"):
        import numpy as np
        ids = np.load(path)
        return torch.from_numpy(ids.astype("int64"))
    if path.endswith(".txt"):
        raw = open(path, encoding="utf-8").read()
        return torch.tensor([ord(c) % vocab_size for c in raw], dtype=torch.long)
    return torch.tensor([int(t) for t in open(path).read().split()], dtype=torch.long)


def make_batch(ids, block, batch, rng):
    ix = torch.randint(len(ids) - block - 1, (batch,), generator=rng)
    x = torch.stack([ids[i: i + block] for i in ix])
    y = torch.stack([ids[i + 1: i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def eval_ppl(model, ids, device, block=128, batches=4, seed=0):
    """PPL under the CURRENT DLA state.

    The DLA state must stay attached: W_eff = W_slow + softplus(P) * W_fast is
    what carries the history, so evaluating with the state detached would
    measure the bare backbone and make every arm identical.
    """
    rng = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(batches):
        x, y = make_batch(ids, block, 4, rng)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        tot += loss.item()
    return math.exp(tot / batches)


def train_history(model, state, ids, device, steps, block, batch, seed):
    """Wake steps over a history chunk; returns the state in place."""
    rng = torch.Generator().manual_seed(seed)
    model.set_dla_state(state)
    for _ in range(steps):
        x, y = make_batch(ids, block, batch, rng)
        x, y = x.to(device), y.to(device)
        model.dla_step(x, y, state)
    return state


def run_probe(model, state, d_train, d_eval, device, steps, block, batch, seed,
              eval_every=2, eval_batches=2):
    """40-step D-probe, mirroring stage55.run_transfer.

    Train and evaluation data are DELIBERATELY separate: the probe steps are
    taken on ``d_train`` and perplexity is measured on the held-out ``d_eval``.
    This mirrors ``run_transfer`` in experiments/stage55_learning_rule_development.py
    (``get_batch(d_train, ...)`` for the step, ``eval_ppl(model, eb["D"])`` for the
    measurement).

    Using the same slice for both (as an earlier version of this script did) measures
    pure memorisation: a strong backbone drives train-set ppl to ~5 within 40 steps,
    gain@40 saturates at 0.98-0.99 for every arm, and the history contrast collapses
    into the difference of two near-1.0 numbers.  Held-out evaluation restores the
    headroom the protocol is supposed to have.

    Returns (pre, curve, gain40) with gain40 the RELATIVE improvement
    (pre - final) / pre, matching the 6.59M/10.65M protocol.

    NOTE: no ``@torch.no_grad()`` on the step loop -- ``dla_step`` needs grad to
    build the graph that produces the W_fast update.  The state stays attached
    throughout, including during evaluation.
    """
    model.set_dla_state(state)
    with torch.no_grad():
        pre = eval_ppl(model, d_eval, device, block, eval_batches, seed=seed)
    rng = torch.Generator().manual_seed(2_000_000 + seed)
    curve = []
    for t in range(steps):
        x, y = make_batch(d_train, block, batch, rng)
        x, y = x.to(device), y.to(device)
        model.dla_step(x, y, state)
        if (t + 1) % eval_every == 0 or (t + 1) == steps:
            with torch.no_grad():
                ppl = eval_ppl(model, d_eval, device, block, eval_batches,
                               seed=900000 + seed)
            curve.append({"step": t + 1, "ppl": ppl, "gain": (pre - ppl) / pre})
    gain40 = curve[-1]["gain"]
    return pre, curve, gain40


def snapshot_fast(state):
    return {k: v["w_fast"].detach().clone() for k, v in state.store.items()}


def inject_fast(state, snap, shuffle=False, seed=0):
    keys = list(snap.keys())
    if shuffle:
        rng = random.Random(seed)
        perm = keys[:]
        rng.shuffle(perm)
        # energy-matched: permute whole tensors between coordinates of equal shape
        by_shape = {}
        for k in keys:
            by_shape.setdefault(tuple(snap[k].shape), []).append(k)
        mapping = {}
        for shape, ks in by_shape.items():
            perm_ks = ks[:]
            rng.shuffle(perm_ks)
            mapping.update(dict(zip(ks, perm_ks)))
    for k in keys:
        src = mapping[k] if shuffle else k
        state.store[k]["w_fast"].copy_(snap[src])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="gpt2-large", choices=list(MODELS))
    ap.add_argument("--corpus", default=None, help="utf-8 text or whitespace ints")
    ap.add_argument("--state-dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--eta-fast", type=float, default=None,
                    help="override DLA fast-learning rate (stability calibration)")
    ap.add_argument("--history-steps", type=int, default=40)
    ap.add_argument("--probe-steps", type=int, default=40)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4000, help="tokens per history chunk")
    ap.add_argument("--d-train", type=int, default=None,
                    help="probe train tokens (default: same as --chunk)")
    ap.add_argument("--d-eval", type=int, default=None,
                    help="probe held-out tokens (default: same as --chunk)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.expanduser("~/dla_scale"))
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate plumbing")
    ap.add_argument("--save-wfast", action="store_true",
                    help="save W_fast snapshots (large; not needed for the EH/HE contrast)")
    a = ap.parse_args()

    if a.smoke:
        a.history_steps, a.probe_steps, a.chunk, a.batch = 4, 4, 512, 1

    os.makedirs(a.out, exist_ok=True)
    device = a.device

    corpus = a.corpus or os.path.expanduser("~/llm-lab/hf_gpt2/shakespeare_bpe.npy")
    model = build(a.model, a.state_dtype, device, eta_fast=a.eta_fast)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[scale] backbone={a.model} params={n_par/1e6:.1f}M "
          f"state_dtype={a.state_dtype} device={device}", flush=True)

    ids = get_corpus(corpus, model.config.vocab_size)
    print(f"[scale] corpus tokens={len(ids):,}", flush=True)

    # Chunk layout: E and H are the two history stages; D is the probe domain,
    # split into a training slice (probe steps) and a HELD-OUT slice (measurement),
    # mirroring stage55's d_train / eb["D"] separation.
    n_dtr = a.d_train or a.chunk
    n_dev = a.d_eval or a.chunk
    e_ids = ids[: a.chunk]
    h_ids = ids[a.chunk: 2 * a.chunk]
    d_train = ids[2 * a.chunk: 2 * a.chunk + n_dtr]
    d_eval = ids[2 * a.chunk + n_dtr: 2 * a.chunk + n_dtr + n_dev]
    if len(d_eval) < n_dev:
        raise ValueError(
            f"corpus too short: need {2 * a.chunk + n_dtr + n_dev}, have {len(ids)}")
    print(f"[scale] chunks: history={a.chunk} D_train={len(d_train)} "
          f"D_eval={len(d_eval)}", flush=True)

    # birth PPL decides which chunk is "easy" vs "hard"
    model.set_dla_state(None)
    ppl_e = eval_ppl(model, e_ids, device, a.block, 2, seed=1000 + a.seed)
    ppl_h = eval_ppl(model, h_ids, device, a.block, 2, seed=2000 + a.seed)
    easy, hard = ("E", "H") if ppl_e <= ppl_h else ("H", "E")
    print(f"[scale] birth ppl  E={ppl_e:.2f} H={ppl_h:.2f} -> easy={easy}", flush=True)

    chunk = {"E": e_ids, "H": h_ids}
    out = {"seed": a.seed, "model": a.model, "params": n_par,
           "state_dtype": a.state_dtype,
           "birth_ppl": {"E": ppl_e, "H": ppl_h}, "easy": easy, "arms": {}}

    # Deterministic per-chunk history seeds: E and H must see different data
    # AND different batch orderings.  (Never use hash() -- it is salted per
    # process and would silently desynchronise arms across runs.)
    chunk_seed = {"E": 11, "H": 23}

    # Both arms get the probe with an IDENTICAL probe seed: the comparison must
    # differ only in the history that produced the state.  Probe steps use
    # d_train; perplexity is measured on the held-out d_eval.
    #
    # Each arm is built, snapshotted and probed ONE AT A TIME and its state is
    # freed before the next arm is built.  Holding both 5-tensor states at once
    # costs 2 x 14.5 GB at 1.5B on top of the 5.9 GB backbone and the probe graph,
    # which overflows a 48 GB card.  The arms share no state (a probe mutates only
    # its own), so releasing between arms changes the memory profile, not the
    # numbers.
    hist_snap = {}
    probe_kw = dict(seed=800000 + a.seed, eval_batches=8)
    for arm in ("EH", "HE"):
        # Release any state the model still references BEFORE allocating the next
        # one.  ``set_dla_state`` keeps its argument alive, so building the second
        # state on top of the first momentarily holds TWO 14.5 GB states at 1.5B --
        # that transient spike is what overflowed a 48 GB card, not steady-state use.
        model.set_dla_state(None)
        torch.cuda.empty_cache()
        state = model.make_state(device)
        for tag in arm:
            train_history(model, state, chunk[tag], device, a.history_steps,
                          a.block, a.batch, seed=a.seed * 1000 + chunk_seed[tag])
        hist_snap[arm] = snapshot_fast(state)
        # W_fast snapshots are large (1.5 GB at 774M, 3 GB at 1.5B per arm) and
        # are NOT needed for the EH/HE contrast -- keep them opt-in so a sweep
        # over scales does not fill the instance disk.
        if a.save_wfast:
            torch.save({k: v.cpu() for k, v in hist_snap[arm].items()},
                       os.path.join(a.out, f"wfast_{a.model}_{arm}_s{a.seed}.pt"))

        # (a) full history state (W_fast, P and Q as the history left them)
        pre, curve, gain = run_probe(model, state, d_train, d_eval, device,
                                     a.probe_steps, a.block, a.batch, **probe_kw)
        out["arms"][arm] = {"pre": pre, "curve": curve, "gain40": gain,
                            "final_ppl": curve[-1]["ppl"]}
        print(f"[scale] arm={arm} pre={pre:.2f} gain40={gain:+.6f} "
              f"final={curve[-1]['ppl']:.2f}", flush=True)
        del state
        model.set_dla_state(None)
        torch.cuda.empty_cache()

    # (b) W_fast-ISOLATED probe: a fresh state (P back at its prior, Q=0) carrying
    #     ONLY that history's W_fast.  This is the paper's carrier manipulation --
    #     it isolates W_fast from every other component the history touched, so a
    #     non-zero contrast cannot be attributed to P, Q or the Adam moments.
    for arm in ("EH", "HE"):
        model.set_dla_state(None)
        torch.cuda.empty_cache()
        st = model.make_state(device)
        inject_fast(st, hist_snap[arm])
        pre, curve, gain = run_probe(model, st, d_train, d_eval, device,
                                     a.probe_steps, a.block, a.batch, **probe_kw)
        out["arms"][arm + "_wfast"] = {"pre": pre, "curve": curve, "gain40": gain,
                                       "final_ppl": curve[-1]["ppl"]}
        print(f"[scale] arm={arm}_wfast pre={pre:.2f} gain40={gain:+.6f} "
              f"final={curve[-1]['ppl']:.2f}", flush=True)
        del st
        model.set_dla_state(None)
        torch.cuda.empty_cache()

    out["delta"] = out["arms"]["EH"]["gain40"] - out["arms"]["HE"]["gain40"]
    out["delta_wfast"] = (out["arms"]["EH_wfast"]["gain40"]
                          - out["arms"]["HE_wfast"]["gain40"])
    print(f"[scale] seed={a.seed} Delta(EH-HE)={out['delta']:+.4f} "
          f"Delta_wfast={out['delta_wfast']:+.4f}", flush=True)

    with open(os.path.join(a.out, f"scale_{a.model}_s{a.seed}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
