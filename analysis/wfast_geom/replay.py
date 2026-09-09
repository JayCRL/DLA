"""P1 Phase B - deterministic instrumented replay of a Stage 5.5e D-probe arm.

Re-executes the exact stored D-probe protocol for one (seed, arm) from the
saved EH/HE bodies (same data, same RNG seeds, same DLA step math) and records
per-step gradient-trajectory summaries WITHOUT storing full tensors:

  gn[t]   = ||grad wrt W_eff||      per group (emb/attn/mlp/global)
  gfn[t]  = ||softplus(p)*grad||    = grad wrt W_fast
  cos_d   = cosine(g_f, ΔW_fast_hat) where ΔW_fast = W_fast(HE) - W_fast(EH)
            of the same seed (the history-induced contrast direction)
  proj ratio = |cos_d| (projection of unit gradient onto Δ direction)

The stored p0 JSON curve is re-evaluated for a parity check (max |Δppl|) so any
divergence from the original runs is reported, not hidden.

Run (one job per seed/arm, orchestrate in parallel):
  python analysis/wfast_geom/replay.py --seed 3 --arm EH/EH --out results/analysis_wfast/replays
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
import torch.nn.functional as F

from common import (Args, arm_tags, build_D, load_body_ckpt, slug,
                    all_seeds, canonical_keys, group_keys, group_of, GROUPS,
                    norm2_blockwise)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import stage55_learning_rule_development as s55  # noqa: E402
import stage55c_causality as s55c  # noqa: E402


def build_delta_matrices(seed):
    """ΔW_fast = W_fast(HE) - W_fast(EH) per key (tensors) + per-group norms."""
    eh = load_body_ckpt(seed, "easy_hard")["store"]
    he = load_body_ckpt(seed, "hard_easy")["store"]
    keys = canonical_keys(eh.keys())
    delta = {k: (he[k]["w_fast"] - eh[k]["w_fast"]).float() for k in keys}
    gk = group_keys(eh.keys())
    norm2 = {g: sum(float(delta[k].pow(2).sum()) for k in gk[g]) for g in GROUPS}
    delta_norm2 = {g: norm2[g] for g in GROUPS}
    return delta, delta_norm2, keys


def instrumented_dla_step(model, x, y, state, delta, delta_norm2):
    """Byte-for-byte re-implementation of DLA.dla_step with gradient capture."""
    cfg = model.dla_cfg
    model.set_dla_state(state)
    model.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    # ---- capture (post-clip, i.e. exactly the g used by the updates) --------
    gk = group_keys(state.store.keys())
    acc = {g: {"g2": 0.0, "gf2": 0.0, "dot": 0.0} for g in list(GROUPS) + ["global"]}
    for key, mod in model.key_modules.items():
        g = mod.weight.grad
        if g is None:
            continue
        gate = F.softplus(state.store[key]["p"])
        gf = gate * g
        grp = group_of(key)
        g2 = float(g.pow(2).sum())
        gf2 = float(gf.pow(2).sum())
        acc[grp]["g2"] += g2
        acc[grp]["gf2"] += gf2
        acc["global"]["g2"] += g2
        acc["global"]["gf2"] += gf2
        if delta is not None and key in delta:
            d = float((gf * delta[key]).sum())
            acc[grp]["dot"] += d
            acc["global"]["dot"] += d

    caps = {}
    for grp in list(GROUPS) + ["global"]:
        gfn = math.sqrt(acc[grp]["gf2"])
        caps[grp] = {"gn": math.sqrt(acc[grp]["g2"]), "gfn": gfn}
    caps["cos_d"] = {}
    for grp in GROUPS:
        dnorm = math.sqrt(delta_norm2[grp]) if delta is not None else 0.0
        if delta is not None and acc[grp]["gf2"] > 0 and dnorm > 0:
            caps["cos_d"][grp] = acc[grp]["dot"] / (math.sqrt(acc[grp]["gf2"]) * dnorm)
        else:
            caps["cos_d"][grp] = None
    if delta is not None and acc["global"]["gf2"] > 0:
        dnorm_g = math.sqrt(sum(delta_norm2.values()))
        caps["cos_d"]["global"] = acc["global"]["dot"] / (math.sqrt(acc["global"]["gf2"]) * dnorm_g)
    else:
        caps["cos_d"]["global"] = None

    # ---- the exact DLA update (mirrors dla_step) ----------------------------
    with torch.no_grad():
        tv = model.tempos.values()
        mask = y != -1
        acc_m = ((logits.argmax(dim=-1)[mask] == y[mask]).float().mean()).item()
        loss_v = loss.item()
        a = cfg.cog_alpha
        prev_loss = state.ema["loss_ema"]
        new_loss_ema = (1 - a) * prev_loss + a * loss_v
        new_acc_ema = (1 - a) * state.ema["acc_ema"] + a * acc_m
        progress = math.tanh((prev_loss.item() - new_loss_ema.item()) / (prev_loss.item() + 1e-4))
        new_progress_ema = (1 - a) * state.ema["progress_ema"] + a * progress
        success = max(0.0, min(1.0, (prev_loss.item() - loss_v) / (prev_loss.item() + 1e-4)))
        for key, mod in model.key_modules.items():
            s = state.store[key]
            g = mod.weight.grad
            if g is None:
                continue
            gate = F.softplus(s["p"])
            t = state.step + 1
            b1, b2 = cfg.beta1, cfg.beta2
            s["m"].mul_(b1).add_(g, alpha=1 - b1)
            s["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = s["m"] / (1 - b1 ** t)
            v_hat = s["v"] / (1 - b2 ** t)
            adam = m_hat / (v_hat.sqrt() + 1e-8)
            dw = -tv["eta_fast"] * gate * adam - tv["fast_decay"] * s["w_fast"]
            s["w_fast"].add_(dw)
            n = g.numel()
            relevance = (g.abs() / (g.norm() / math.sqrt(n) + 1e-6)).clamp(0.0, 5.0)
            dp = tv["eta_plast"] * progress * relevance - tv["stability_pressure"] * (s["p"] - cfg.p0)
            s["p"].add_(dp).clamp_(cfg.p_min, cfg.p_max)
            q_inc = dw * success
            s["q"].mul_(1 - tv["alpha_q"]).add_(q_inc, alpha=tv["alpha_q"])
        state.ema["loss_ema"] = new_loss_ema
        state.ema["acc_ema"] = new_acc_ema
        state.ema["progress_ema"] = new_progress_ema
        state.step += 1
        state.life_step += 1
    return {"loss": loss_v, "acc": acc_m, "progress": progress, "success": success,
            "loss_ema": new_loss_ema.item(), "life_step": state.life_step}, caps


COMBO = {
    "EH/EH": ("easy_hard", None),
    "HE/HE": ("hard_easy", None),
    "EH_body+HE_fast": ("easy_hard", "hard_easy"),
    "HE_body+EH_fast": ("hard_easy", "easy_hard"),
}


def copy_wfast(src_state, dst_state):
    for key, ss in src_state.store.items():
        if key in dst_state.store:
            dst_state.store[key]["w_fast"].copy_(ss["w_fast"])


def run_one(seed, tag, out_dir, args):
    base_o, src_o = COMBO[tag]
    torch.set_num_threads(2)
    device = "cpu"
    stoi = _stoi()
    domains, _phases, d_train, d_val = s55.build_curriculum(seed, args, stoi)
    eb = s55.make_eval_batches(domains, d_val, args, device, seed)
    delta, delta_norm2, _keys = build_delta_matrices(seed)

    model, state = s55c.load_body(str(_body_path(seed, base_o)), args, device)
    if src_o is not None:
        _sm, src_state = s55c.load_body(str(_body_path(seed, src_o)), args, device)
        copy_wfast(src_state, state)
    model.train()
    rng = random.Random(800000 + seed)

    pre = s55.eval_ppl(model, eb["D"], state=state)
    curve = []
    stepcaps = []
    for step in range(1, args.d_steps + 1):
        x, y = s55.get_batch(d_train, args.block, args.batch, rng, device)
        info, caps = instrumented_dla_step(model, x, y, state, delta, delta_norm2)
        caps["step"] = step
        stepcaps.append(caps)
        if step % args.eval_every == 0 or step == args.d_steps:
            ppl = s55.eval_ppl(model, eb["D"], state=state)
            curve.append({"step": step, "ppl": ppl, "gain": (pre - ppl) / pre})

    # parity vs archived p0 json (ppl curve at same steps)
    parity = None
    p0p = ROOT / "results" / "stage55e" / "seeds" / f"seed{seed}.json"
    if p0p.exists():
        arch = json.load(open(p0p))["results"][tag]["curve"]
        if len(arch) == len(curve):
            diffs = [abs(a["ppl"] - b["ppl"]) for a, b in zip(arch, curve)]
            parity = {"max_ppl_diff": max(diffs), "mean_ppl_diff": sum(diffs) / len(diffs)}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"seed{seed}_{slug(tag)}.json"), "w") as f:
        json.dump({"seed": seed, "tag": tag, "parity": parity, "pre": pre,
                   "curve": curve, "steps": stepcaps}, f)
    return pre, curve[-1]["gain"] if curve else None, parity


def _body_path(seed, order):
    c1 = ROOT / "results" / "stage55c" / f"seed{seed}" / "bodies" / f"seed{seed}_{order}_meta.pt"
    c2 = ROOT / "results" / "stage55e" / "bodies" / f"seed{seed}_{order}_meta.pt"
    return c1 if c1.exists() else c2


_stoi_cache = None


def _stoi():
    global _stoi_cache
    if _stoi_cache is None:
        import pickle
        _stoi_cache = pickle.load(open(s55.META, "rb"))["stoi"]
    return _stoi_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", type=str, required=True, choices=list(COMBO.keys()))
    ap.add_argument("--out", default="results/analysis_wfast/replays")
    args = ap.parse_args()
    # fix stray helper
    a = Args()
    pre, gain, parity = run_one(args.seed, args.arm, os.path.join(args.out, "seeds"), a)
    msg = f"seed {args.seed} {args.arm}: pre={pre:.2f} gain40={gain:.4f}"
    if parity:
        msg += f" parity_ppl_diff={parity['max_ppl_diff']:.4f}"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
