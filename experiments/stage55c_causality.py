"""Stage 5.5c - Mechanism + Causality dissection of History -> Learner -> Future.

Three sub-experiments:

5.5c-A  H -> State
        saves per-stage state snapshots of meta individuals along easy->hard and
        hard->easy histories: phi trajectory, P statistics, W_fast/W_slow norms.

5.5c-B  State -> Future (counterfactual cross-injection)
        2x2 grid: Body (EH/HE) x phi (EH/HE), all probed on unseen D.
        Body = W_slow + P + W_fast + Q (plus Adam state reset after sleep);
        phi  = the 9 tempo parameters.

5.5c-C  phi sensitivity along the EH -> HE direction
        phi(alpha) = phi_EH + alpha * normalize(phi_HE - phi_EH)
        alpha in {-1,-0.5,0,0.5,1,1.5}, body fixed to EH.

Run (parallelise per seed if needed):
    ~/llm-lab/venv/bin/python experiments/stage55c_causality.py \
        --seeds 0,1,2 --cur-steps 40 --d-steps 40 --out results/stage55c
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NANO_DIR = os.path.expanduser("~/llm-lab/nanoGPT")
sys.path.insert(0, NANO_DIR)

_STAGE55 = ROOT / "experiments" / "stage55_learning_rule_development.py"
_spec = importlib.util.spec_from_file_location("stage55", _STAGE55)
_stage55 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stage55)

from dla.transformer_dla import DLAState, TransformerDLAConfig, make_dla_gpt_class  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

DLA_GPT = make_dla_gpt_class(GPT)
# tempo names: value -> (attr, kind)
_TEMPO_SPEC = [
    ("eta_fast", "log_eta_fast", "softplus"),
    ("eta_plast", "log_eta_plast", "softplus"),
    ("fast_decay", "logit_fast_decay", "sigmoid"),
    ("stability_pressure", "log_stability", "softplus"),
    ("alpha_q", "log_alpha_q", "softplus"),
    ("consolidate_beta", "logit_consolidate_beta", "sigmoid"),
    ("consolidate_fast_direct", "logit_consolidate_fast_direct", "sigmoid"),
    ("consolidate_fast_decay", "logit_consolidate_fast_decay", "sigmoid"),
    ("consolidate_q_decay", "logit_consolidate_q_decay", "sigmoid"),
]


def _logit(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def set_tempo_values(model, values):
    with torch.no_grad():
        for name, attr, kind in _TEMPO_SPEC:
            v = max(values[name], 1e-5)
            if kind == "softplus":
                t = torch.tensor(math.log(v))
            else:
                t = torch.tensor(_logit(v))
            getattr(model.tempos, attr).copy_(t)


def state_stats(model, state):
    gates, wf2, q2 = [], 0.0, 0.0
    for s in state.store.values():
        gates.append(torch.nn.functional.softplus(s["p"]).mean().item())
        wf2 += s["w_fast"].norm().item() ** 2
        q2 += s["q"].norm().item() ** 2
    slow2 = sum(mod.weight.detach().norm().item() ** 2 for mod in model.key_modules.values())
    return {
        "P_gate_mean": sum(gates) / len(gates),
        "W_fast_norm": math.sqrt(wf2),
        "Q_norm": math.sqrt(q2),
        "W_slow_norm": math.sqrt(slow2),
    }


def save_body(path, model, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store = {str(k): {kk: vv.detach().clone() for kk, vv in v.items()} for k, v in state.store.items()}
    torch.save({"model": model.state_dict(), "store": store,
                "ema": {k: v.detach().clone() for k, v in state.ema.items()},
                "config": model.config,
                "phi": model.tempos.snapshot()}, path)


def load_body(path, args, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = TransformerDLAConfig(eta_fast=args.eta_fast)
    if "config" in ckpt:
        model = DLA_GPT(ckpt["config"], cfg)
    else:
        model = DLA_GPT(GPTConfig(**ckpt["model_args"]), cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    store = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in ckpt["store"].items()}
    state = DLAState(store, {k: v.to(device) for k, v in ckpt["ema"].items()})
    return model, state


def run_meta_history(seed, order_name, phases, eb, d_train, args, device, body_dir):
    train_rng = random.Random(600000 + seed * 10 + (0 if order_name == "easy_hard" else 1))
    meta_rng = random.Random(610000 + seed * 10 + (0 if order_name == "easy_hard" else 1))
    cfg = TransformerDLAConfig(eta_fast=args.eta_fast)
    model = _stage55.load_checkpoint(DLA_GPT, cfg)
    model.train()
    state = model.make_state(device)
    meta_opt = torch.optim.Adam(model.tempos.parameters(), lr=args.meta_lr)
    trace = {"phi": [], "state": []}
    for t_idx, (name, train_ids, val_dom) in enumerate(phases):
        if t_idx > 0:
            before = model.tempos.snapshot()
            batches = [_stage55.get_batch(train_ids, args.block, args.meta_batch, meta_rng, device)
                       for _ in range(args.meta_unroll_len)]
            model.meta_update_phi(state, batches, meta_opt)
            after = model.tempos.snapshot()
            trace["phi"].append({"stage": t_idx, "before": before, "after": after,
                                 "norm": _stage55.tempo_delta_norm(before, after)})
        for step in range(1, args.cur_steps + 1):
            x, y = _stage55.get_batch(train_ids, args.block, args.batch, train_rng, device)
            model.dla_step(x, y, state)
        trace["state"].append({"stage": t_idx, "phi": model.tempos.snapshot(),
                               "state": state_stats(model, state)})
        model.dla_sleep(state)
    path = os.path.join(body_dir, f"seed{seed}_{order_name}_meta.pt")
    save_body(path, model, state)
    return path, trace


def run_d_probe(model, state, d_train, eb, args, device, seed, tag):
    rng = random.Random(620000 + seed + (0 if "EH" in tag else 1))
    args.max_steps = args.d_steps
    d = _stage55.run_transfer(model, d_train, eb, args, rng, device, state=state, dla=True)
    cs = _stage55.curve_stats(d["curve"], args.d_steps)
    return {"tag": tag, "LE_D": cs["LE"], "T80_D": cs["T80"],
            "best_gain_D": cs["best_gain"], "raw_lambda_D": _stage55.raw_lambda(d["curve"], args.target_gain)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--cur-steps", type=int, default=40)
    ap.add_argument("--d-steps", type=int, default=40)
    ap.add_argument("--target-gain", type=float, default=0.04)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--wiki-start", type=int, default=40_000_000)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--eta-fast", type=float, default=6e-4)
    ap.add_argument("--meta-lr", type=float, default=0.1)
    ap.add_argument("--meta-batch", type=int, default=4)
    ap.add_argument("--meta-unroll-len", type=int, default=2)
    ap.add_argument("--alpha-grid", type=str, default="-1,-0.5,0,0.5,1,1.5")
    ap.add_argument("--out", type=str, default="results/stage55c")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
    os.makedirs(args.out, exist_ok=True)
    body_dir = os.path.join(args.out, "bodies")
    os.makedirs(body_dir, exist_ok=True)
    torch.set_num_threads(3)
    device = "cpu"
    stoi = pickle.load(open(_stage55.META, "rb"))["stoi"]

    all_out = {"meta_trace": {}, "cross": {}, "sensitivity": {}}
    for seed in args.seeds:
        print(f"\n===== seed {seed}: curriculum histories =====", flush=True)
        domains, phases, d_train, d_val = _stage55.build_curriculum(seed, args, stoi)
        eb = _stage55.make_eval_batches(domains, d_val, args, device, seed)
        probe = _stage55.load_checkpoint(GPT)
        pre_all = {dom: _stage55.eval_ppl(probe, eb[dom]) for dom in domains}
        order = sorted(domains, key=lambda d: pre_all[d])
        fwd = sorted(phases, key=lambda ph: order.index(ph[2]))
        rev = list(reversed(fwd))
        for order_name, ph in (("easy_hard", fwd), ("hard_easy", rev)):
            t0 = time.time()
            path, trace = run_meta_history(seed, order_name, ph, eb, d_train, args, device, body_dir)
            all_out["meta_trace"][f"seed{seed}_{order_name}"] = {"body": path, "trace": trace}
            print(f"  {order_name} done {time.time()-t0:.0f}s -> {path}", flush=True)

        # -------- 5.5c-B cross injection
        print(f"  seed {seed}: 2x2 cross-injection ...", flush=True)
        body_paths = {o: os.path.join(body_dir, f"seed{seed}_{o}_meta.pt") for o in ("easy_hard", "hard_easy")}
        phis = {}
        for o, path in body_paths.items():
            m, _ = load_body(path, args, device)
            phis[o] = {k: v.clone() for k, v in m.tempos.state_dict().items()}
            del m
        for body_o in ("easy_hard", "hard_easy"):
            for phi_o in ("easy_hard", "hard_easy"):
                model, state = load_body(body_paths[body_o], args, device)
                model.tempos.load_state_dict(phis[phi_o])
                tag = f"Body_{body_o[:2].upper()}/phi_{phi_o[:2].upper()}"
                res = run_d_probe(model, state, d_train, eb, args, device, seed, tag)
                all_out["cross"].setdefault(f"seed{seed}", {})[tag] = res
                print(f"    {tag}: LE_D={res['LE_D']:.3f} T80={res['T80_D']}", flush=True)
                del model, state

        # -------- 5.5c-C phi sensitivity along EH->HE direction
        print(f"  seed {seed}: phi sensitivity ...", flush=True)
        eh_model, _ = load_body(body_paths["easy_hard"], args, device)
        he_model, _ = load_body(body_paths["hard_easy"], args, device)
        phi_eh = {k: float(v) for k, v in eh_model.tempos.snapshot().items()}
        phi_he = {k: float(v) for k, v in he_model.tempos.snapshot().items()}
        dvec = {k: phi_he[k] - phi_eh[k] for k in phi_eh}
        norm = math.sqrt(sum(v * v for v in dvec.values()))
        if norm > 0:
            for alpha in args.alpha_grid:
                model, state = load_body(body_paths["easy_hard"], args, device)  # fixed EH body
                target = {k: phi_eh[k] + alpha * dvec[k] / norm for k in phi_eh}
                set_tempo_values(model, target)
                res = run_d_probe(model, state, d_train, eb, args, device, seed, f"alpha={alpha}")
                all_out["sensitivity"].setdefault(f"seed{seed}", {})[str(alpha)] = res
                print(f"    alpha {alpha}: LE_D={res['LE_D']:.3f} T80={res['T80_D']}", flush=True)
                del model, state
        else:
            print("    EH==HE phi, sensitivity skipped", flush=True)
        del eh_model, he_model

    with open(os.path.join(args.out, "causality.json"), "w") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}/causality.json", flush=True)


if __name__ == "__main__":
    import pickle
    main()
