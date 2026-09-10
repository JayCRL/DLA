"""P2: continual-learning baselines vs DLA on a unified 10-task sequence.

Question
--------
DLA's selective consolidation is only interesting if it buys something a standard
continual-learning method does not.  This runs the SAME 10-task sequence through

    dla     : frozen backbone + fast weights + sleep consolidation (the paper's rule)
    adamw   : plain fine-tuning of the backbone
    ewc     : fine-tuning + online EWC penalty
    si      : fine-tuning + online synaptic-intelligence penalty
    replay  : fine-tuning + a replay buffer of earlier tasks

and reports, per method:

    forward    : mean relative perplexity drop on each task's OWN held-out slice
                 (higher is better -- can it still learn?)
    retention  : mean(after / end) over tasks
                 (higher is better -- did it keep them?)
    forgetting : mean relative perplexity RISE from just-after-learning to the end
                 (lower is better)

Sequence: 10 categories of 20 Newsgroups, GPT-2 BPE.  Distinct English domains, so
the backbone is in-distribution and "learning a task" is meaningful.

Memory note: EWC and SI are the ONLINE variants (one running anchor + one accumulated
penalty weight).  Storing 10 separate anchors at 1.5B would cost 10x the model, which
does not fit in 48 GB.  This is the standard bounded-memory formulation.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import audit_scale as S  # noqa: E402

RAW = os.path.expanduser("~/llm-lab/tasks/20ng.jsonl")
CACHE = os.path.expanduser("~/llm-lab/tasks/ng10")

# A spread of unrelated domains: science, hardware, politics, sport, cars, religion.
TASK_ORDER = [
    "sci.space", "comp.graphics", "talk.politics.misc", "rec.autos",
    "sci.med", "comp.sys.mac.hardware", "talk.politics.guns",
    "rec.sport.baseball", "soc.religion.christian", "misc.forsale",
]


def build_tasks(tokens_per_task=120_000, eval_tokens=20_000):
    """Tokenise the 10 categories once and cache as .npy."""
    import tiktoken
    os.makedirs(CACHE, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    rows = None
    tasks = []
    for name in TASK_ORDER:
        tr_p = os.path.join(CACHE, f"{name}.train.npy")
        ev_p = os.path.join(CACHE, f"{name}.eval.npy")
        if not (os.path.exists(tr_p) and os.path.exists(ev_p)):
            if rows is None:
                rows = [json.loads(l) for l in open(RAW)]
            texts = [r["text"] for r in rows if r["label_text"] == name]
            if not texts:
                raise SystemExit(f"no rows for {name}")
            buf = " \n ".join(texts)
            out = []
            for i in range(0, len(buf), 100_000):
                out.extend(enc.encode_ordinary(buf[i:i + 100_000]))
            arr = np.array(out, dtype=np.int32)
            need = tokens_per_task + eval_tokens
            if len(arr) < need:
                raise SystemExit(f"{name}: only {len(arr)} tokens, need {need}")
            np.save(tr_p, arr[:tokens_per_task])
            np.save(ev_p, arr[tokens_per_task:need])
        tasks.append((name,
                      torch.tensor(np.load(tr_p).astype(np.int64)),
                      torch.tensor(np.load(ev_p).astype(np.int64))))
    return tasks


def batches(ids, block, batch, rng, n, device):
    for _ in range(n):
        ix = torch.randint(len(ids) - block - 1, (batch,), generator=rng)
        x = torch.stack([ids[i:i + block] for i in ix]).to(device)
        y = torch.stack([ids[i + 1:i + 1 + block] for i in ix]).to(device)
        yield x, y


@torch.no_grad()
def eval_ppl(model, state, ids, block, batch, nb, seed, device):
    rng = torch.Generator().manual_seed(seed)
    was = model.training
    model.eval()
    model.set_dla_state(state)
    tot, cnt = 0.0, 0
    for x, y in batches(ids, block, batch, rng, nb, device):
        _, loss = model(x, y)
        tot += float(loss) * y.numel()
        cnt += y.numel()
    if was:
        model.train()
    return math.exp(tot / max(cnt, 1))


class Trainer:
    """One method's training loop over a task sequence."""

    def __init__(self, model, method, args, device):
        self.model, self.method, self.args, self.device = model, method, args, device
        self.params = [p for n, p in model.named_parameters()
                       if not n.startswith("tempos.")]
        self.names = [n for n, _ in model.named_parameters()
                      if not n.startswith("tempos.")]
        # DLA needs backbone grads (the wake rule differentiates through them to
        # drive Adam on W_fast) but must NOT step the backbone itself.
        for p in model.parameters():
            p.requires_grad_(True)
        self.state = model.make_state(device) if method == "dla" else None
        self.opt = (torch.optim.AdamW(self.params, lr=args.lr, weight_decay=0.0)
                    if method != "dla" else None)
        self.anchor, self.fisher, self.omega = None, None, None
        self.replay = None

    def step(self, x, y):
        if self.method == "dla":
            self.model.set_dla_state(self.state)
            self.model.dla_step(x, y, self.state)
            return
        self.opt.zero_grad(set_to_none=True)
        self.model.set_dla_state(None)
        _, loss = self.model(x, y)
        if self.method == "ewc" and self.fisher is not None:
            pen = sum((self.fisher[n] * (p.detach().float() - self.anchor[n].float()) ** 2).sum()
                      for n, p in self.model.named_parameters() if n in self.fisher)
            loss = loss + self.args.ewc_lambda * pen
        elif self.method == "si" and self.omega is not None:
            pen = sum((self.omega[n] * (p.detach().float() - self.anchor[n].float()) ** 2).sum()
                      for n, p in self.model.named_parameters() if n in self.omega)
            loss = loss + self.args.si_lambda * pen
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()

    def snapshot(self):
        return {n: p.detach().clone().to(torch.bfloat16)
                for n, p in self.model.named_parameters() if not n.startswith("tempos.")}

    def end_task(self, tr_ids, rng):
        """Bookkeeping a method needs after finishing a task."""
        if self.method == "dla":
            self.model.dla_sleep(self.state)
        elif self.method in ("ewc", "si"):
            new = self.snapshot()
            if self.anchor is None:
                self.anchor = new
                if self.method == "ewc":
                    self.fisher = {n: torch.zeros_like(t, dtype=torch.float32)
                                   for n, t in new.items()}
                else:
                    self.omega = {n: torch.zeros_like(t, dtype=torch.float32)
                                  for n, t in new.items()}
            else:
                if self.method == "ewc":
                    fr = {n: torch.zeros_like(t, dtype=torch.float32) for n, t in new.items()}
                    for x, y in batches(tr_ids, self.args.block, self.args.batch, rng, 8,
                                        self.device):
                        self.model.zero_grad(set_to_none=True)
                        _, ls = self.model(x, y)
                        ls.backward()
                        for n, p in self.model.named_parameters():
                            if n in fr and p.grad is not None:
                                fr[n] += p.grad.detach().float() ** 2
                    for n in fr:
                        self.fisher[n] = 0.5 * self.fisher[n] + 0.5 * fr[n] / 8.0
                else:  # si: accumulate |dtheta| as the importance proxy
                    for n, p in self.model.named_parameters():
                        if n in self.omega:
                            self.omega[n] += (p.detach().float() - self.anchor[n].float()).abs()
                self.anchor = new
        elif self.method == "replay":
            per = max(1, self.args.replay_size // len(TASK_ORDER))
            add = tr_ids[:per].clone()
            self.replay = add if self.replay is None \
                else torch.cat([self.replay, add])[-self.args.replay_size:]

    def maybe_replay(self, x, y, k, t, rng):
        if self.method != "replay" or self.replay is None or k % 2 == 0:
            return x, y
        if len(self.replay) <= self.args.block + 2:
            return x, y
        ix = torch.randint(len(self.replay) - self.args.block - 1,
                           (self.args.batch,), generator=rng)
        return (torch.stack([self.replay[i:i + self.args.block] for i in ix]).to(self.device),
                torch.stack([self.replay[i + 1:i + 1 + self.args.block] for i in ix]).to(self.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="gpt2", choices=list(S.MODELS))
    ap.add_argument("--methods", default="dla,adamw,ewc,si,replay")
    ap.add_argument("--steps", type=int, default=60, help="training steps per task")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ewc-lambda", type=float, default=100.0)
    ap.add_argument("--si-lambda", type=float, default=1.0)
    ap.add_argument("--replay-size", type=int, default=20_000)
    ap.add_argument("--tokens-per-task", type=int, default=120_000)
    ap.add_argument("--eval-tokens", type=int, default=20_000)
    ap.add_argument("--state-dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.expanduser("~/dla_cl"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = S.build(a.model, a.state_dtype, a.device)
    n_par = sum(p.numel() for p in model.parameters())
    tasks = build_tasks(a.tokens_per_task, a.eval_tokens)
    print(f"[cl] backbone={a.model} params={n_par/1e6:.1f}M tasks={len(tasks)} "
          f"steps/task={a.steps}", flush=True)

    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = {"seed": a.seed, "model": a.model, "steps_per_task": a.steps,
           "note": "EWC/SI use online (single running anchor) variants", "methods": {}}

    for method in [m.strip() for m in a.methods.split(",") if m.strip()]:
        model.load_state_dict(base_sd, strict=False)
        model.set_dla_state(None)
        torch.cuda.empty_cache()
        tr = Trainer(model, method, a, a.device)

        after, forward = [], []
        for t, (name, tr_ids, ev_ids) in enumerate(tasks):
            rng = torch.Generator().manual_seed(90000 + a.seed * 100 + t)
            pre = eval_ppl(model, tr.state, ev_ids, a.block, a.batch, a.eval_batches,
                           700000 + t, a.device)
            for k, (x, y) in enumerate(batches(tr_ids, a.block, a.batch, rng, a.steps,
                                               a.device)):
                x, y = tr.maybe_replay(x, y, k, t, rng)
                tr.step(x, y)
            tr.end_task(tr_ids, rng)
            post = eval_ppl(model, tr.state, ev_ids, a.block, a.batch, a.eval_batches,
                            700000 + t, a.device)
            after.append(post)
            forward.append((pre - post) / pre)
            print(f"[cl] {method:7s} task{t+1:2d} {name:22s} pre={pre:8.2f} "
                  f"post={post:8.2f} gain={forward[-1]:+.4f}", flush=True)

        end = [eval_ppl(model, tr.state, ev_ids, a.block, a.batch, a.eval_batches,
                        700000 + t, a.device)
               for t, (_, _, ev_ids) in enumerate(tasks)]
        forget = [(e - p) / p for e, p in zip(end, after)]
        m = {"forward": forward, "after": after, "end": end, "forgetting": forget,
             "forward_mean": float(np.mean(forward)),
             "retention_mean": float(np.mean([p / e for p, e in zip(after, end)])),
             "forgetting_mean": float(np.mean(forget))}
        out["methods"][method] = m
        print(f"[cl] === {method:7s} forward={m['forward_mean']:+.4f} "
              f"retention={m['retention_mean']:.4f} forgetting={m['forgetting_mean']:+.4f}",
              flush=True)
        del tr
        model.set_dla_state(None)
        torch.cuda.empty_cache()

    p = os.path.join(a.out, f"cl_{a.model}_s{a.seed}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[cl] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
