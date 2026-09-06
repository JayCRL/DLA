"""Lifetime runners and research metrics.

Metrics implemented here:

* per-task online test-accuracy curve (skill acquisition)
* post-task accuracy (with fast weights active)
* consolidated accuracy (W_slow only) - the knowledge memory
* forgetting / backward transfer from the T x T accuracy matrix
* steps-to-threshold - the sample efficiency of acquiring a new task, whose
  inverse is our operationalisation of the developmental variable Lambda_t
  (Learning Capacity)
* plasticity trajectory - mean softplus(P) over the lifetime
* slow-weight drift - a proxy for accumulated knowledge
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .baselines import StaticMLP
from .meta import LifetimeAdapter
from .model import DevelopmentalNet


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate_dla(net: DevelopmentalNet, x: torch.Tensor, y: torch.Tensor, state: Dict, slow_only: bool = False):
    s = net.slow_state(state) if slow_only else state
    logits, _ = net.forward(x, s)
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return acc


@torch.no_grad()
def evaluate_static(model: StaticMLP, x: torch.Tensor, y: torch.Tensor):
    logits = model(x)
    return (logits.argmax(dim=1) == y).float().mean().item()


def steps_to_threshold(curve: List[float], threshold: float = 0.85) -> int:
    """Number of training batches needed to reach ``threshold`` test accuracy."""
    for i, acc in enumerate(curve):
        if acc >= threshold:
            return i + 1
    return len(curve)


def forgetting_from_matrix(M: List[List[float]]) -> Dict[str, float]:
    """Forgetting = mean drop of each task's consolidated accuracy to end of life.

    ``M[k][i]`` is the accuracy of task i after task k has been learned
    (k >= i).  Positive forgetting is bad.
    """
    T = len(M)
    if T < 2:
        return {"forgetting": 0.0, "forgetting_pos": 0.0, "bwt": 0.0}
    drops = [M[i][i] - M[T - 1][i] for i in range(T - 1)]
    return {
        "forgetting": sum(drops) / len(drops),
        "forgetting_pos": sum(max(d, 0.0) for d in drops) / len(drops),
        "bwt": -sum(drops) / len(drops),  # backward transfer, positive = good
    }


# ------------------------------------------------------------- DLA lifetime
def run_dla_stream(
    net: DevelopmentalNet,
    tasks,
    batch_size: int = 8,
    threshold: float = 0.85,
    adapter: Optional[LifetimeAdapter] = None,
    test_every: int = 1,
) -> Dict:
    """Run one individual through one "life" of tasks. Slow weights are never reset."""
    net.eval()
    device = next(net.parameters()).device
    state = net.make_state(device)
    w0 = [w.detach().clone() for w in net.slow_w]

    curves: List[List[float]] = []
    post_acc: List[float] = []
    slow_matrix: List[List[float]] = []
    plasticity: List[float] = []
    knowledge_drift: List[float] = []
    info_trace: List[Dict[str, float]] = []

    for t_idx, task in enumerate(tasks):
        net.reset_fast(state)
        curve = []
        for b_idx, (x, y) in enumerate(task.train_batches(batch_size)):
            x, y = x.to(device), y.to(device)
            state, info = net.step(x, y, state)
            if adapter is not None:
                adapter.on_batch(x, y)
            info_trace.append({k: float(v.detach().cpu()) for k, v in info.items()})
            if (b_idx + 1) % test_every == 0 or b_idx == 0:
                curve.append(evaluate_dla(net, task.x_test, task.y_test, state))
        curves.append(curve)
        post_acc.append(curve[-1] if curve else 0.0)
        plasticity.append(sum(net.plasticity_summary(state)) / net.n_layers)

        # sleep: consolidate Q into W_slow, decay W_fast and Q
        net.sleep(state)

        # consolidated (slow-only) evaluation of every task seen so far
        row = [evaluate_dla(net, tasks[i].x_test, tasks[i].y_test, state, slow_only=True) for i in range(t_idx + 1)]
        slow_matrix.append(row)

        drift = 0.0
        for w, wi in zip(net.slow_w, w0):
            drift += ((w.detach() - wi).norm() / (wi.norm() + 1e-8)).item()
        knowledge_drift.append(drift / net.n_layers)

    slow_matrix = _ragged_to_square(slow_matrix)
    steps = [steps_to_threshold(c, threshold) for c in curves]
    forget = forgetting_from_matrix(slow_matrix)
    return {
        "curves": curves,
        "post_acc": post_acc,
        "slow_matrix": slow_matrix,
        "plasticity": plasticity,
        "knowledge_drift": knowledge_drift,
        "steps_to_threshold": steps,
        "info_trace": info_trace,
        "avg_post_acc": sum(post_acc) / len(post_acc),
        "avg_slow_final": sum(slow_matrix[-1]) / len(slow_matrix[-1]),
        "avg_steps": sum(steps) / len(steps),
        **forget,
    }


def _ragged_to_square(rows: List[List[float]]) -> List[List[float]]:
    T = len(rows)
    M = [[float("nan")] * T for _ in range(T)]
    for k, row in enumerate(rows):
        for i, v in enumerate(row):
            M[k][i] = v
    return M


# ------------------------------------------------------------ static lifetime
def run_static_stream(
    model: StaticMLP,
    tasks,
    lr: float = 0.1,
    batch_size: int = 8,
    threshold: float = 0.85,
) -> Dict:
    """Same life, but the only learning rule is external SGD on W_slow."""
    model.train()
    device = next(model.parameters()).device
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    w0 = [w.detach().clone() for w in model.parameters()]

    curves, post_acc, slow_matrix, knowledge_drift = [], [], [], []
    for t_idx, task in enumerate(tasks):
        curve = []
        for x, y in task.train_batches(batch_size):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            curve.append(evaluate_static(model, task.x_test, task.y_test))
        curves.append(curve)
        post_acc.append(curve[-1])
        row = [evaluate_static(model, tasks[i].x_test, tasks[i].y_test) for i in range(t_idx + 1)]
        slow_matrix.append(row)

        drift = 0.0
        for w, wi in zip(model.parameters(), w0):
            drift += ((w.detach() - wi).norm() / (wi.norm() + 1e-8)).item()
        knowledge_drift.append(drift / len(w0))

    slow_matrix = _ragged_to_square(slow_matrix)
    steps = [steps_to_threshold(c, threshold) for c in curves]
    forget = forgetting_from_matrix(slow_matrix)
    return {
        "curves": curves,
        "post_acc": post_acc,
        "slow_matrix": slow_matrix,
        "knowledge_drift": knowledge_drift,
        "steps_to_threshold": steps,
        "avg_post_acc": sum(post_acc) / len(post_acc),
        "avg_slow_final": sum(slow_matrix[-1]) / len(slow_matrix[-1]),
        "avg_steps": sum(steps) / len(steps),
        **forget,
    }


# ------------------------------------------------------------------ summaries
def mean_curve(curves: List[List[float]], max_len: int | None = None) -> List[float]:
    n = max_len or max(len(c) for c in curves)
    out = []
    for i in range(n):
        vals = [c[i] for c in curves if i < len(c)]
        out.append(sum(vals) / len(vals))
    return out


def summarize_records(records: List[Dict]) -> Dict:
    """Mean +/- std over seeds for the scalar research metrics."""
    keys = ["avg_post_acc", "avg_slow_final", "avg_steps", "forgetting", "forgetting_pos", "bwt"]
    out = {}
    for k in keys:
        vals = [r[k] for r in records]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals))
        out[k] = {"mean": mu, "std": math.sqrt(var)}
    out["mean_curve"] = mean_curve([r["curves"][i] for r in records for i in range(len(r["curves"]))])
    return out


def sanitize_for_json(obj):
    """Recursively replace non-finite floats / tensors with JSON-safe values."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj
