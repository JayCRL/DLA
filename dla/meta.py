"""Meta-learning machinery for DLA.

Two nested loops:

1. Inner loop (experience): the online DLA update
       X_{t+1} = L_phi(X_t, E_t)
   executed by ``DevelopmentalNet.step``.

2. Outer loop (meta-learning): phi is updated from the learning history
       phi_{t+1} = M(phi_t, H_t)
   implemented either as
     * meta-training across episodes (differentiable unrolling, classic
       learning-to-learn), or
     * lifetime meta-adaptation (periodic meta-gradient steps on a replay buffer
       of the individual's own recent experiences) - this is what makes the
       learning rule itself a developmental state.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .model import DevelopmentalNet


# --------------------------------------------------------------------- unrolling
def unroll_loss(
    net: DevelopmentalNet,
    state: Dict,
    batches: List[tuple[torch.Tensor, torch.Tensor]],
    reduce: str = "mean",
) -> tuple[torch.Tensor, Dict]:
    """Differentiable inner loop over a sequence of experience batches.

    Returns (meta_loss, final_state).  ``state`` is not modified in place; the
    initial W_fast/P/Q are broadcast into the graph as needed.
    """
    losses = []
    s = state
    for x, y in batches:
        s, info = net.step(x, y, s)
        losses.append(info["loss"])
    if reduce == "last":
        loss = losses[-1]
    else:
        loss = torch.stack(losses).mean()
    return loss, s


def _clip_param_grads(params, clip: float):
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        nn.utils.clip_grad_norm_(grads, clip)


def lifetime_unroll_loss(
    net: DevelopmentalNet,
    tasks,
    batch_size: int = 8,
    retain_weight: float = 1.0,
    task_seed: int = 0,
    n_retain_eval: int = 0,
):
    """Differentiable lifetime: learn tasks sequentially, sleep between them.

    The meta-loss is the mean of all online losses plus (optionally) retention
    losses evaluated with consolidated W_slow on the tasks seen so far.  This
    makes the stability-plasticity trade-off itself a meta-training objective:
    phi (including the consolidation parameters) is trained for lifetimes, not
    for isolated episodes.

    Returns (loss, final_slow_w, final_state).
    """
    if not tasks:
        raise ValueError("lifetime_unroll_loss needs at least one task")
    device = next(net.parameters()).device
    slow_w = [w for w in net.slow_w]  # reference to the persistent initial W_slow
    state = net.make_state(device)
    losses: List[torch.Tensor] = []

    for t_idx, task in enumerate(tasks):
        for b_idx, (x, y) in enumerate(task.train_batches(batch_size, seed=task_seed * 10007 + t_idx)):
            x, y = x.to(device), y.to(device)
            state, info = net.step(x, y, state, slow_w=slow_w)
            losses.append(info["loss"])

        # sleep: consolidate Q into functional W_slow, decay W_fast and Q
        slow_w, state = net.sleep_functional(slow_w, state)

        # retention / consolidation quality on every task seen so far
        if retain_weight > 0.0:
            slow_view = net.slow_state(state)
            for i in range(t_idx + 1):
                xt, yt = tasks[i].x_test, tasks[i].y_test
                if n_retain_eval and n_retain_eval < xt.shape[0]:
                    xt, yt = xt[:n_retain_eval], yt[:n_retain_eval]
                xt, yt = xt.to(device), yt.to(device)
                logits, _ = net.forward(xt, slow_view, slow_w=slow_w)
                losses.append(torch.nn.functional.cross_entropy(logits, yt) * retain_weight)

        # next task starts with a clean fast store, developmental states persist
        state = net.start_task_functional(state)

    return torch.stack(losses).mean(), slow_w, state


def meta_train(
    net: DevelopmentalNet,
    train_tasks,
    epochs: int,
    batch_size: int = 8,
    lr_rule: float = 3e-3,
    lr_slow: float = 1e-2,
    include_tempos: bool = True,
    grad_clip: float = 2.0,
    task_seed: int = 0,
    verbose: bool = True,
    lifetime_len: int = 1,
    retain_weight: float = 0.0,
    n_retain_eval: int = 0,
) -> List[float]:
    """Outer loop: train phi (learning rule) and the initial W_slow.

    * ``lifetime_len == 1``: classic learning-to-learn, one episode at a time.
    * ``lifetime_len >= 2``: tasks are grouped into developmental lifetimes and
      the meta-objective includes sleep consolidation + retention on earlier
      tasks (``retain_weight`` > 0 recommended).

    For ``rule_mode="hebbian"`` pass ``include_tempos=False`` so only W_slow is
    optimised: the fixed-Hebbian baseline still gets a meta-learned initialisation
    (a strong, fair baseline) but no learned rule.
    """
    param_groups = [{"params": net.slow_parameters(), "lr": lr_slow}]
    rule_params = []
    if net.rule is not None:
        rule_params += list(net.rule.parameters())
    if include_tempos:
        rule_params += list(net.tempos.parameters())
    if rule_params:
        param_groups.append({"params": rule_params, "lr": lr_rule})
    opt = torch.optim.Adam(param_groups)

    history = []
    rng = random.Random(task_seed)
    for epoch in range(epochs):
        tasks = list(train_tasks)
        rng.shuffle(tasks)
        epoch_losses = []
        if lifetime_len <= 1:
            for task in tasks:
                batches = list(task.train_batches(batch_size, seed=epoch * 10007 + task.n_train))
                state = net.make_state()
                opt.zero_grad(set_to_none=True)
                loss, _ = unroll_loss(net, state, batches)
                loss.backward()
                all_params = net.slow_parameters() + rule_params
                _clip_param_grads(all_params, grad_clip)
                opt.step()
                epoch_losses.append(loss.item())
        else:
            for i in range(0, len(tasks), lifetime_len):
                chunk = tasks[i : i + lifetime_len]
                if len(chunk) < 2:
                    continue
                opt.zero_grad(set_to_none=True)
                loss, _, _ = lifetime_unroll_loss(
                    net,
                    chunk,
                    batch_size=batch_size,
                    retain_weight=retain_weight,
                    task_seed=epoch * 10007 + i,
                    n_retain_eval=n_retain_eval,
                )
                loss.backward()
                all_params = net.slow_parameters() + rule_params
                _clip_param_grads(all_params, grad_clip)
                opt.step()
                epoch_losses.append(loss.item())
        hist = sum(epoch_losses) / max(1, len(epoch_losses))
        history.append(hist)
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1):
            print(f"    meta epoch {epoch + 1:3d}/{epochs}  mean loss {hist:.4f}", flush=True)
    return history


def meta_adapt_step(
    net: DevelopmentalNet,
    batches: List[tuple[torch.Tensor, torch.Tensor]],
    opt: torch.optim.Optimizer,
    grad_clip: float = 2.0,
):
    """One lifetime meta-adaptation step: phi <- phi - lr * grad_phi L(H_t).

    Only the learning-rule parameters receive gradients; W_slow is held fixed so
    the individual does not overwrite its consolidated knowledge while improving
    how it learns.
    """
    params = net.rule_parameters()
    if not params:
        return 0.0
    state = net.make_state()
    loss, _ = unroll_loss(net, state, batches)
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
    for p, g in zip(params, grads):
        p.grad = g.detach() if g is not None else None
    _clip_param_grads(params, grad_clip)
    opt.step()
    return loss.item()


class LifetimeAdapter:
    """phi_t adaptation inside a single lifetime.

    Keeps a bounded replay buffer H_t of recent experiences.  Every ``every``
    batches it samples a few mini-episodes from H_t and takes meta-gradient steps
    on phi (rule net + tempos).  This is the implementation of

        phi_{t+1} = phi_t + G(experience history).

    The frozen arm simply never instantiates this object.
    """

    def __init__(
        self,
        net: DevelopmentalNet,
        capacity: int = 128,
        unroll_len: int = 3,
        meta_batch: int = 8,
        lr: float = 1e-3,
        every: int = 2,
        min_items: int = 24,
        seed: int = 0,
    ):
        self.net = net
        self.capacity = capacity
        self.unroll_len = unroll_len
        self.meta_batch = meta_batch
        self.every = every
        self.min_items = min_items
        self.rng = random.Random(seed)
        self.buffer: deque = deque(maxlen=capacity)
        self.opt = torch.optim.Adam(net.rule_parameters(), lr=lr)
        self.counter = 0
        self.last_meta_loss = float("nan")

    def on_batch(self, x: torch.Tensor, y: torch.Tensor):
        # store individual experiences, not batches: the replay buffer H_t is a
        # set of (x, y) experiences
        self.buffer.extend((xi, yi) for xi, yi in zip(x.detach().cpu().unbind(0), y.detach().cpu().unbind(0)))
        self.counter += 1
        if self.counter % self.every == 0 and len(self.buffer) >= self.min_items:
            batches = []
            for _ in range(self.unroll_len):
                idx = [self.rng.randrange(len(self.buffer)) for _ in range(self.meta_batch)]
                xs = torch.stack([self.buffer[i][0] for i in idx]).to(next(self.net.parameters()).device)
                ys = torch.stack([self.buffer[i][1] for i in idx]).to(xs.device)
                batches.append((xs, ys))
            self.last_meta_loss = meta_adapt_step(self.net, batches, self.opt)
