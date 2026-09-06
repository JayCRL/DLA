"""Sanity tests for DLA v0.2.

Run on the training server:

    cd ~/llm-lab/dla-v0.2 && ~/llm-lab/venv/bin/python tests/test_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dla import (
    CoreConfig,
    DLAConfig,
    DevelopmentalNet,
    LifetimeAdapter,
    StaticMLP,
    make_task_stream,
    meta_train,
)
from dla.baselines import hebbian_dla_config
from dla.metrics import run_dla_stream, run_static_stream


def _small_cfg(mode="learned"):
    cfg = DLAConfig(
        core=CoreConfig(input_dim=6, hidden_dims=(8,), output_dim=2),
        rule_hidden=8,
        rule_mode=mode,
        eta_fast=0.25,
        eta_plast=0.05,
        fast_decay=0.05,
        plasticity_prior=0.5,
    )
    return cfg


def test_forward_and_shapes():
    cfg = _small_cfg()
    net = DevelopmentalNet(cfg)
    state = net.make_state()
    x = torch.randn(4, 6)
    logits, acts = net.forward(x, state)
    assert logits.shape == (4, 2)
    assert len(acts) == 2
    # expressed effective weight
    assert torch.allclose(net.slow_w[0] + torch.nn.functional.softplus(state["p"][0]) * state["w_fast"][0],
                          net.slow_w[0])
    print("test_forward_and_shapes OK")


def test_step_updates_all_three_levels():
    cfg = _small_cfg()
    net = DevelopmentalNet(cfg)
    state = net.make_state()
    x = torch.randn(8, 6)
    y = torch.randint(0, 2, (8,))
    wf_before = state["w_fast"][0].clone()
    p_before = state["p"][0].clone()
    q_before = state["q"][0].clone()
    state, info = net.step(x, y, state)
    assert state["w_fast"][0].shape == wf_before.shape
    assert not torch.equal(state["w_fast"][0], wf_before), "W_fast must change"
    assert not torch.equal(state["p"][0], p_before), "P must change (meta-plasticity)"
    assert not torch.equal(state["q"][0], q_before), "Q must change (eligibility trace)"
    for k in ("loss", "acc", "knowledge", "novelty", "uncertainty", "progress", "fatigue"):
        assert torch.isfinite(info[k])
    print("test_step_updates_all_three_levels OK")


def test_learning_improves_on_one_task():
    torch.manual_seed(0)
    cfg = _small_cfg()
    net = DevelopmentalNet(cfg)
    train_tasks, _ = make_task_stream(1, 1, input_dim=6, n_train=64, n_test=100)
    task = train_tasks[0].to("cpu")
    state = net.make_state()
    from dla.metrics import evaluate_dla

    acc0 = evaluate_dla(net, task.x_test, task.y_test, state)
    # a fresh (untrained) rule is not expected to be useful; meta-train it first
    meta_train(net, train_tasks, epochs=2, batch_size=8, lr_rule=5e-3, lr_slow=1e-2, verbose=False)
    state = net.make_state()
    for x, y in task.train_batches(8):
        state, _ = net.step(x, y, state)
    acc1 = evaluate_dla(net, task.x_test, task.y_test, state)
    print(f"    acc before {acc0:.3f} -> after meta-train+online {acc1:.3f}")
    assert acc1 > acc0 + 0.05, "meta-trained DLA online learning should lift accuracy"
    print("test_learning_improves_on_one_task OK")


def test_hebbian_mode_runs():
    cfg = _small_cfg(mode="hebbian")
    net = DevelopmentalNet(cfg)
    assert net.rule is None
    state = net.make_state()
    x = torch.randn(8, 6)
    y = torch.randint(0, 2, (8,))
    state, info = net.step(x, y, state)
    assert torch.isfinite(info["loss"])
    print("test_hebbian_mode_runs OK")


def test_meta_gradient_flows_into_phi():
    torch.manual_seed(1)
    cfg = _small_cfg()
    net = DevelopmentalNet(cfg)
    x = torch.randn(8, 6)
    y = torch.randint(0, 2, (8,))
    # run a 2-step unroll and backprop into phi and W_slow
    from dla.meta import unroll_loss

    state0 = net.make_state()
    loss, _ = unroll_loss(net, state0, [(x, y), (x, y)])
    loss.backward()
    rule_grads = [p.grad for p in net.rule.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in rule_grads), "phi must receive gradient"
    slow_grad = net.slow_w[0].grad
    assert slow_grad is not None and torch.isfinite(slow_grad).all(), "W_slow init must receive gradient"
    print("test_meta_gradient_flows_into_phi OK")


def test_meta_train_reduces_loss():
    torch.manual_seed(2)
    cfg = _small_cfg()
    net = DevelopmentalNet(cfg)
    train_tasks, _ = make_task_stream(2, 1, input_dim=6, n_train=32, n_test=60)
    history = meta_train(net, train_tasks, epochs=2, batch_size=8, lr_rule=3e-3, lr_slow=1e-2, verbose=False)
    assert history[0] > history[-1], "meta-training should reduce inner-loop loss"
    print(f"test_meta_train_reduces_loss OK  {history[0]:.3f} -> {history[-1]:.3f}")


def test_sleep_consolidates():
    cfg = _small_cfg()
    net = DevelopmentalNet(cfg)
    state = net.make_state()
    x = torch.randn(8, 6)
    y = torch.randint(0, 2, (8,))
    state, _ = net.step(x, y, state)
    slow_before = net.slow_w[0].detach().clone()
    wf_before = state["w_fast"][0].clone()
    net.sleep(state)
    assert not torch.equal(net.slow_w[0], slow_before), "sleep must consolidate into W_slow"
    assert state["w_fast"][0].norm() < wf_before.norm(), "sleep must decay W_fast"
    print("test_sleep_consolidates OK")


def test_lifetime_runner_and_static_baseline():
    torch.manual_seed(3)
    cfg = _small_cfg()
    net = DevelopmentalNet(cfg)
    train_tasks, eval_tasks = make_task_stream(2, 3, input_dim=6, n_train=32, n_test=60)
    meta_train(net, train_tasks, epochs=1, batch_size=8, verbose=False)
    rec = run_dla_stream(net, eval_tasks, batch_size=8)
    assert len(rec["post_acc"]) == 3
    assert len(rec["slow_matrix"]) == 3
    assert all(0 <= a <= 1 for a in rec["post_acc"])
    assert 0 <= rec["avg_post_acc"] <= 1

    static = StaticMLP.from_dla(net)
    rec_s = run_static_stream(static, eval_tasks, lr=0.1, batch_size=8)
    assert len(rec_s["post_acc"]) == 3
    print(f"test_lifetime_runner_and_static_baseline OK  dla={rec['avg_post_acc']:.3f} static={rec_s['avg_post_acc']:.3f}")


def test_lifetime_adapter_runs():
    torch.manual_seed(4)
    cfg = _small_cfg()
    net = DevelopmentalNet(cfg)
    _, eval_tasks = make_task_stream(0, 2, input_dim=6, n_train=24, n_test=40)
    adapter = LifetimeAdapter(net, capacity=64, unroll_len=2, meta_batch=6, lr=1e-3, every=1, min_items=8)
    rec = run_dla_stream(net, eval_tasks, batch_size=6, adapter=adapter)
    assert math.isfinite(rec["avg_post_acc"])
    assert adapter.last_meta_loss == adapter.last_meta_loss  # nan check via inequality below
    assert not (adapter.last_meta_loss != adapter.last_meta_loss), "meta adaptation should have run"
    print("test_lifetime_adapter_runs OK")


if __name__ == "__main__":
    import math

    torch.set_num_threads(4)
    for fn in [
        test_forward_and_shapes,
        test_step_updates_all_three_levels,
        test_learning_improves_on_one_task,
        test_hebbian_mode_runs,
        test_meta_gradient_flows_into_phi,
        test_meta_train_reduces_loss,
        test_sleep_consolidates,
        test_lifetime_runner_and_static_baseline,
        test_lifetime_adapter_runs,
    ]:
        fn()
    print("\nALL TESTS PASSED")
