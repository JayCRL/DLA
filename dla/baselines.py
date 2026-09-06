"""Baselines used by the staged experiments.

* ``StaticMLP`` - the same Neural Core but with a traditional external optimizer
  (SGD).  This is the "learning rule is outside the model" control.
* Fixed-Hebbian DLA - implemented inside ``DevelopmentalNet`` with
  ``rule_mode="hebbian"`` and frozen tempos; its meta-training (see ``meta.py``)
  only optimises the initial W_slow, so it is a fair, strong control.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoreConfig, DLAConfig
from .model import DevelopmentalNet


class StaticMLP(nn.Module):
    """Plain MLP trained with SGD. No fast weights, no plasticity, no cognitive state."""

    def __init__(self, core_cfg: CoreConfig):
        super().__init__()
        dims = [core_cfg.input_dim, *core_cfg.hidden_dims, core_cfg.output_dim]
        self.core_cfg = core_cfg
        self.layers = nn.ModuleList()
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            layer = nn.Linear(fan_in, fan_out)
            # same birth initialisation as DevelopmentalNet's W_slow
            nn.init.orthogonal_(layer.weight, gain=0.7)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)

    @classmethod
    def from_dla(cls, net: DevelopmentalNet):
        """Clone W_slow/bias initialisation from a DLA individual (same DNA start)."""
        m = cls(net.cfg.core)
        with torch.no_grad():
            for src, dst in zip(net.slow_w, [layer.weight for layer in m.layers]):
                dst.copy_(src)
            for src, dst in zip(net.slow_b, [layer.bias for layer in m.layers]):
                dst.copy_(src)
        return m

    def forward(self, x: torch.Tensor):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = torch.tanh(h) if self.core_cfg.activation == "tanh" else F.relu(h)
        return h


def train_static_one_batch(model: StaticMLP, x: torch.Tensor, y: torch.Tensor, opt: torch.optim.Optimizer):
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    opt.step()
    return loss.item(), (logits.argmax(dim=1) == y).float().mean().item()


def make_static_optimizer(model: StaticMLP, lr: float):
    return torch.optim.SGD(model.parameters(), lr=lr)


def hebbian_dla_config(cfg: DLAConfig) -> DLAConfig:
    """A copy of ``cfg`` with the fixed-Hebbian rule and frozen DNA tempos."""
    import dataclasses

    cfg = dataclasses.replace(cfg)
    cfg.rule_mode = "hebbian"
    cfg.learn_tempos = False
    return cfg
