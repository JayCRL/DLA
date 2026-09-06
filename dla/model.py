"""DevelopmentalNet v0.2 - the core DLA model.

Levels of parameters kept by the system (see config.py):

    W_slow : long-term knowledge (initialised by DNA, consolidated during sleep,
             and optimised by the outer / meta learner)
    W_fast : fast weights - the current learning trace, updated per experience
    P      : per-connection plasticity - the system's ability to learn,
             itself updated per experience (meta-plasticity)
    Q      : slow-eligibility trace - which synapses are tagged for consolidation

One experience step computes

    basis_ij = a_heb_ij * (h_j x_i) + a_del_ij * (delta_j x_i)
    dW_fast_ij = eta_fast * softplus(P_ij) * m_ij * basis_ij - lambda * W_fast_ij
    dP_ij      = eta_plast * pd_ij * |h_j x_i + delta_j x_i| - kappa * (P_ij - P0_ij)
    dQ_ij      = alpha_q * qs_ij * softplus(P_ij) * m_ij * basis_ij * success

where [a_heb, a_del, m, pd, qs] are produced per connection by the Learning Rule
Network F_phi from local signals (x_i, h_j, delta_j, w_ij, p_ij) and
global/cognitive signals (error, novelty, uncertainty, reward, confidence, skill,
fatigue, progress, knowledge).  ``delta_j`` is the post-synaptic teaching signal
(dL/d post-activation); F_phi learns how much to trust the Hebbian basis versus
the teaching basis per connection, so the rule space contains pure Hebbian,
delta-rule and mixed rules.  In ``rule_mode="hebbian"`` the bracket is fixed to
[a_heb=1, a_del=0, m=1, pd=0, qs=0.5], i.e. a fixed Hebbian rule.

Sleep (task boundary) consolidates the eligibility trace into slow weights:

    W_slow += consolidate_beta * Q
    W_fast *= consolidate_fast_decay
    Q      *= consolidate_q_decay

All updates are written functionally (new tensors, no in-place parameter edits)
so the whole inner loop can be unrolled and differentiated for meta-learning.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DLAConfig, SIGNAL_NAMES


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _init_weight(w: torch.Tensor, gain: float = 0.7):
    nn.init.orthogonal_(w, gain=gain)


class TempoParams(nn.Module):
    """Learning-rule parameters (part of phi).

    Stored in log / logit space so they stay in valid ranges; all values are
    differentiable and can be meta-learned or kept fixed (DNA priors).
    """

    def __init__(self, cfg: DLAConfig):
        super().__init__()
        self.log_eta_fast = nn.Parameter(torch.tensor(math.log(max(cfg.eta_fast, 1e-4))))
        self.log_eta_plast = nn.Parameter(torch.tensor(math.log(max(cfg.eta_plast, 1e-4))))
        self.logit_fast_decay = nn.Parameter(torch.tensor(_logit(cfg.fast_decay)))
        self.log_stability = nn.Parameter(torch.tensor(math.log(max(cfg.stability_pressure, 1e-5))))
        self.log_alpha_q = nn.Parameter(torch.tensor(math.log(max(cfg.alpha_q, 1e-4))))
        self.logit_consolidate_beta = nn.Parameter(torch.tensor(_logit(cfg.consolidate_beta)))
        self.logit_consolidate_fast_decay = nn.Parameter(torch.tensor(_logit(cfg.consolidate_fast_decay)))
        self.logit_consolidate_q_decay = nn.Parameter(torch.tensor(_logit(cfg.consolidate_q_decay)))

    def values(self) -> Dict[str, torch.Tensor]:
        return {
            "eta_fast": F.softplus(self.log_eta_fast),
            "eta_plast": F.softplus(self.log_eta_plast),
            "fast_decay": torch.sigmoid(self.logit_fast_decay),
            "stability_pressure": F.softplus(self.log_stability),
            "alpha_q": F.softplus(self.log_alpha_q),
            "consolidate_beta": torch.sigmoid(self.logit_consolidate_beta),
            "consolidate_fast_decay": torch.sigmoid(self.logit_consolidate_fast_decay),
            "consolidate_q_decay": torch.sigmoid(self.logit_consolidate_q_decay),
        }


class _RuleNet(nn.Module):
    """F_phi: local + global signals -> per-connection rule outputs.

    Outputs (per connection):
        a_heb = tanh(z0)   : alignment on the Hebbian basis (h x^T)
        a_del = tanh(z1)   : alignment on the teaching basis (delta x^T)
        m     = sigmoid(z2): magnitude gate of the fast update
        pd    = tanh(z3)   : direction of the meta-plasticity update dP
        qs    = sigmoid(z4): consolidation-eligibility gate

    The update direction is therefore a per-connection learned mixture

        dW_fast ~ m * (a_heb * h x^T + a_del * delta x^T)

    so F_phi can discover anything between a pure Hebbian rule (a_del -> 0) and
    a delta-rule / backprop-like rule (a_del dominant) - the rule is learned,
    not hand-designed.  Initialised near the fixed-Hebbian corner.
    """

    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 5),
        )
        # initialise close to a fixed Hebbian rule: a_heb>0, a_del~0, m~1, pd~0, qs~0.5
        with torch.no_grad():
            self.net[0].weight.normal_(0.0, 0.4)
            self.net[0].bias.zero_()
            self.net[2].weight.normal_(0.0, 0.2)
            self.net[2].bias.copy_(torch.tensor([0.7, 0.0, 2.0, 0.0, 0.0]))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        z = self.net(f)  # (..., 5)
        return torch.stack(
            [
                torch.tanh(z[..., 0]),
                torch.tanh(z[..., 1]),
                torch.sigmoid(z[..., 2]),
                torch.tanh(z[..., 3]),
                torch.sigmoid(z[..., 4]),
            ],
            dim=-1,
        )


class DevelopmentalNet(nn.Module):
    def __init__(self, cfg: DLAConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.layer_dims = cfg.layer_dims()
        self.n_layers = len(self.layer_dims)
        self.p0 = cfg.p0

        # ---- W_slow / biases ------------------------------------------------
        self.slow_w = nn.ParameterList()
        self.slow_b = nn.ParameterList()
        for fan_in, fan_out in self.layer_dims:
            w = torch.empty(fan_out, fan_in)
            _init_weight(w)
            self.slow_w.append(nn.Parameter(w))
            self.slow_b.append(nn.Parameter(torch.zeros(fan_out)))

        # ---- Learning Rule Network F_phi ------------------------------------
        self.rule: Optional[_RuleNet] = _RuleNet(cfg.feat_dim, cfg.rule_hidden) if cfg.rule_mode == "learned" else None

        # ---- learning-rule parameters (eta, decay, consolidation, ...) ------
        self.tempos = TempoParams(cfg)

        # ---- fixed DNA priors on signal channels -----------------------------
        weights = [cfg.signal_weights[name] for name in SIGNAL_NAMES]
        self.register_buffer("signal_weights", torch.tensor(weights))

    # ------------------------------------------------------------------ state
    def make_state(self, device: torch.device | str | None = None) -> Dict:
        """Birth state: W_fast=0, Q=0, P=P0, cognitive EMAs at neutral values."""
        device = device or next(self.parameters()).device
        w_fast, p, q = [], [], []
        for fan_in, fan_out in self.layer_dims:
            w_fast.append(torch.zeros(fan_out, fan_in, device=device))
            p.append(torch.full((fan_out, fan_in), self.p0, device=device))
            q.append(torch.zeros(fan_out, fan_in, device=device))
        ema = {
            "loss_ema": torch.tensor(math.log(2.0), device=device),
            "error_ema": torch.tensor(0.5, device=device),
            "acc_ema": torch.tensor(0.5, device=device),
            "surprise_ema": torch.tensor(0.5, device=device),
            "progress_ema": torch.tensor(0.0, device=device),
        }
        return {"w_fast": w_fast, "p": p, "q": q, "ema": ema, "step": 0}

    def reset_fast(self, state: Dict):
        """Start a new task: keep W_slow, P and cognitive state; clear W_fast, Q."""
        for wf, qt in zip(state["w_fast"], state["q"]):
            wf.zero_()
            qt.zero_()
        state["step"] = 0
        return state

    def slow_state(self, state: Dict) -> Dict:
        """Evaluation view with only consolidated knowledge (W_fast = 0)."""
        w_fast = [torch.zeros_like(w) for w in state["w_fast"]]
        return {"w_fast": w_fast, "p": state["p"], "q": state["q"], "ema": state["ema"], "step": state["step"]}

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, state: Dict, slow_w=None, slow_b=None):
        """W_eff = W_slow + softplus(P) * W_fast.

        ``slow_w`` / ``slow_b`` are optional functional overrides used when a
        lifetime is unrolled through sleep consolidation (meta-training); by
        default the module's persistent W_slow is used.
        """
        ws_list = self.slow_w if slow_w is None else slow_w
        bs_list = self.slow_b if slow_b is None else slow_b
        acts: List[Dict] = []
        h = x
        for l in range(self.n_layers):
            ws, wf, p = ws_list[l], state["w_fast"][l], state["p"][l]
            w_eff = ws + F.softplus(p) * wf
            z = F.linear(h, w_eff, bs_list[l])
            acts.append({"pre": h, "z": z})
            if l == self.n_layers - 1:
                h = z
            elif self.cfg.core.activation == "tanh":
                h = torch.tanh(z)
            else:
                h = F.relu(z)
            acts[-1]["post"] = h
        return h, acts

    # ---------------------------------------------------------------- signals
    def _compute_signals(self, logits: torch.Tensor, y: torch.Tensor, ema: Dict):
        c = logits.shape[-1]
        probs = F.softmax(logits, dim=1)
        logp = torch.log(probs.clamp_min(1e-7))
        logp_true = logp[torch.arange(logits.shape[0], device=logits.device), y]
        p_true = torch.exp(logp_true)
        loss = -(logp_true).mean()

        error = 1.0 - p_true
        surprise = -logp_true / math.log(c)  # normalised negative log-likelihood
        entropy = -(probs * logp).sum(dim=1) / math.log(c)
        correct = (probs.argmax(dim=1) == y).float()
        reward = 2.0 * correct - 1.0  # +1 / -1

        a = self.cfg.cog_alpha
        new_ema = dict(ema)
        new_ema["loss_ema"] = (1 - a) * ema["loss_ema"] + a * loss
        new_ema["error_ema"] = (1 - a) * ema["error_ema"] + a * error.mean()
        new_ema["acc_ema"] = (1 - a) * ema["acc_ema"] + a * correct.mean()
        new_ema["surprise_ema"] = (1 - a) * ema["surprise_ema"] + a * surprise.mean()

        progress_raw = torch.tanh((ema["loss_ema"] - new_ema["loss_ema"]) / (ema["loss_ema"] + 1e-3))
        new_ema["progress_ema"] = (1 - a) * ema["progress_ema"] + a * progress_raw

        novelty = torch.tanh((surprise - ema["surprise_ema"]) / (ema["surprise_ema"] + 0.05))
        confidence = 1.0 - new_ema["error_ema"]
        skill = new_ema["acc_ema"]
        fatigue = torch.clamp(-new_ema["progress_ema"], 0.0, 1.0)
        progress = new_ema["progress_ema"]

        per_sample = {
            "error": error,
            "novelty": novelty,
            "uncertainty": entropy,
            "reward": reward,
        }
        scalars = {
            "loss": loss,
            "acc": correct.mean(),
            "confidence": confidence,
            "skill": skill,
            "fatigue": fatigue,
            "progress": progress,
        }
        return per_sample, scalars, new_ema

    # ------------------------------------------------------- one experience step
    def step(self, x: torch.Tensor, y: torch.Tensor, state: Dict, slow_w=None, slow_b=None):
        """One online learning step.  Returns (new_state, info).

        Functional and fully differentiable: meta-learning can backpropagate
        through a whole sequence of steps into phi (rule net + tempos) and W_slow.
        """
        ws_list = self.slow_w if slow_w is None else slow_w
        logits, acts = self.forward(x, state, slow_w=ws_list, slow_b=slow_b)
        per_sample, scalars, ema_new = self._compute_signals(logits, y, state["ema"])

        # ---- local teaching signals: dL/d(post_j) per layer, detached ------
        # F_phi sees delta as information and can mix it into the update through
        # the teaching basis (delta x^T).  This is what allows the learned rule
        # to discover delta-rule / backprop-like behaviour - or to suppress it
        # in favour of pure Hebbian learning.
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(y, logits.shape[-1]).float()
        deltas: List[torch.Tensor] = [None] * self.n_layers
        deltas[-1] = (probs - onehot).clamp(-5.0, 5.0)
        for l in range(self.n_layers - 2, -1, -1):
            w_next = ws_list[l + 1] + F.softplus(state["p"][l + 1]) * state["w_fast"][l + 1]
            d = deltas[l + 1] @ w_next.detach()
            if self.cfg.core.activation == "tanh":
                d = d * (1.0 - acts[l]["post"] ** 2)
            deltas[l] = d.clamp(-5.0, 5.0)
        # teaching signals are local observations for F_phi, not second-order
        # graph edges: the outer learner differentiates through the rule, not
        # through the gradient computation itself.
        deltas = [d.detach() for d in deltas]

        # knowledge = magnitude of currently expressed fast knowledge
        knowledge = 0.0
        for wf, p in zip(state["w_fast"], state["p"]):
            knowledge = knowledge + (F.softplus(p) * wf.abs()).mean()
        knowledge = torch.tanh(knowledge / self.n_layers)

        # global / cognitive channels (B, 9), shaped by fixed DNA priors
        b = x.shape[0]
        g = torch.stack(
            [
                per_sample["error"],
                per_sample["novelty"],
                per_sample["uncertainty"],
                per_sample["reward"],
                scalars["confidence"].expand(b),
                scalars["skill"].expand(b),
                scalars["fatigue"].expand(b),
                scalars["progress"].expand(b),
                knowledge.expand(b),
            ],
            dim=1,
        )
        g = g * self.signal_weights

        tv = self.tempos.values()
        success = (per_sample["reward"] + 1.0).mean() * 0.5  # fraction correct, scalar

        new_w_fast: List[torch.Tensor] = []
        new_p: List[torch.Tensor] = []
        new_q: List[torch.Tensor] = []

        for l in range(self.n_layers):
            pre = acts[l]["pre"]          # (B, fan_in)
            post = acts[l]["post"]        # (B, fan_out)
            delta = deltas[l]             # (B, fan_out)
            wf = state["w_fast"][l]       # (fan_out, fan_in)
            p = state["p"][l]
            q = state["q"][l]
            ws = ws_list[l]
            fan_in = pre.shape[-1]
            fan_out = post.shape[-1]

            # ---- local features for F_phi ----------------------------------
            loc = torch.stack(
                [
                    torch.tanh(2.0 * pre).unsqueeze(1).expand(b, fan_out, fan_in),
                    torch.tanh(2.0 * post).unsqueeze(2).expand(b, fan_out, fan_in),
                    torch.tanh(2.0 * delta.detach()).unsqueeze(2).expand(b, fan_out, fan_in),
                    torch.tanh(0.5 * wf.detach()).unsqueeze(0).expand(b, fan_out, fan_in),
                    torch.tanh(0.2 * p.detach()).unsqueeze(0).expand(b, fan_out, fan_in),
                    torch.tanh(0.5 * ws.detach()).unsqueeze(0).expand(b, fan_out, fan_in),
                ],
                dim=-1,
            )  # (B, out, in, 6)
            feat = torch.cat([loc, g.unsqueeze(1).unsqueeze(2).expand(b, fan_out, fan_in, g.shape[1])], dim=-1)

            if self.rule is not None:
                ro = self.rule(feat).mean(dim=0)  # (out, in, 5)
                a_heb, a_del, m, pd, qs = ro[..., 0], ro[..., 1], ro[..., 2], ro[..., 3], ro[..., 4]
            else:
                a_heb = torch.ones(fan_out, fan_in, device=x.device)
                a_del = torch.zeros_like(a_heb)
                m = torch.ones_like(a_heb)
                pd = torch.zeros_like(a_heb)
                qs = torch.full_like(a_heb, 0.5)

            gate = F.softplus(p)  # (out, in)
            hebb = (post.unsqueeze(2) * pre.unsqueeze(1)).mean(dim=0) * (fan_in ** -0.5)
            teach = (delta.unsqueeze(2) * pre.unsqueeze(1)).mean(dim=0) * (fan_in ** -0.5)

            # per-connection learned mixture of the two bases
            basis = a_heb * hebb + a_del * teach
            d_w_fast = tv["eta_fast"] * gate * m * basis - tv["fast_decay"] * wf
            d_p = tv["eta_plast"] * pd * (hebb.abs() + teach.abs()) - tv["stability_pressure"] * (p - self.p0)
            d_q = tv["alpha_q"] * qs * gate * m * basis * success

            new_w_fast.append(wf + d_w_fast)
            new_p.append(torch.clamp(p + d_p, self.cfg.p_min, self.cfg.p_max))
            new_q.append(q + d_q)

        new_state = {
            "w_fast": new_w_fast,
            "p": new_p,
            "q": new_q,
            "ema": ema_new,
            "step": state["step"] + 1,
        }
        info = {
            "loss": scalars["loss"],
            "acc": scalars["acc"],
            "knowledge": knowledge,
            "novelty": per_sample["novelty"].mean(),
            "uncertainty": per_sample["uncertainty"].mean(),
            "progress": scalars["progress"],
            "fatigue": scalars["fatigue"],
        }
        return new_state, info

    # ------------------------------------------------------------- sleep phase
    @torch.no_grad()
    def sleep(self, state: Dict):
        """Consolidate eligible fast learning into W_slow, then decay W_fast and Q."""
        tv = self.tempos.values()
        for l in range(self.n_layers):
            self.slow_w[l].add_(tv["consolidate_beta"] * state["q"][l])
            state["w_fast"][l].mul_(tv["consolidate_fast_decay"])
            state["q"][l].mul_(tv["consolidate_q_decay"])
        return state

    def sleep_functional(self, slow_w, state: Dict):
        """Differentiable version of ``sleep`` for lifetime unrolling.

        Returns (new_slow_w, new_state); no module parameter is modified in
        place, so the meta-learner can backpropagate from later tasks through
        the consolidation step (and therefore optimise the consolidation
        parameters and the rule that created the eligibility trace Q).
        """
        tv = self.tempos.values()
        new_slow_w = [ws + tv["consolidate_beta"] * state["q"][l] for l, ws in enumerate(slow_w)]
        new_w_fast = [wf * tv["consolidate_fast_decay"] for wf in state["w_fast"]]
        new_q = [q * tv["consolidate_q_decay"] for q in state["q"]]
        new_state = dict(state)
        new_state["w_fast"] = new_w_fast
        new_state["q"] = new_q
        return new_slow_w, new_state

    def start_task_functional(self, state: Dict):
        """Functional task boundary: keep W_slow, P and cognitive state; clear W_fast/Q."""
        new_state = dict(state)
        new_state["w_fast"] = [torch.zeros_like(wf) for wf in state["w_fast"]]
        new_state["q"] = [torch.zeros_like(q) for q in state["q"]]
        new_state["step"] = 0
        return new_state

    # ------------------------------------------------------------ introspection
    def plasticity_summary(self, state: Dict):
        """Mean expressed plasticity softplus(P) per layer."""
        return [F.softplus(p).mean().item() for p in state["p"]]

    def slow_parameters(self):
        return list(self.slow_w) + list(self.slow_b)

    def rule_parameters(self):
        params = []
        if self.rule is not None:
            params += list(self.rule.parameters())
        params += list(self.tempos.parameters())
        return params

    def copy_rule_from(self, other: "DevelopmentalNet"):
        """Share the learned DNA phi (rule net) with another individual."""
        if self.rule is not None and other.rule is not None:
            self.rule.load_state_dict(other.rule.state_dict())
        return self
