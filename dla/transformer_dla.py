"""DLA-Transformer (Stage 4): the developmental mechanism mounted on a standard GPT.

Design constraints (per the 2026-09-07 decision):
* the Transformer skeleton is NOT modified - every wrapped module still performs
  the original computation, with an effective weight

      W_eff = W_slow + softplus(P) * W_fast

* the developmental machinery lives in per-parameter states:

      W_fast : fast weights (current learning trace)
      P      : per-parameter plasticity (ability to learn, itself developmental)
      Q      : slow-eligibility trace (what gets consolidated during sleep)
      m, v   : Adam moments of the fast-weight update

* one experience step (wake):

      g      = dL/dW_eff            (standard backprop teaching basis, clipped)
      adam   = Adam(g; m, v)
      dW_fast = -eta_fast * softplus(P) * adam - fast_decay * W_fast
      dP      = eta_plast * progress * relevance(g) - stability * (P - P0)
      dQ      = alpha_q * dW_fast * max(progress, 0)

* task boundary (sleep):

      W_slow += consolidate_beta * Q + consolidate_fast_direct * W_fast
      W_fast *= consolidate_fast_decay
      Q      *= consolidate_q_decay
      m, v    reset (fresh Adam state for the new task)

This is deliberately a *per-parameter* rule rather than the per-connection
Learning Rule Network used on the MLP core: a per-connection F_phi over ~6.6M
weights is computationally infeasible on the CPU-only server, and the research
question of Stage 4 is the transfer of the developmental dynamics
(fast/slow/plasticity/consolidation) to a real Transformer, not the rule net.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerDLAConfig:
    plasticity_prior: float = 0.5
    eta_fast: float = 6e-4        # effective Adam step size for W_fast
    eta_plast: float = 0.02       # meta-plasticity step size for P
    fast_decay: float = 0.02      # per-step forgetting of W_fast
    stability_pressure: float = 0.01  # pull of P back to P0
    alpha_q: float = 0.30         # EMA rate of Q
    consolidate_beta: float = 1.0     # sleep: W_slow += beta * Q
    consolidate_fast_direct: float = 0.15  # sleep: W_slow += beta_f * W_fast
    consolidate_fast_decay: float = 0.5
    consolidate_q_decay: float = 0.7
    consolidate_p_decay: float = 0.0  # sleep relaxation of P towards P0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    p_min: float = -6.0
    p_max: float = 8.0
    cog_alpha: float = 0.1
    initial_loss_ema: float = 4.0
    initial_acc_ema: float = 0.3

    @property
    def p0(self) -> float:
        if self.plasticity_prior <= 0.0:
            return -6.0
        y = min(self.plasticity_prior, 20.0)
        return math.log(math.exp(y) - 1.0)


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class TransformerTempoParams(nn.Module):
    """phi_t - the learning-rule parameters of the Transformer DLA individual.

    These are the object of Stage 5.5: they start as DNA priors and can be
    updated in a lifetime by meta-gradient on FUTURE learning.  Stored in
    log/logit space so they stay in valid ranges.
    """

    def __init__(self, cfg: TransformerDLAConfig):
        super().__init__()
        self.log_eta_fast = nn.Parameter(torch.tensor(math.log(max(cfg.eta_fast, 1e-5))))
        self.log_eta_plast = nn.Parameter(torch.tensor(math.log(max(cfg.eta_plast, 1e-5))))
        self.logit_fast_decay = nn.Parameter(torch.tensor(_logit(cfg.fast_decay)))
        self.log_stability = nn.Parameter(torch.tensor(math.log(max(cfg.stability_pressure, 1e-5))))
        self.log_alpha_q = nn.Parameter(torch.tensor(math.log(max(cfg.alpha_q, 1e-5))))
        self.logit_consolidate_beta = nn.Parameter(torch.tensor(_logit(cfg.consolidate_beta)))
        self.logit_consolidate_fast_direct = nn.Parameter(torch.tensor(_logit(cfg.consolidate_fast_direct)))
        self.logit_consolidate_fast_decay = nn.Parameter(torch.tensor(_logit(cfg.consolidate_fast_decay)))
        self.logit_consolidate_q_decay = nn.Parameter(torch.tensor(_logit(cfg.consolidate_q_decay)))

    def values(self):
        return {
            "eta_fast": F.softplus(self.log_eta_fast),
            "eta_plast": F.softplus(self.log_eta_plast),
            "fast_decay": torch.sigmoid(self.logit_fast_decay),
            "stability_pressure": F.softplus(self.log_stability),
            "alpha_q": F.softplus(self.log_alpha_q),
            "consolidate_beta": torch.sigmoid(self.logit_consolidate_beta),
            "consolidate_fast_direct": torch.sigmoid(self.logit_consolidate_fast_direct),
            "consolidate_fast_decay": torch.sigmoid(self.logit_consolidate_fast_decay),
            "consolidate_q_decay": torch.sigmoid(self.logit_consolidate_q_decay),
        }

    def snapshot(self):
        return {k: float(v.detach()) for k, v in self.values().items()}


class DLAState:
    """Per-parameter developmental state of one DLA-Transformer individual."""

    def __init__(self, store: Dict[int, Dict[str, torch.Tensor]], ema: Dict[str, torch.Tensor]):
        self.store = store
        self.ema = ema
        self.step = 0
        self.life_step = 0

    def reset_moments(self):
        for s in self.store.values():
            s["m"].zero_()
            s["v"].zero_()
        self.step = 0


class _Ref:
    """Mutable holder so wrapper modules can see the currently active DLA state."""

    def __init__(self):
        self.state: Optional[DLAState] = None


class _DLA_Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, ref: _Ref):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.ref = ref
        self.state_key: Optional[int] = None

    def forward(self, x: torch.Tensor):
        if self.ref.state is None:
            return F.linear(x, self.weight, self.bias)
        s = self.ref.state.store[self.state_key]
        w_eff = self.weight + F.softplus(s["p"]) * s["w_fast"]
        return F.linear(x, w_eff, self.bias)


class _DLA_Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, ref: _Ref):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.ref = ref
        self.state_key: Optional[int] = None

    def forward(self, idx: torch.Tensor):
        if self.ref.state is None:
            return F.embedding(idx, self.weight)
        s = self.ref.state.store[self.state_key]
        w_eff = self.weight + F.softplus(s["p"]) * s["w_fast"]
        return F.embedding(idx, w_eff)


def make_dla_gpt_class(GPT):
    """Create a DLA variant of the given GPT class (nanoGPT ``model.GPT``)."""

    class DLA_GPT(GPT):
        def __init__(self, config, dla_cfg: TransformerDLAConfig):
            super().__init__(config)
            self.dla_cfg = dla_cfg
            self.tempos = TransformerTempoParams(dla_cfg)
            self.ref = _Ref()
            self.wrapped: List[Tuple[str, nn.Module]] = []
            self._wrap_all(self, "")

            # nanoGPT ties the token embedding with lm_head; restore the tie
            # after both modules have been replaced by DLA wrappers.
            self.transformer.wte.weight = self.lm_head.weight

            self.key_modules: Dict[int, nn.Module] = {}
            self.state_shapes: List[Tuple[int, Tuple[int, ...]]] = []
            self._collect_state_keys()

        # ------------------------------------------------------------- wrapping
        def _wrap_all(self, module: nn.Module, path: str):
            for name, child in list(module.named_children()):
                p = f"{path}.{name}" if path else name
                if isinstance(child, nn.Linear):
                    new = _DLA_Linear(child.in_features, child.out_features, child.bias is not None, self.ref)
                    with torch.no_grad():
                        new.weight.copy_(child.weight)
                        if new.bias is not None:
                            new.bias.copy_(child.bias)
                    setattr(module, name, new)
                    self.wrapped.append((p, new))
                elif isinstance(child, nn.Embedding):
                    new = _DLA_Embedding(child.num_embeddings, child.embedding_dim, self.ref)
                    with torch.no_grad():
                        new.weight.copy_(child.weight)
                    setattr(module, name, new)
                    self.wrapped.append((p, new))
                else:
                    self._wrap_all(child, p)

        def _collect_state_keys(self):
            seen = set()
            for name, mod in self.wrapped:
                key = mod.weight.data_ptr()
                mod.state_key = key
                if key not in seen:
                    seen.add(key)
                    self.state_shapes.append((key, tuple(mod.weight.shape)))
                    self.key_modules[key] = mod

        # ---------------------------------------------------------------- state
        def set_dla_state(self, state: Optional[DLAState]):
            self.ref.state = state
            return self

        def make_state(self, device: Optional[torch.device | str] = None) -> DLAState:
            device = device or next(self.parameters()).device
            store: Dict[int, Dict[str, torch.Tensor]] = {}
            p0 = self.dla_cfg.p0
            for key, shape in self.state_shapes:
                store[key] = {
                    "w_fast": torch.zeros(shape, device=device),
                    "p": torch.full(shape, p0, device=device),
                    "q": torch.zeros(shape, device=device),
                    "m": torch.zeros(shape, device=device),
                    "v": torch.zeros(shape, device=device),
                }
            ema = {
                "loss_ema": torch.tensor(self.dla_cfg.initial_loss_ema, device=device),
                "acc_ema": torch.tensor(self.dla_cfg.initial_acc_ema, device=device),
                "progress_ema": torch.tensor(0.0, device=device),
            }
            return DLAState(store, ema)

        # ------------------------------------------------------- wake (one step)
        def dla_step(self, idx: torch.Tensor, targets: torch.Tensor, state: DLAState) -> Dict[str, float]:
            cfg = self.dla_cfg
            self.set_dla_state(state)
            self.zero_grad(set_to_none=True)
            logits, loss = self(idx, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)

            with torch.no_grad():
                tv = self.tempos.values()
                mask = targets != -1
                acc = ((logits.argmax(dim=-1)[mask] == targets[mask]).float().mean()).item()
                loss_v = loss.item()

                a = cfg.cog_alpha
                prev_loss = state.ema["loss_ema"]
                new_loss_ema = (1 - a) * prev_loss + a * loss_v
                new_acc_ema = (1 - a) * state.ema["acc_ema"] + a * acc
                progress = math.tanh((prev_loss.item() - new_loss_ema.item()) / (prev_loss.item() + 1e-4))
                new_progress_ema = (1 - a) * state.ema["progress_ema"] + a * progress
                # consolidation success: relative improvement of the current step
                # over the recent loss EMA, clipped to [0,1].  Unlike ``progress``
                # (which can oscillate and average to ~0) this is always >= 0,
                # so Q reliably accumulates updates that actually helped.
                success = max(0.0, min(1.0, (prev_loss.item() - loss_v) / (prev_loss.item() + 1e-4)))

                for key, mod in self.key_modules.items():
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

                    # meta-plasticity: relevant params become more plastic while
                    # loss improves, less plastic while it stagnates/degrades
                    n = g.numel()
                    relevance = (g.abs() / (g.norm() / math.sqrt(n) + 1e-6)).clamp(0.0, 5.0)
                    dp = tv["eta_plast"] * progress * relevance - tv["stability_pressure"] * (s["p"] - self.dla_cfg.p0)
                    s["p"].add_(dp).clamp_(self.dla_cfg.p_min, self.dla_cfg.p_max)

                    # slow-eligibility trace: remember updates that worked
                    q_inc = dw * success
                    s["q"].mul_(1 - tv["alpha_q"]).add_(q_inc, alpha=tv["alpha_q"])

                state.ema["loss_ema"] = new_loss_ema
                state.ema["acc_ema"] = new_acc_ema
                state.ema["progress_ema"] = new_progress_ema
                state.step += 1
                state.life_step += 1

            return {
                "loss": loss_v,
                "ppl": math.exp(loss_v),
                "acc": acc,
                "progress": progress,
                "success": success,
                "loss_ema": new_loss_ema.item(),
                "life_step": state.life_step,
            }

        # ----------------------------------------------------- sleep (boundary)
        @torch.no_grad()
        def dla_sleep(self, state: DLAState):
            cfg = self.dla_cfg
            tv = self.tempos.values()
            for key, mod in self.key_modules.items():
                s = state.store[key]
                # sleep consolidation: eligibility trace Q + a direct transfer of
                # currently expressed fast knowledge into the slow store
                mod.weight.add_(tv["consolidate_beta"] * s["q"] + tv["consolidate_fast_direct"] * s["w_fast"])
                s["w_fast"].mul_(tv["consolidate_fast_decay"])
                s["q"].mul_(tv["consolidate_q_decay"])
                if cfg.consolidate_p_decay > 0:
                    s["p"].add_((self.dla_cfg.p0 - s["p"]) * cfg.consolidate_p_decay)
            state.reset_moments()
            return state

        # --------------------------------------- meta-learning of phi (Stage 5.5)
        def meta_unroll_loss(self, state: DLAState, batches):
            """Differentiable surrogate of the wake dynamics.

            Runs the same fast/plasticity updates (plain-gradient variant, no
            Adam moments) on a short sequence of batches, with
            ``backward(create_graph=True)`` so the meta-gradient can flow from
            FUTURE losses back into the learning-rule parameters phi (tempos).

            The actual ``state`` is NOT modified: a detached snapshot is used
            for the unroll and discarded afterwards.
            """
            cfg = self.dla_cfg
            tv = self.tempos.values()
            cur = {
                key: {
                    "w_fast": s["w_fast"].detach().clone(),
                    "p": s["p"].detach().clone(),
                    "q": s["q"].detach().clone(),
                }
                for key, s in state.store.items()
            }
            losses = []
            prev_loss = state.ema["loss_ema"].detach()
            for x, y in batches:
                fake = DLAState(cur, state.ema)
                self.set_dla_state(fake)
                self.zero_grad(set_to_none=True)
                logits, loss = self(x, y)
                # create_graph: the teaching signal g itself is differentiable
                # w.r.t. the fast weights/phi of previous steps.
                loss.backward(create_graph=True)
                loss_v = loss.detach()
                success = max(0.0, min(1.0, (prev_loss.item() - loss_v.item()) / (prev_loss.item() + 1e-4)))
                prev_loss = (1 - cfg.cog_alpha) * prev_loss + cfg.cog_alpha * loss_v

                new_cur = {}
                for key, mod in self.key_modules.items():
                    s = cur[key]
                    g = mod.weight.grad
                    if g is None:
                        new_cur[key] = {k: v.clone() for k, v in s.items()}
                        continue
                    gate = F.softplus(s["p"])
                    dw = -tv["eta_fast"] * gate * g - tv["fast_decay"] * s["w_fast"]
                    n = g.numel()
                    relevance = (g.abs() / (g.norm() / math.sqrt(n) + 1e-6)).clamp(0.0, 5.0)
                    progress = math.tanh((prev_loss.item() - loss_v.item()) / (prev_loss.item() + 1e-4))
                    dp = tv["eta_plast"] * progress * relevance - tv["stability_pressure"] * (s["p"] - cfg.p0)
                    q_inc = dw * success
                    new_cur[key] = {
                        "w_fast": s["w_fast"] + dw,
                        "p": (s["p"] + dp).clamp(cfg.p_min, cfg.p_max),
                        "q": (1 - tv["alpha_q"]) * s["q"] + tv["alpha_q"] * q_inc,
                    }
                cur = new_cur
                losses.append(loss)
            self.set_dla_state(state)
            self.zero_grad(set_to_none=True)
            return torch.stack(losses).mean()

        def meta_update_phi(self, state: DLAState, batches, opt: torch.optim.Optimizer, grad_clip: float = 1.0):
            """phi <- phi - lr * grad_phi L_future (one lifetime meta step).

            ``batches`` must come from the FUTURE task / held-out transfer task,
            so the gradient measures future learning, not current performance.
            """
            params = list(self.tempos.parameters())
            self.zero_grad(set_to_none=True)
            loss = self.meta_unroll_loss(state, batches)
            grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
            for p, g in zip(params, grads):
                p.grad = g.detach() if g is not None else None
            if any(g is not None for g in grads):
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            self.zero_grad(set_to_none=True)
            return loss.item()

        def tempo_delta(self, before: Dict[str, float]) -> Dict[str, float]:
            after = self.tempos.snapshot()
            return {k: after[k] - before[k] for k in before}

        # ------------------------------------------------------------ analysis
        def plasticity_by_layer(self, state: DLAState) -> Dict[str, float]:
            out = {}
            for name, mod in self.wrapped:
                s = state.store[mod.state_key]
                out[name] = F.softplus(s["p"]).mean().item()
            return out

        @torch.no_grad()
        def eval_loss(
            self,
            ids,
            block_size: int,
            batch_size: int,
            num_batches: int,
            state: Optional[DLAState] = None,
            rng=None,
            device=None,
        ) -> float:
            was_training = self.training
            self.eval()
            self.set_dla_state(state)
            device = device or next(self.parameters()).device
            n = len(ids) - block_size - 1
            if n <= 0:
                self.train(was_training)
                return float("nan")
            import random as _random

            rng = rng or _random.Random(1234)
            losses = []
            for _ in range(num_batches):
                ix = [rng.randrange(n) for _ in range(batch_size)]
                x = torch.stack([torch.tensor(ids[i : i + block_size], dtype=torch.long, device=device) for i in ix])
                y = torch.stack([torch.tensor(ids[i + 1 : i + 1 + block_size], dtype=torch.long, device=device) for i in ix])
                _, loss = self(x, y)
                losses.append(loss.item())
            self.train(was_training)
            return sum(losses) / len(losses)

    return DLA_GPT
