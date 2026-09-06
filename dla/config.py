"""Configuration objects for the Developmental Learning Architecture (DLA) v0.2.

Theory mapping
--------------
The four levels from the v0.2 whitepaper:

    DNA phi  ->  Learning Rules  ->  Cognitive State  ->  Knowledge / Skill

are realised as:

    DNA phi        : rule_net parameters + tempo parameters (learning-rule parameters)
                     + fixed architectural priors (plasticity prior, signal weights)
    Learning rules : the per-connection update F_phi -> (dW_fast, dP, dQ_slow)
    Cognitive state: EMA state vector
                     [knowledge, confidence, uncertainty, skill, fatigue, progress]
    Knowledge/skill: W_slow (consolidated knowledge) and W_fast (current skill trace)

The three parameter levels of the system are:

    W_slow  - long-term knowledge
    W_fast  - current learning / fast weights
    P       - plasticity (the ability to learn), itself a developmental state
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

# Names of the 9 global/cognitive channels fed to the Learning Rule Network.
SIGNAL_NAMES: Tuple[str, ...] = (
    "error",
    "novelty",
    "uncertainty",
    "reward",
    "confidence",
    "skill",
    "fatigue",
    "progress",
    "knowledge",
)

# 6 local channels (x_i, h_j, delta_j, w_fast_ij, p_ij, w_slow_ij) + 9 global channels
N_LOCAL_FEATURES = 6
FEAT_DIM = N_LOCAL_FEATURES + len(SIGNAL_NAMES)


@dataclass
class CoreConfig:
    """The Neural Core (any architecture is allowed in principle; v0.2 ships an MLP)."""

    input_dim: int = 6
    hidden_dims: Tuple[int, ...] = (32,)
    output_dim: int = 2
    activation: str = "tanh"


def _default_signal_weights() -> dict:
    return {name: 1.0 for name in SIGNAL_NAMES}


@dataclass
class DLAConfig:
    """Full DLA configuration.

    All fields that start with ``eta_``, ``fast_decay``, ``stability_pressure``,
    ``alpha_q`` and ``consolidate_*`` are *learning-rule parameters* (part of phi).
    When ``learn_tempos=True`` they are initialised to the values below and are
    themselves meta-learned; otherwise they stay fixed, i.e. they are pure DNA priors.
    """

    core: CoreConfig = field(default_factory=CoreConfig)

    # ---- Learning Rule Network (phi) -------------------------------------
    rule_hidden: int = 16
    # "learned": F_phi outputs per-connection [alignment, magnitude, dP, q_gate]
    # "hebbian": fixed rule  dW = eta * softplus(P) * (h x^T)  (baseline)
    rule_mode: str = "learned"

    # ---- learning-rule parameters (initial values; meta-learnable) -------
    eta_fast: float = 0.25       # step size of the fast-weight update
    eta_plast: float = 0.05      # step size of the meta-plasticity update dP
    fast_decay: float = 0.05     # per-experience forgetting of W_fast
    stability_pressure: float = 0.01  # pull of P back to the plasticity prior P0
    plasticity_prior: float = 0.5     # softplus(P0) at birth; DNA plasticity prior
    alpha_q: float = 0.10        # EMA rate of the slow-eligibility trace Q
    consolidate_beta: float = 0.10    # sleep: W_slow += beta * Q
    consolidate_fast_decay: float = 0.5   # sleep: W_fast *= ...
    consolidate_q_decay: float = 0.7      # sleep: Q *= ...

    # ---- plasticity bounds ------------------------------------------------
    p_min: float = -6.0
    p_max: float = 8.0

    # ---- cognitive state ---------------------------------------------------
    cog_alpha: float = 0.10  # EMA rate of cognitive state variables

    # ---- DNA signal priors -------------------------------------------------
    # Fixed multiplicative priors on the global/cognitive channels.  These are NOT
    # trained; they model "learning tendencies" encoded in the DNA of the system.
    signal_weights: dict = field(default_factory=_default_signal_weights)

    # ---- meta-learning ------------------------------------------------------
    learn_tempos: bool = True

    # ------------------------------------------------------------------ helpers
    @property
    def feat_dim(self) -> int:
        return FEAT_DIM

    @property
    def p0(self) -> float:
        """Raw plasticity value whose softplus equals ``plasticity_prior``."""
        if self.plasticity_prior <= 0.0:
            return -6.0
        y = min(self.plasticity_prior, 20.0)
        return math.log(math.exp(y) - 1.0)

    def layer_dims(self):
        dims = [self.core.input_dim, *self.core.hidden_dims, self.core.output_dim]
        return list(zip(dims[:-1], dims[1:]))

    def validate(self):
        assert self.rule_mode in ("learned", "hebbian")
        assert self.eta_fast > 0 and self.eta_plast > 0
        assert 0.0 <= self.fast_decay < 1.0
        assert self.core.input_dim > 1 and self.core.output_dim >= 2
        for name in SIGNAL_NAMES:
            self.signal_weights.setdefault(name, 1.0)
        return self
