"""DNA variants - fixed "learning tendencies" of an individual.

DNA is NOT knowledge.  It is the set of priors that shape how the individual
learns: plasticity prior, update tempos, forgetting, consolidation pressure,
and the weight of novelty / uncertainty / reward channels fed into F_phi.

Four individuals used in the lifetime experiment:

    DNA-A  high plasticity   : learns fast, forgets fast
    DNA-B  high stability    : learns slowly, forgets little
    DNA-C  novelty seeking   : novelty / uncertainty channels amplified
    DNA-D  conservative      : novelty suppressed, reward signal dominant

Every individual shares the same meta-learned F_phi (the rule network weights),
so the trajectories differ only because of DNA priors - exactly the
"same life, different developmental trajectories" experiment.
"""

from __future__ import annotations

from dataclasses import replace

from .config import DLAConfig


def _weights(**kwargs) -> dict:
    w = {k: 1.0 for k in (
        "error", "novelty", "uncertainty", "reward",
        "confidence", "skill", "fatigue", "progress", "knowledge",
    )}
    w.update(kwargs)
    return w


def dna_presets(base: DLAConfig) -> dict:
    A = replace(base)
    A.plasticity_prior = 0.85
    A.eta_fast = 0.35
    A.fast_decay = 0.12
    A.stability_pressure = 0.005
    A.consolidate_beta = 0.08

    B = replace(base)
    B.plasticity_prior = 0.15
    B.eta_fast = 0.12
    B.fast_decay = 0.02
    B.stability_pressure = 0.02
    B.consolidate_beta = 0.30

    C = replace(base)
    C.signal_weights = _weights(novelty=1.8, uncertainty=1.5, error=1.2)

    D = replace(base)
    D.plasticity_prior = 0.4
    D.signal_weights = _weights(novelty=0.3, uncertainty=0.3, reward=1.5)

    return {
        "A_high_plasticity": A,
        "B_high_stability": B,
        "C_novelty_seeking": C,
        "D_conservative": D,
    }
