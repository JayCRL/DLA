"""Developmental Learning Architecture (DLA) v0.2.

A neural system in which the learning rule itself is a developmental state:

    DNA phi -> Learning Rules -> Cognitive State -> Knowledge / Skill

See README.md for the theory-to-code mapping and the staged experimental plan.
"""

from .config import CoreConfig, DLAConfig, SIGNAL_NAMES
from .model import DevelopmentalNet
from .baselines import StaticMLP
from .tasks import make_task_stream, make_gaussian_task, make_xor_task
from .meta import meta_train, LifetimeAdapter, unroll_loss

__all__ = [
    "CoreConfig",
    "DLAConfig",
    "SIGNAL_NAMES",
    "DevelopmentalNet",
    "StaticMLP",
    "make_task_stream",
    "make_gaussian_task",
    "make_xor_task",
    "meta_train",
    "LifetimeAdapter",
    "unroll_loss",
]
