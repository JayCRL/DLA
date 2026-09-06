"""Matplotlib-based reporting (Agg backend, no display needed on the server)."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_learning_curves(
    curves_by_name: Dict[str, Sequence[Sequence[float]]],
    save_path: str,
    title: str = "Online test accuracy (learning curves)",
    xlabel: str = "training batches",
    ylabel: str = "test accuracy",
):
    """Each entry: name -> list of curves (one per task); mean curve is drawn."""
    _ensure_dir(save_path)
    plt.figure(figsize=(7, 4.5))
    for name, curves in curves_by_name.items():
        n = max(len(c) for c in curves)
        xs, means, stds = [], [], []
        for i in range(n):
            vals = [c[i] for c in curves if i < len(c)]
            xs.append(i + 1)
            means.append(sum(vals) / len(vals))
            stds.append((sum((v - means[-1]) ** 2 for v in vals) / len(vals)) ** 0.5)
        plt.plot(xs, means, label=name)
        plt.fill_between(xs, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.15)
    plt.axhline(0.85, color="gray", linestyle="--", linewidth=0.8, label="threshold (85%)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.35, 1.02)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_bars(
    names: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    save_path: str,
    ylabel: str,
    title: str,
    ylim=None,
):
    _ensure_dir(save_path)
    plt.figure(figsize=(0.55 * len(names) + 2.5, 4.2))
    xs = list(range(len(names)))
    plt.bar(xs, means, yerr=stds, capsize=4, alpha=0.85)
    plt.xticks(xs, names)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_trajectories(
    series_by_name: Dict[str, List[float]],
    save_path: str,
    xlabel: str = "task index",
    ylabel: str = "value",
    title: str = "",
):
    _ensure_dir(save_path)
    plt.figure(figsize=(7, 4.5))
    for name, series in series_by_name.items():
        plt.plot(list(range(len(series))), series, marker="o", markersize=3, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_lambda(series_by_name: Dict[str, List[float]], save_path: str, threshold: float = 0.85):
    """Lambda_t = 1 / steps-to-threshold, the operational learning capacity."""
    plot_trajectories(
        series_by_name,
        save_path,
        xlabel="task index in lifetime",
        ylabel=f"Lambda_t = 1 / steps-to-{threshold:.0%}",
        title="Development of learning capacity over a lifetime",
    )
