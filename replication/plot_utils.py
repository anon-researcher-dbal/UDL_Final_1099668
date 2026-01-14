# Plotting utilities for active learning experiments

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_and_save_results(
    results: Dict[str, np.ndarray],
    output_dir: str,
    title: str = "Active learning accuracy vs no. of labelled queries",
    exp_type: str = "bayesian",
    init_size: int = 20,
    query_size: int = 10,
) -> None:
    """Plot results and save to file.

    Args:
        results: Dict mapping curve names to accuracy arrays.
        output_dir: Directory to save plot.png.
        title: Plot title.
        exp_type: "bayesian" (pastel blue) or "mixed" (pastel colors per type).
        init_size: Initial number of labeled samples.
        query_size: Number of samples queried per round.
    """
    # Pastel color palette
    pastel_colors = {
        "bayesian": "#A8D8EA",  # Pastel blue
        "deterministic": "#FFB3BA",  # Pastel red
        "uniform": "#BAFFC9",  # Pastel green
        "max_entropy": "#FFFFBA",  # Pastel yellow
        "bald": "#D4BAFF",  # Pastel purple
        "var_ratios": "#FFDFBA",  # Pastel orange
        "mean_std": "#FFB3E6",  # Pastel pink
    }

    if exp_type == "bayesian":
        friendly_labels = {
            "uniform-MC_dropout=True": "Uniform (Bayesian)",
            "max_entropy-MC_dropout=True": "Max Entropy (Bayesian)",
            "bald-MC_dropout=True": "BALD (Bayesian)",
            "var_ratios-MC_dropout=True": "Variation Ratios (Bayesian)",
            "mean_std-MC_dropout=True": "Mean STD (Bayesian)",
        }

        def color_fn(name: str) -> str:
            # Map by acquisition function name
            if "uniform" in name:
                return pastel_colors["uniform"]
            elif "max_entropy" in name:
                return pastel_colors["max_entropy"]
            elif "bald" in name:
                return pastel_colors["bald"]
            elif "var_ratios" in name:
                return pastel_colors["var_ratios"]
            elif "mean_std" in name:
                return pastel_colors["mean_std"]
            return pastel_colors["bayesian"]

    elif exp_type == "mixed":
        friendly_labels = {
            "uniform-MC_dropout=True": "Uniform (Bayesian)",
            "uniform-MC_dropout=False": "Uniform (Deterministic)",
            "max_entropy-MC_dropout=True": "Max Entropy (Bayesian)",
            "max_entropy-MC_dropout=False": "Max Entropy (Deterministic)",
            "bald-MC_dropout=True": "BALD (Bayesian)",
            "bald-MC_dropout=False": "BALD (Deterministic)",
            "var_ratios-MC_dropout=True": "Variation Ratios (Bayesian)",
            "var_ratios-MC_dropout=False": "Variation Ratios (Deterministic)",
            "mean_std-MC_dropout=True": "Mean STD (Bayesian)",
            "mean_std-MC_dropout=False": "Mean STD (Deterministic)",
        }

        def color_fn(name: str) -> str:
            # Darker pastel for deterministic, lighter for bayesian
            label = friendly_labels.get(name, name)
            is_det = "Deterministic" in label

            if "uniform" in name:
                return "#7FB069" if is_det else pastel_colors["uniform"]
            elif "max_entropy" in name:
                return "#D4AF37" if is_det else pastel_colors["max_entropy"]
            elif "bald" in name:
                return "#9966CC" if is_det else pastel_colors["bald"]
            elif "var_ratios" in name:
                return "#CC8800" if is_det else pastel_colors["var_ratios"]
            elif "mean_std" in name:
                return "#FF66B2" if is_det else pastel_colors["mean_std"]

            return (
                pastel_colors["deterministic"] if is_det else pastel_colors["bayesian"]
            )

    else:
        raise ValueError(f"Unknown exp_type: {exp_type}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for name, curve in results.items():
        label = friendly_labels.get(name, name)
        color = color_fn(name)
        x_values = init_size + np.arange(len(curve)) * query_size
        ax.plot(x_values, curve, label=label, color=color, linewidth=2.5)

    ax.set_xlabel("Number of acquired images", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.close()
