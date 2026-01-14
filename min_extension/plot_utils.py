from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    results_rmse: Dict[str, list],
    models: list,
    acq_fns: list,
    title: str = "Validation RMSE: Random vs Predictive (Analytic vs MFVI)",
    save_path: str | None = None,
) -> None:
    friendly_labels = {
        "predictive-analytic": "Predictive (Analytic)",
        "random-analytic": "Random (Analytic)",
        "predictive-mfvi": "Predictive (MFVI)",
        "random-mfvi": "Random (MFVI)",
    }

    colors = {
        "analytic": "tab:blue",
        "mfvi": "tab:red",
    }

    linestyles = {
        "predictive": "-",
        "random": "--",
    }

    plt.figure(figsize=(8, 5))
    for inf in models:
        for acq in acq_fns:
            key = f"{acq}-{inf}"
            if key in results_rmse:
                curve = results_rmse[key]
                x_vals = np.arange(0, len(curve))
                label = friendly_labels.get(key, key)
                color = colors.get(inf, "tab:gray")
                linestyle = linestyles.get(acq, "-")
                plt.plot(x_vals, curve, label=label, linestyle=linestyle, color=color)
    plt.xlabel("Acquisition round")
    plt.ylabel("RMSE")
    plt.xlim(left=0, right=100)
    plt.title(title)
    plt.grid()
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
