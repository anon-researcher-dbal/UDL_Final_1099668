from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt


def plot_results(
    experiments: Dict[str, dict],
    save_path: str | None = None,
):
    tab20 = plt.cm.tab20.colors

    colors = {
        "Analytical_Variance": tab20[0],
        "Analytical_Random": tab20[1],
        "MFVI_Variance": tab20[6],
        "MFVI_Random": tab20[7],
        "Deterministic_L1": tab20[4],
        "Deterministic_MSE": tab20[5],
    }
    linestyles = {
        "Analytical_Variance": "-",
        "Analytical_Random": "--",
        "MFVI_Variance": "-",
        "MFVI_Random": "--",
        "Deterministic_L1": "-",
        "Deterministic_MSE": "--",
    }
    markers = {
        "Analytical_Variance": None,
        "Analytical_Random": None,
        "MFVI_Variance": None,
        "MFVI_Random": None,
        "Deterministic_L1": "s",
        "Deterministic_MSE": "s",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for exp_name, history in experiments.items():
        ax.plot(
            history["n_labeled"],
            history["val_mae"],
            color=colors.get(exp_name, "tab:gray"),
            linestyle=linestyles.get(exp_name, "-"),
            marker=markers.get(exp_name, None),
            markersize=4 if markers.get(exp_name) else 0,
            linewidth=2.5,
            label=exp_name.replace("_", " | "),
        )
    ax.set_xlabel("# Labeled Samples")
    ax.set_ylabel("Validation MAE")
    ax.set_title("Validation MAE vs Labeled Samples")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for exp_name, history in experiments.items():
        ax.plot(
            history["n_labeled"],
            history["val_rmse"],
            color=colors.get(exp_name, "tab:gray"),
            linestyle=linestyles.get(exp_name, "-"),
            marker=markers.get(exp_name, None),
            markersize=4 if markers.get(exp_name) else 0,
            linewidth=2.5,
            label=exp_name.replace("_", " | "),
        )
    ax.set_xlabel("# Labeled Samples")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation RMSE vs Labeled Samples")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def print_summary(experiments: Dict[str, dict]):
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL EXPERIMENTS (MAE / RMSE)")
    print("=" * 80)
    print(
        f"{'Configuration':<30} {'Final Val MAE':<15} {'Final Val RMSE':<17} {'Final Test MAE':<15} {'Final Test RMSE':<17}"
    )
    print("-" * 80)
    for exp_name, history in sorted(experiments.items()):
        final_val_mae = history["val_mae"][-1]
        final_val_rmse = history["val_rmse"][-1]
        final_test_mae = history.get("final_test_mae", float("nan"))
        final_test_rmse = history.get("final_test_rmse", float("nan"))
        print(
            f"{exp_name:<30} "
            f"{final_val_mae:<15.4f} "
            f"{final_val_rmse:<17.4f} "
            f"{final_test_mae:<15.4f} "
            f"{final_test_rmse:<17.4f}"
        )
    print("=" * 80)
