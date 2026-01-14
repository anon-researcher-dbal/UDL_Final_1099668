"""Script for Experiment 1 (Bayesian runs, all acquisition functions)."""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from replication.active_learning import (
    active_learning_procedure,
    select_acq_function,
    set_seed,
)
from replication.cnn_model import ConvNN
from replication.load_data import LoadData
from replication.plot_utils import plot_and_save_results


def load_cnn_model() -> nn.Module:
    """Factory to create the convolutional model used in experiments."""
    return ConvNN()


def save_as_npy(data: np.ndarray, folder: str, name: str) -> None:
    file_name = os.path.join(folder, name + ".npy")
    np.save(file_name, data)
    print(f"Saved: {file_name}")


def print_elapsed_time(start_time: float, exp: int, acq_func: str) -> None:
    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int(elapsed % 3600 // 60)
    s = int(elapsed % 60)
    print(f"********** Trial {exp} ({acq_func}): {h}:{m}:{s} **********")


def compute_images_to_error_table(
    results: Dict[str, np.ndarray], query_size: int = 10
) -> None:
    """Compute and print table of images needed to reach error thresholds."""
    init_size = 20
    error_thresholds = [0.10, 0.05]
    accuracy_thresholds = [1 - e for e in error_thresholds]
    func_map = {
        "bald": "BALD",
        "var_ratios": "Var Ratios",
        "max_entropy": "Max Ent",
        "mean_std": "Mean STD",
        "uniform": "Random",
    }

    table_data = {func: [] for func in func_map.values()}

    for error_threshold, acc_threshold in zip(error_thresholds, accuracy_thresholds):
        for name, curve in results.items():
            func_name = name.split("-")[0]
            if func_name in func_map:
                display_name = func_map[func_name]
                indices = np.where(curve >= acc_threshold)[0]
                if len(indices) > 0:
                    step = indices[0]
                    n_images = init_size + step * query_size
                    table_data[display_name].append(n_images)
                else:
                    table_data[display_name].append(None)

    print("\n" + "=" * 70)
    print("Table: Number of acquired images to reach model error thresholds on MNIST")
    print("=" * 70)

    header = "% error  | " + " | ".join(func_map.values())
    print(header)
    print("-" * len(header))

    for i, error_pct in enumerate([10, 5]):
        row = f"{error_pct:>7}% | "
        values = []
        for func in func_map.values():
            val = table_data[func][i]
            if val is not None:
                values.append(f"{val:>10}")
            else:
                values.append(f"{'N/A':>10}")
        row += " | ".join(values)
        print(row)

    print("=" * 70 + "\n")


def train_active_learning(args, device, datasets: dict) -> Dict[str, np.ndarray]:
    """Run active learning for configured acquisition functions."""
    acq_functions = select_acq_function(args.acq_func)
    results: Dict[str, np.ndarray] = {}
    state_loop = [True, False] if args.determ else [True]

    for state in state_loop:
        for acq_func in acq_functions:
            avg_hist: List[List[float]] = []
            test_scores: List[float] = []
            acq_func_name = f"{acq_func.__name__}-MC_dropout={state}"
            print(f"\n---------- Start {acq_func_name} training ----------")
            for e in range(args.trials):
                set_seed(args.seed + e)
                start_time = time.time()
                print(f"********** Trials: {e + 1}/{args.trials} **********")
                training_hist, test_score = active_learning_procedure(
                    query_strategy=acq_func,
                    X_val=datasets["X_val"],
                    y_val=datasets["y_val"],
                    X_test=datasets["X_test"],
                    y_test=datasets["y_test"],
                    X_pool=datasets["X_pool"],
                    y_pool=datasets["y_pool"],
                    X_init=datasets["X_init"],
                    y_init=datasets["y_init"],
                    build_model=load_cnn_model,
                    T=args.dropout_iter,
                    n_query=args.query,
                    training=state,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                )
                avg_hist.append(training_hist)
                test_scores.append(test_score)
                print_elapsed_time(start_time, e + 1, acq_func_name)
            avg_hist_arr = np.average(np.array(avg_hist), axis=0)
            avg_test = sum(test_scores) / len(test_scores)
            print(f"Average Test score for {acq_func_name}: {avg_test}")
            results[acq_func_name] = avg_hist_arr
            save_as_npy(
                data=avg_hist_arr,
                folder=args.result_dir,
                name=acq_func_name,
            )
    print("--------------- Training Complete ---------------")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1: Bayesian (MC-dropout)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--dropout_iter", type=int, default=10)
    parser.add_argument("--query", type=int, default=10)
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--result_dir", type=str, default="exp1_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.acq_func = 0  # all acquisition functions
    args.determ = False  # Bayesian only
    args.weight_decay = 1e-2  # L2 regularization

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_loader = LoadData(val_size=args.val_size, seed=args.seed)
    (
        X_init,
        y_init,
        X_val,
        y_val,
        X_pool,
        y_pool,
        X_test,
        y_test,
    ) = data_loader.load_all()

    datasets = {
        "X_init": X_init,
        "y_init": y_init,
        "X_val": X_val,
        "y_val": y_val,
        "X_pool": X_pool,
        "y_pool": y_pool,
        "X_test": X_test,
        "y_test": y_test,
    }

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    results = train_active_learning(args, device, datasets)
    results_arr = np.array(results, dtype=object)
    np.save(
        os.path.join(args.result_dir, "results.npy"),
        results_arr,
    )

    plot_and_save_results(
        results,
        args.result_dir,
        title="Active learning accuracy vs no. of acquired images (Experiment 1)",
        exp_type="bayesian",
        init_size=20,
        query_size=args.query,
    )

    compute_images_to_error_table(results, query_size=args.query)


if __name__ == "__main__":
    main()
