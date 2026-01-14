"""Script for Experiment 4: Bayesian Active Learning on BIWI Head Pose Dataset."""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from novel_extension.acquisition_functions import (
    random_acquisition,
    variance_acquisition,
)
from novel_extension.active_learning import (
    active_learning_loop,
    evaluate_deterministic,
    set_seed,
    train_deterministic,
)
from novel_extension.load_data import compute_label_stats, prep_data
from novel_extension.models import (
    BayesianEfficientNetModel,
    DeterministicRegressor,
    EfficientNetBasis,
)
from novel_extension.plot_utils import plot_results, print_summary


def download_dataset(data_dir: str) -> None:
    import tarfile

    import requests

    if not os.path.exists(data_dir):
        url = "https://s3.amazonaws.com/fast-ai-imagelocal/biwi_head_pose.tgz"
        save_path = "biwi_head_pose.tgz"
        print("Downloading BIWI dataset...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Extracting...")
        with tarfile.open(save_path, "r:gz") as tar:
            tar.extractall()
        print("Done!")
    else:
        print("Dataset already exists.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4: Novel Extension - BIWI Head Pose"
    )
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--acq_rounds", type=int, default=110)
    parser.add_argument("--n_query", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pool_ratio", type=float, default=0.05)
    parser.add_argument("--pool_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--likelihood_variance", type=float, default=1.0)
    parser.add_argument("--prior_variance", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default="biwi_head_pose")
    parser.add_argument("--result_dir", type=str, default="novel_extension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    download_dataset(args.data_dir)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 70)
    print("RUNNING ALL ACTIVE LEARNING EXPERIMENTS")
    print("=" * 70)

    experiments = {}

    model_configs = [
        ("mfvi", "MFVI"),
        ("analytical", "Analytical"),
    ]

    acquisition_configs = [
        (variance_acquisition, "Variance"),
        (random_acquisition, "Random"),
    ]

    (
        (init_loader, pool_loader, determ_train_loader, val_loader, test_loader),
        pool_dataset,
        (idx_labeled, idx_pool),
    ) = prep_data(
        data_dir=args.data_dir,
        seed=args.seed,
        pool_ratio=args.pool_ratio,
        pool_size=args.pool_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    for head_type, head_label in model_configs:
        for acq_fn, acq_label in acquisition_configs:
            exp_name = f"{head_label}_{acq_label}"
            print(f"\n{'=' * 70}")
            print(f"Experiment: {exp_name}")
            print(f"{'=' * 70}")

            backbone = EfficientNetBasis()
            model = BayesianEfficientNetModel(
                backbone=backbone,
                num_outputs=3,
                head_type=head_type,
                likelihood_variance=args.likelihood_variance,
                prior_variance=args.prior_variance,
            ).to(device)

            idx_labeled_al = idx_labeled.clone()
            idx_pool_al = idx_pool.clone()

            history = active_learning_loop(
                model=model,
                dataset=pool_dataset,
                idx_labeled=idx_labeled_al,
                idx_pool=idx_pool_al,
                val_loader=val_loader,
                device=device,
                acquisition_fn=acq_fn,
                test_loader=test_loader,
                acq_rounds=args.acq_rounds,
                n_query=args.n_query,
                batch_size=args.batch_size,
            )
            experiments[exp_name] = history
            final_val_mae = history["val_mae"][-1]
            final_val_rmse = history["val_rmse"][-1]
            final_test_mae = history.get("final_test_mae")
            final_test_rmse = history.get("final_test_rmse")
            print(
                f"Final Val MAE: {final_val_mae:.4f}, Final Val RMSE: {final_val_rmse:.4f}"
            )
            if final_test_mae is not None:
                print(
                    f"Final Test MAE: {final_test_mae:.4f}, Final Test RMSE: {final_test_rmse:.4f}"
                )
                print(f"Labeled samples: {history['n_labeled'][-1]}")

    print(f"\n{'=' * 70}")
    print("All experiments completed!")
    print(f"{'=' * 70}\n")

    print("\n" + "=" * 70)
    print("RUNNING DETERMINISTIC REGRESSOR (RANDOM SELECTION)")
    print("=" * 70)

    init_size = len(idx_labeled)
    total_determ_data = len(determ_train_loader.dataset)
    checkpoint_interval = 10
    det_sample_sizes = [
        init_size + i * args.n_query * checkpoint_interval
        for i in range(args.acq_rounds // checkpoint_interval + 1)
    ]

    y_mean_det, y_std_det = compute_label_stats(determ_train_loader, device)
    for loss_fn in ["L1", "MSE"]:
        print(f"{'=' * 70}")
        print(f"Experiment: Deterministic_{loss_fn}")
        print(f"{'=' * 70}")
        det_history = {
            "iteration": [],
            "n_labeled": [],
            "val_mse": [],
            "val_rmse": [],
            "val_mae": [],
        }
        for idx, sample_size in enumerate(det_sample_sizes):
            subset_frac = sample_size / total_determ_data
            backbone_det = EfficientNetBasis()
            det_model = DeterministicRegressor(backbone_det, num_outputs=3).to(device)
            if loss_fn == "L1":
                final_loss = train_deterministic(
                    det_model,
                    determ_train_loader,
                    device,
                    y_mean_det,
                    y_std_det,
                    loss_fn=nn.L1Loss(),
                    epochs=50,
                    lr=1e-3,
                    weight_decay=1e-4,
                    subset_frac=subset_frac,
                    seed=args.seed + idx,
                )
            else:
                final_loss = train_deterministic(
                    det_model,
                    determ_train_loader,
                    device,
                    y_mean_det,
                    y_std_det,
                    loss_fn=nn.MSELoss(),
                    epochs=30,
                    lr=1e-3,
                    weight_decay=1e-4,
                    subset_frac=subset_frac,
                    seed=args.seed + idx,
                )
            val_metrics_det = evaluate_deterministic(
                det_model, val_loader, device, y_mean_det, y_std_det
            )

            det_history["iteration"].append(idx + 1)
            det_history["n_labeled"].append(sample_size)
            det_history["val_mse"].append(val_metrics_det["mse"])
            det_history["val_rmse"].append(val_metrics_det["rmse"])
            det_history["val_mae"].append(val_metrics_det["mae"])
            print(
                f"Iteration {idx + 1}/{len(det_sample_sizes)} | Labeled set size: {sample_size} | Val RMSE: {val_metrics_det['rmse']:.4f}, Val MAE: {val_metrics_det['mae']:.4f}"
            )

        test_metrics_det = evaluate_deterministic(
            det_model, test_loader, device, y_mean_det, y_std_det
        )
        det_history["final_test_rmse"] = test_metrics_det["rmse"]
        det_history["final_test_mae"] = test_metrics_det["mae"]
        print(
            f"Final Val RMSE: {det_history['val_rmse'][-1]:.4f} | Final Val MAE: {det_history['val_mae'][-1]:.4f}"
        )
        print(
            f"Final Test RMSE: {test_metrics_det['rmse']:.4f} | Final Test MAE: {test_metrics_det['mae']:.4f}"
        )
        print(f"\nDeterministic trained on: {det_history['n_labeled']} samples")
        experiments[f"Deterministic_{loss_fn}"] = det_history

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    plot_results(
        experiments,
        save_path=os.path.join(args.result_dir, "experiment4_results.png"),
    )

    print_summary(experiments)


if __name__ == "__main__":
    main()
