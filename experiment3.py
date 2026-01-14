"""Script for Experiment 3: Bayesian Active Learning on MNIST"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from min_extension.active_learning import (
    _mse_rmse,
    acquisition_round,
    init_head_posterior,
    one_hot_labels,
    pretrainer,
    set_seed,
)
from min_extension.load_data import LoadData
from min_extension.models import MFVI_HB, Analytical_HB, ConvNNBackbone
from min_extension.plot_utils import plot_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Hierarchical Bayesian Active Learning"
    )
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--acq_rounds", type=int, default=95)
    parser.add_argument("--vi_method", type=str, default="closed")
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_weight_decay", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--query", type=int, default=10)
    parser.add_argument("--pretrain_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--initial_labeled_per_class", type=int, default=5)
    parser.add_argument("--pretrain_task", type=str, default="classification")
    parser.add_argument("--inference_type", type=str, default="all")
    parser.add_argument("--acq_fn", type=str, default="all")
    parser.add_argument("--likelihood_variance", type=float, default=1.0)
    parser.add_argument("--prior_variance", type=float, default=1.0)
    parser.add_argument("--result_dir", type=str, default="min_extension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_loader = LoadData(
        seed=args.seed,
        pretrain_size=args.pretrain_size,
        val_size=args.val_size,
        initial_per_class=args.initial_labeled_per_class,
    )
    (
        X_pretrain,
        y_pretrain,
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
        "X_pretrain": X_pretrain,
        "y_pretrain": y_pretrain,
        "X_init": X_init,
        "y_init": y_init,
        "X_val": X_val,
        "y_val": y_val,
        "X_pool": X_pool,
        "y_pool": y_pool,
        "X_test": X_test,
        "y_test": y_test,
    }

    backbone = ConvNNBackbone(
        num_filters=32,
        kernel_size=4,
        img_rows=28,
        img_cols=28,
        dense_layer=128,
        use_fc2_features=True,
    )
    backbone, backbone_train_acc, backbone_val_acc, backbone_val_mse = pretrainer(
        backbone_model=backbone,
        X_pretrain=datasets["X_pretrain"],
        Y_pretrain=datasets["y_pretrain"],
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        weight_decay=args.pretrain_weight_decay,
        device=device,
        task=args.pretrain_task,
        X_val=datasets["X_val"],
        y_val=datasets["y_val"],
    )
    print(f"Backbone pretraining complete. Validation MSE: {backbone_val_mse:.6f}")
    backbone.freeze()

    models = []
    acq_fns = []
    if args.inference_type == "all":
        models = ["analytic", "mfvi"]
    else:
        models.append(args.inference_type)
    if args.acq_fn == "all":
        acq_fns = ["predictive", "random"]
    else:
        acq_fns.append(args.acq_fn)

    results_rmse = {}
    results_mse = {}
    for model_type in models:
        for acq_fn in acq_fns:
            if model_type == "analytic":
                model = Analytical_HB(
                    feature_extractor=backbone,
                    likelihood_variance=args.likelihood_variance,
                    prior_variance=args.prior_variance,
                )
            elif model_type == "mfvi":
                model = MFVI_HB(
                    feature_extractor=backbone,
                    vi_method=args.vi_method,
                    likelihood_variance=args.likelihood_variance,
                    prior_variance=args.prior_variance,
                )

            X_labeled, y_labeled_oh = init_head_posterior(
                model,
                datasets["X_init"],
                datasets["y_init"],
                device,
                num_classes=10,
            )

            X_pool_run = datasets["X_pool"].copy()
            y_pool_oh_run = one_hot_labels(
                datasets["y_pool"].astype(int), num_classes=10
            )

            rmse_curve = []
            mse_curve = []
            for rnd in range(args.acq_rounds):
                X_labeled, y_labeled_oh, X_pool_run, y_pool_oh_run, _ = (
                    acquisition_round(
                        model=model,
                        X_labeled=X_labeled,
                        y_labeled_oh=y_labeled_oh,
                        X_pool_run=X_pool_run,
                        y_pool_oh_run=y_pool_oh_run,
                        n_query=args.query,
                        acq_fn=acq_fn,
                        device=device,
                    )
                )
                val_mse, val_rmse = _mse_rmse(
                    model, datasets["X_val"], datasets["y_val"], device
                )
                mse_curve.append(val_mse)
                rmse_curve.append(val_rmse)

                print(
                    f"Round {rnd + 1}/{args.acq_rounds} [{model_type}-{acq_fn}] | MSE: {val_mse:.6f} | RMSE: {val_rmse:.6f}"
                )

            key = f"{acq_fn}-{model_type}"
            results_rmse[key] = rmse_curve
            results_mse[key] = mse_curve

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    np.save(
        os.path.join(args.result_dir, "results_rmse.npy"),
        np.array(results_rmse, dtype=object),
    )
    np.save(
        os.path.join(args.result_dir, "results_mse.npy"),
        np.array(results_mse, dtype=object),
    )

    plot_results(
        results_rmse=results_rmse,
        models=models,
        acq_fns=acq_fns,
        title="Validation RMSE: Random vs Predictive (Analytic vs MFVI)",
        save_path=os.path.join(args.result_dir, "experiment3_results.png"),
    )


if __name__ == "__main__":
    main()
