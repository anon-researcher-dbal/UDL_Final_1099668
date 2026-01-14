from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    y_mean: Optional[torch.Tensor] = None,
    y_std: Optional[torch.Tensor] = None,
) -> dict:
    model.backbone.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds, _ = model.predict_with_uncertainty(batch_x)
            predictions.append(preds.cpu())
            targets.append(batch_y.cpu())
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    if y_mean is not None and y_std is not None:
        predictions = predictions * y_std.cpu() + y_mean.cpu()
    mse = torch.mean((predictions - targets) ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets)).item()
    return {"mse": mse, "rmse": rmse, "mae": mae}


def active_learning_loop(
    model,
    dataset,
    idx_labeled: torch.Tensor,
    idx_pool: torch.Tensor,
    val_loader: DataLoader,
    device: torch.device,
    acquisition_fn,
    acq_rounds: int = 10,
    n_query: int = 10,
    batch_size: int = 64,
    test_loader: Optional[DataLoader] = None,
) -> dict:
    from novel_extension.load_data import compute_label_stats

    history = {
        "iteration": [],
        "n_labeled": [],
        "val_mse": [],
        "val_rmse": [],
        "val_mae": [],
    }

    for iteration in range(acq_rounds):
        labeled_subset = Subset(dataset, idx_labeled.tolist())
        stats_loader = DataLoader(
            labeled_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        y_mean, y_std = compute_label_stats(stats_loader, device)
        labeled_loader = DataLoader(
            labeled_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        model.fit_posterior(labeled_loader, device, y_mean=y_mean, y_std=y_std)
        pool_subset = Subset(dataset, idx_pool.tolist())
        pool_loader = DataLoader(
            pool_subset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
        )

        selected_pool_indices = acquisition_fn(
            model, pool_loader, device, n_query=n_query, y_std=y_std
        )
        selected_full_indices = [idx_pool[i].item() for i in selected_pool_indices]

        idx_labeled = torch.cat(
            [idx_labeled, torch.tensor(selected_full_indices, dtype=idx_labeled.dtype)]
        )
        idx_pool = torch.tensor(
            [i for i in idx_pool.tolist() if i not in selected_full_indices],
            dtype=idx_pool.dtype,
        )

        val_metrics = evaluate_model(
            model, val_loader, device, y_mean=y_mean, y_std=y_std
        )

        history["iteration"].append(iteration + 1)
        history["n_labeled"].append(len(idx_labeled))
        history["val_mse"].append(val_metrics["mse"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        print(
            f"Iteration {iteration + 1}/{acq_rounds} | Labeled set size: {len(idx_labeled)} | Val RMSE: {val_metrics['rmse']:.3f} | Val MAE: {val_metrics['mae']:.3f}"
        )

    if test_loader is not None:
        final_stats_loader = DataLoader(
            Subset(dataset, idx_labeled.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        y_mean_final, y_std_final = compute_label_stats(final_stats_loader, device)
        test_metrics = evaluate_model(
            model, test_loader, device, y_mean=y_mean_final, y_std=y_std_final
        )
        history["final_test_rmse"] = test_metrics["rmse"]
        history["final_test_mae"] = test_metrics["mae"]
    return history


def train_deterministic(
    model,
    loader: DataLoader,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    loss_fn: Optional[Callable],
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    subset_frac: float | None = None,
    seed: int = 271,
):
    if subset_frac is not None:
        full_dataset = loader.dataset
        full_size = len(full_dataset)
        new_size = int(subset_frac * full_size)
        g = torch.Generator().manual_seed(seed)
        subset_idx = torch.randperm(full_size, generator=g)[:new_size]
        loader = DataLoader(
            Subset(full_dataset, subset_idx),
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=getattr(loader, "num_workers", 0),
            pin_memory=getattr(loader, "pin_memory", False),
        )
    model.train()
    optimizer = torch.optim.Adam(
        model.head.parameters(), lr=lr, weight_decay=weight_decay
    )
    if loss_fn is None:
        loss_fn = nn.L1Loss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_std_safe = y_std.to(device)
            y_mean_safe = y_mean.to(device)
            targets_std = (batch_y - y_mean_safe) / y_std_safe
            preds_std = model(batch_x)
            loss = loss_fn(preds_std, targets_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(loader.dataset)
    return epoch_loss


def evaluate_deterministic(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds_std = model(batch_x)
            preds.append(preds_std)
            targets.append(batch_y)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    preds = preds * y_std + y_mean
    mse = torch.mean((preds - targets) ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(preds - targets)).item()
    return {"mse": mse, "rmse": rmse, "mae": mae}
