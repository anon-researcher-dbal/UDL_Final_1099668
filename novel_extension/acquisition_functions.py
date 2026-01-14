from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def variance_acquisition(
    model,
    pool_loader: DataLoader,
    device: torch.device,
    n_query: int = 10,
    y_std: Optional[torch.Tensor] = None,
) -> list:
    model.backbone.eval()

    all_variances = []
    pool_indices = []
    idx_counter = 0

    with torch.no_grad():
        for batch_x, _ in pool_loader:
            batch_x = batch_x.to(device)
            _, uncertainties = model.predict_with_uncertainty(batch_x)
            if y_std is not None:
                scale = y_std.to(device) ** 2
                uncertainties = uncertainties * scale
            mean_var = torch.mean(uncertainties, dim=1)
            all_variances.append(mean_var.cpu())
            batch_size = batch_x.size(0)
            pool_indices.extend(range(idx_counter, idx_counter + batch_size))
            idx_counter += batch_size
    all_variances = torch.cat(all_variances, dim=0)
    _, top_indices = torch.topk(all_variances, k=min(n_query, len(all_variances)))
    selected = [pool_indices[i] for i in top_indices.tolist()]

    return selected


def random_acquisition(
    model,
    pool_loader: DataLoader,
    device: torch.device,
    n_query: int = 10,
    y_std: Optional[torch.Tensor] = None,
) -> list:
    total_pool_size = 0
    for batch_x, _ in pool_loader:
        total_pool_size += batch_x.size(0)

    selected = list(
        np.random.choice(
            total_pool_size, size=min(n_query, total_pool_size), replace=False
        )
    )
    return selected


def max_entropy_acquisition(
    model,
    pool_loader: DataLoader,
    device: torch.device,
    n_query: int = 10,
    y_std: Optional[torch.Tensor] = None,
) -> list:
    model.backbone.eval()
    entropies = []
    pool_indices = []
    idx_counter = 0
    with torch.no_grad():
        for batch_x, _ in pool_loader:
            batch_x = batch_x.to(device)
            _, variances = model.predict_with_uncertainty(batch_x)
            if y_std is not None:
                scale = y_std.to(device) ** 2
                variances = variances * scale
            variances = torch.clamp(variances, min=1e-8)
            entropy = 0.5 * torch.sum(
                torch.log(2 * math.pi * math.e * variances), dim=1
            )
            entropies.append(entropy.cpu())
            batch_size = batch_x.size(0)
            pool_indices.extend(range(idx_counter, idx_counter + batch_size))
            idx_counter += batch_size
    entropies = torch.cat(entropies, dim=0)
    _, top_indices = torch.topk(entropies, k=min(n_query, len(entropies)))
    selected = [pool_indices[i] for i in top_indices.tolist()]
    return selected
