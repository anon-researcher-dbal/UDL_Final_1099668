# Acquisition functions for Deep Bayesian Active Learning

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy import stats
from torch import nn


@torch.no_grad()
def _forward_probs(
    model: nn.Module,
    batch: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """Run a forward pass with optional MC-dropout enabled.

    Args:
        model: Trained classifier producing logits.
        batch: Input batch on the correct device.
        training: If True, keep dropout active to sample from posterior.
    """
    if training:
        model.train()
    else:
        model.eval()
    logits = model(batch)
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def predictions_from_pool(
    model: nn.Module,
    X_pool: np.ndarray,
    T: int = 100,
    training: bool = True,
    subset_size: int = 2000,
    device: torch.device | str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """MC-dropout predictions on a random subset of the pool.

    Returns stacked predictions of shape (T, subset, num_classes) and the
    corresponding subset indices (relative to the full pool).
    """
    subset_size = min(subset_size, len(X_pool))
    random_subset = np.random.choice(
        range(len(X_pool)), size=subset_size, replace=False
    )

    x_tensor = torch.from_numpy(X_pool[random_subset]).to(device)
    outputs = [
        _forward_probs(model, x_tensor, training=training).cpu().numpy()
        for _ in range(T)
    ]
    return np.stack(outputs), random_subset


def uniform(
    model: nn.Module,
    X_pool: np.ndarray,
    n_query: int = 10,
    **_: object,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly sample points from the pool."""
    n_query = min(n_query, len(X_pool))
    query_idx = np.random.choice(range(len(X_pool)), size=n_query, replace=False)
    return query_idx, X_pool[query_idx]


def shannon_entropy_function(
    model: nn.Module,
    X_pool: np.ndarray,
    T: int = 100,
    E_H: bool = False,
    training: bool = True,
    device: torch.device | str | None = None,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs, random_subset = predictions_from_pool(
        model, X_pool, T=T, training=training, device=device
    )
    pc = outputs.mean(axis=0)
    H = (-pc * np.log(pc + 1e-10)).sum(axis=-1)
    if E_H:
        E = -np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)
        return H, E, random_subset
    return H, random_subset


def max_entropy(
    model: nn.Module,
    X_pool: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    training: bool = True,
    device: torch.device | str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    acquisition, random_subset = shannon_entropy_function(
        model, X_pool, T=T, training=training, device=device
    )
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]


def bald(
    model: nn.Module,
    X_pool: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    training: bool = True,
    device: torch.device | str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    H, E_H, random_subset = shannon_entropy_function(
        model, X_pool, T=T, E_H=True, training=training, device=device
    )
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]


def var_ratios(
    model: nn.Module,
    X_pool: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    training: bool = True,
    device: torch.device | str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    outputs, random_subset = predictions_from_pool(
        model, X_pool, T=T, training=training, device=device
    )
    preds = np.argmax(outputs, axis=2)
    # stats.mode returns (mode, count)
    _, count = stats.mode(preds, axis=0, keepdims=False)
    acquisition = (1 - count / preds.shape[0]).reshape((-1,))
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]


def mean_std(
    model: nn.Module,
    X_pool: np.ndarray,
    n_query: int = 10,
    T: int = 100,
    training: bool = True,
    device: torch.device | str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    outputs, random_subset = predictions_from_pool(
        model, X_pool, T=T, training=training, device=device
    )
    sigma_c = np.std(outputs, axis=0)
    acquisition = np.mean(sigma_c, axis=-1)
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]


@torch.no_grad()
def variance_regression(
    model: nn.Module,
    X_pool: np.ndarray,
    n_query: int = 10,
    batch_size: int = 256,
    device: torch.device | str | None = None,
    include_likelihood_noise: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select points with largest predictive variance for regression models.

    Assumes model exposes predict_with_uncertainty(x, include_likelihood_noise) -> (mean, var).
    For analytic regressor, set include_likelihood_noise=False to use epistemic variance only.
    """

    if device is None:
        device = torch.device("cpu")

    model.eval()
    X_pool_t = torch.from_numpy(X_pool)
    variances = []
    indices = []

    # Batched evaluation to avoid memory blow-up
    for start in range(0, len(X_pool_t), batch_size):
        end = min(start + batch_size, len(X_pool_t))
        batch = X_pool_t[start:end].to(device)
        _, pred_var = model.predict_with_uncertainty(
            batch, include_likelihood_noise=include_likelihood_noise
        )
        # pred_var shape: (b, d); take mean over outputs (identical if model uses shared variance)
        var_scalar = pred_var.mean(dim=1).detach().cpu()
        variances.append(var_scalar)
        indices.append(torch.arange(start, end))

    variances = torch.cat(variances)
    indices = torch.cat(indices)

    # Top-k by variance
    topk = min(n_query, variances.numel())
    _, top_idx = torch.topk(variances, k=topk)
    query_idx = indices[top_idx].numpy()
    return query_idx, X_pool[query_idx]


@torch.no_grad()
def predictive_variance_query(
    feature_extractor: nn.Module,
    vi_head: nn.Module,
    X_pool: np.ndarray,
    n_query: int,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select samples with largest phi^T Var(W) phi using MFVI head."""

    feature_extractor.eval()
    vi_head.eval()
    x_t = torch.from_numpy(X_pool).float()
    variances: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []

    var_w = vi_head.variance().to(device)  # (out, in)

    for start in range(0, len(x_t), batch_size):
        end = min(start + batch_size, len(x_t))
        batch = x_t[start:end].to(device)
        phi = feature_extractor(batch).flatten(1)
        phi_sq = phi.pow(2)
        per_dim_var = torch.matmul(phi_sq, var_w.T)
        total_var = per_dim_var.sum(dim=1)
        variances.append(total_var.cpu())
        indices.append(torch.arange(start, end))

    variances_cat = torch.cat(variances)
    indices_cat = torch.cat(indices)
    topk = min(n_query, variances_cat.numel())
    _, top_idx = torch.topk(variances_cat, k=topk)
    query_idx = indices_cat[top_idx].numpy()
    return query_idx, X_pool[query_idx]
