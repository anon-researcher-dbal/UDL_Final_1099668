# Active learning utilities

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from replication.acquisition_functions import (
    bald,
    max_entropy,
    mean_std,
    predictive_variance_query,
    uniform,
    var_ratios,
    variance_regression,
)
from replication.cnn_model import (
    AnalyticBasisFunctionRegressor,
    MFVIBasisFunctionRegressor,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _make_regression_loader(
    X: np.ndarray,
    y_onehot: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader for regression targets (one-hot float)."""
    dataset = TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(y_onehot).float()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_model(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model


def _train_extractor_mse(
    model: AnalyticBasisFunctionRegressor,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> AnalyticBasisFunctionRegressor:
    """Train only the basis function extractor with MSE loss against one-hot targets."""
    model.to(device)
    # Optimize only extractor params
    optimizer = torch.optim.Adam(
        model.feature_extractor.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    model.feature_extractor.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model.get_basis_functions(xb)  # (B, k)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    return model


def _pretrain_feature_extractor_mse(
    feature_extractor: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    num_outputs: int = 10,
) -> nn.Module:
    """Pretrain feature extractor with a lightweight linear probe using MSE."""
    probe = nn.Linear(getattr(feature_extractor, "feature_dim"), num_outputs).to(device)
    params = list(feature_extractor.parameters()) + list(probe.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    feature_extractor.to(device)
    feature_extractor.train()
    probe.train()

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            phi = feature_extractor(xb)
            preds = probe(phi)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    return feature_extractor


def _train_vi_head(
    feature_extractor: nn.Module,
    vi_head: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    beta: float,
    device: torch.device,
    elbo_fn=None,
) -> nn.Module:
    """Train MFVI head with frozen feature extractor."""
    if elbo_fn is None:
        raise ValueError("elbo_fn must be provided for MFVI training")
    optimizer = torch.optim.Adam(vi_head.parameters(), lr=lr)
    feature_extractor.to(device).eval()
    vi_head.to(device).train()

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                phi = feature_extractor(xb)
            preds = vi_head(phi)
            loss = elbo_fn(preds, yb, vi_head, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return vi_head


def _accuracy(
    model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device
) -> float:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        yb = torch.from_numpy(y).long().to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        return float((preds == yb).float().mean().cpu().item())


def _regression_accuracy(
    model: AnalyticBasisFunctionRegressor,
    X: np.ndarray,
    y_onehot: np.ndarray,
    device: torch.device,
) -> float:
    """Compute accuracy by comparing argmax of predictive mean to one-hot targets."""
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        yb = torch.from_numpy(y_onehot).float().to(device)
        pred_mean, _ = model.predict_with_uncertainty(xb, include_likelihood_noise=True)
        preds = torch.argmax(pred_mean, dim=1)
        true = torch.argmax(yb, dim=1)
        return float((preds == true).float().mean().cpu().item())


def _mfvi_accuracy(
    feature_extractor: nn.Module,
    vi_head: nn.Module,
    X: np.ndarray,
    y_onehot: np.ndarray,
    device: torch.device,
) -> float:
    feature_extractor.eval()
    vi_head.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        yb = torch.from_numpy(y_onehot).float().to(device)
        phi = feature_extractor(xb)
        preds = vi_head(phi)
        pred_labels = torch.argmax(preds, dim=1)
        true_labels = torch.argmax(yb, dim=1)
        return float((pred_labels == true_labels).float().mean().cpu().item())


def active_learning_procedure(
    query_strategy: Callable,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_init: np.ndarray,
    y_init: np.ndarray,
    build_model: Callable[[], nn.Module],
    *,
    T: int = 100,
    n_query: int = 10,
    training: bool = True,
    batch_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    device: torch.device,
) -> Tuple[List[float], float]:
    """Run the active learning loop.

    Returns a validation accuracy history (first element is initial test accuracy
    like the original code) and the final test accuracy.
    """

    # Initial training on the balanced seed set
    model = build_model().to(device)
    train_loader = _make_loader(X_init, y_init, batch_size, shuffle=True)
    model = _train_model(
        model,
        train_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    perf_hist: List[float] = [_accuracy(model, X_test, y_test, device=device)]

    for index in range(T):
        # Query a subset from the remaining pool
        query_idx, _ = query_strategy(
            model,
            X_pool,
            n_query=n_query,
            T=T,
            training=training,
            device=device,
        )

        # Update labeled and pool sets
        X_train = np.concatenate([X_init, X_pool[query_idx]], axis=0)
        y_train = np.concatenate([y_init, y_pool[query_idx]], axis=0)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        # Re-train a fresh model on the expanded labeled set
        model = build_model().to(device)
        train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
        model = _train_model(
            model,
            train_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

        val_acc = _accuracy(model, X_val, y_val, device=device)
        if (index + 1) % 5 == 0:
            print(f"Val Accuracy after query {index + 1}: {val_acc:0.4f}")
        perf_hist.append(val_acc)

        # Persist the expanded labeled set for next iteration
        X_init, y_init = X_train, y_train

    final_test_acc = _accuracy(model, X_test, y_test, device=device)
    print(f"********** Test Accuracy per experiment: {final_test_acc:.4f} **********")
    return perf_hist, final_test_acc


def active_learning_regression_procedure(
    X_val: np.ndarray,
    y_val_onehot: np.ndarray,
    X_test: np.ndarray,
    y_test_onehot: np.ndarray,
    X_pool: np.ndarray,
    y_pool_onehot: np.ndarray,
    X_init: np.ndarray,
    y_init_onehot: np.ndarray,
    build_model: Callable[[], AnalyticBasisFunctionRegressor],
    *,
    T: int = 20,
    n_query: int = 10,
    batch_size: int = 128,
    n_extractor_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    device: torch.device,
) -> Tuple[List[float], float]:
    """Active learning loop for regression with analytic BLR last layer.

    Workflow per round:
    1) Fresh model → train extractor with MSE on labeled set for n_extractor_epochs.
    2) Freeze extractor → compute posterior from labeled loader.
    3) Acquire top-n by epistemic variance from pool.
    4) Expand labeled set; repeat.
    Returns val accuracy history and final test accuracy.
    """

    # Initial extractor training
    model = build_model().to(device)
    labeled_loader = _make_regression_loader(
        X_init, y_init_onehot, batch_size, shuffle=True
    )
    model = _train_extractor_mse(
        model,
        labeled_loader,
        epochs=n_extractor_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )
    # Freeze and fit posterior
    model.freeze_basis_functions()
    model.compute_posterior_from_loader(labeled_loader, device=device)

    perf_hist: List[float] = [
        _regression_accuracy(model, X_test, y_test_onehot, device=device)
    ]

    for index in range(T):
        # Acquisition based on epistemic variance
        query_idx, _ = variance_regression(
            model,
            X_pool,
            n_query=n_query,
            batch_size=batch_size,
            device=device,
            include_likelihood_noise=False,
        )

        # Update labeled and pool sets
        X_train = np.concatenate([X_init, X_pool[query_idx]], axis=0)
        y_train = np.concatenate([y_init_onehot, y_pool_onehot[query_idx]], axis=0)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool_onehot = np.delete(y_pool_onehot, query_idx, axis=0)

        # Fresh model per round, per your spec
        model = build_model().to(device)
        labeled_loader = _make_regression_loader(
            X_train, y_train, batch_size, shuffle=True
        )
        model = _train_extractor_mse(
            model,
            labeled_loader,
            epochs=n_extractor_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        model.freeze_basis_functions()
        model.compute_posterior_from_loader(labeled_loader, device=device)

        val_acc = _regression_accuracy(model, X_val, y_val_onehot, device=device)
        if (index + 1) % 5 == 0:
            print(f"Val Accuracy after query {index + 1}: {val_acc:0.4f}")
        perf_hist.append(val_acc)

        # Persist labeled set
        X_init, y_init_onehot = X_train, y_train

    final_test_acc = _regression_accuracy(model, X_test, y_test_onehot, device=device)
    print(f"********** Test Accuracy (regression BLR): {final_test_acc:.4f} **********")
    return perf_hist, final_test_acc


def active_learning_mfvi_procedure(
    X_val: np.ndarray,
    y_val_onehot: np.ndarray,
    X_test: np.ndarray,
    y_test_onehot: np.ndarray,
    X_pool: np.ndarray,
    y_pool_onehot: np.ndarray,
    X_init: np.ndarray,
    y_init_onehot: np.ndarray,
    build_model: Callable[[], MFVIBasisFunctionRegressor],
    *,
    T: int = 20,
    n_query: int = 10,
    batch_size: int = 128,
    n_extractor_epochs: int = 10,
    vi_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    beta: float = 1e-4,
    device: torch.device,
) -> Tuple[List[float], float]:
    """Active learning with MFVI linear head and frozen ConvNN backbone."""

    # Initial pre-train of the backbone using labeled set
    model = build_model().to(device)
    labeled_loader = _make_regression_loader(
        X_init, y_init_onehot, batch_size, shuffle=True
    )
    model.feature_extractor = _pretrain_feature_extractor_mse(
        model.feature_extractor,
        labeled_loader,
        epochs=n_extractor_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        num_outputs=y_init_onehot.shape[1],
    )
    model.freeze_feature_extractor()
    model.vi_head = _train_vi_head(
        model.feature_extractor,
        model.vi_head,
        labeled_loader,
        epochs=vi_epochs,
        lr=lr,
        beta=beta,
        device=device,
        elbo_fn=None,
    )

    perf_hist: List[float] = [
        _mfvi_accuracy(
            model.feature_extractor, model.vi_head, X_test, y_test_onehot, device=device
        )
    ]

    for index in range(T):
        # Acquisition via MFVI predictive variance
        query_idx, _ = predictive_variance_query(
            model.feature_extractor,
            model.vi_head,
            X_pool,
            n_query=n_query,
            device=device,
            batch_size=batch_size,
        )

        X_train = np.concatenate([X_init, X_pool[query_idx]], axis=0)
        y_train = np.concatenate([y_init_onehot, y_pool_onehot[query_idx]], axis=0)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool_onehot = np.delete(y_pool_onehot, query_idx, axis=0)

        # Fresh model each round with updated labeled set
        model = build_model().to(device)
        labeled_loader = _make_regression_loader(
            X_train, y_train, batch_size, shuffle=True
        )
        model.feature_extractor = _pretrain_feature_extractor_mse(
            model.feature_extractor,
            labeled_loader,
            epochs=n_extractor_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            num_outputs=y_train.shape[1],
        )
        model.freeze_feature_extractor()
        model.vi_head = _train_vi_head(
            model.feature_extractor,
            model.vi_head,
            labeled_loader,
            epochs=vi_epochs,
            lr=lr,
            beta=beta,
            device=device,
            elbo_fn=elbo_loss,
        )

        val_acc = _mfvi_accuracy(
            model.feature_extractor, model.vi_head, X_val, y_val_onehot, device=device
        )
        if (index + 1) % 5 == 0:
            print(f"Val Accuracy after query {index + 1}: {val_acc:0.4f}")
        perf_hist.append(val_acc)

        X_init, y_init_onehot = X_train, y_train

    final_test_acc = _mfvi_accuracy(
        model.feature_extractor, model.vi_head, X_test, y_test_onehot, device=device
    )
    print(
        f"********** Test Accuracy (regression MFVI): {final_test_acc:.4f} **********"
    )
    return perf_hist, final_test_acc


def select_acq_function(acq_func: int = 0) -> List[Callable]:
    """Return acquisition strategies based on selection flag.

    acq_func: 0-all, 1-uniform, 2-max_entropy, 3-bald, 4-var_ratios, 5-mean_std
    """
    acq_func_dict: Dict[int, List[Callable]] = {
        0: [uniform, max_entropy, bald, var_ratios, mean_std],
        1: [uniform],
        2: [max_entropy],
        3: [bald],
        4: [var_ratios],
        5: [mean_std],
    }
    return acq_func_dict[acq_func]
