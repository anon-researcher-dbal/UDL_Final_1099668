from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _accuracy(
    model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device
) -> float:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        yb = torch.from_numpy(y).long().to(device)
        preds = torch.argmax(model(xb), dim=1)
        return float((preds == yb).float().mean().cpu().item())


def one_hot_labels(y_labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    y_labels = y_labels.astype(int)
    y_one_hot = np.zeros((len(y_labels), num_classes), dtype=np.float32)
    y_one_hot[np.arange(len(y_labels)), y_labels] = 1.0
    return y_one_hot


def _mse_rmse(
    model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        y_one_hot = (
            torch.from_numpy(one_hot_labels(y.astype(int), num_classes=10))
            .float()
            .to(device)
        )
        logits = model(xb)
        mse = torch.mean((logits - y_one_hot) ** 2).item()
        rmse = float(np.sqrt(mse))
        return mse, rmse


def pretrainer(
    backbone_model,
    X_pretrain: np.ndarray,
    Y_pretrain: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    batch_size: int = 128,
    task: str = "classification",
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    val_batch_size: int = 256,
) -> Tuple:
    opt = torch.optim.Adam(
        list(backbone_model.parameters()), lr=lr, weight_decay=weight_decay
    )
    if task == "classification":
        crit = nn.CrossEntropyLoss()
    else:
        crit = nn.MSELoss()
    backbone_model.to(device).train()

    if task == "classification":
        y_labels = Y_pretrain.astype(int)
        loader = _make_loader(X_pretrain, y_labels, batch_size, shuffle=True)
    else:
        y_labels = Y_pretrain.astype(int)
        y_one_hot = np.zeros((len(y_labels), 10), dtype=np.float32)
        y_one_hot[np.arange(len(y_labels)), y_labels] = 1.0
        dataset = TensorDataset(
            torch.from_numpy(X_pretrain).float(), torch.from_numpy(y_one_hot).float()
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            preds = backbone_model(xb)
            loss = crit(preds, yb)
            loss.backward()
            opt.step()

    backbone_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = backbone_model(xb)
            if task == "classification":
                yb_labels = yb.to(device)
                pred_labels = torch.argmax(preds, dim=1)
                correct += (pred_labels == yb_labels).sum().item()
            else:
                yb = yb.to(device)
                pred_labels = torch.argmax(preds, dim=1)
                true_labels = torch.argmax(yb, dim=1)
                correct += (pred_labels == true_labels).sum().item()
            total += xb.size(0)
    pretrain_acc = correct / total

    val_acc = float("nan")
    val_mse = float("nan")
    if X_val is not None and y_val is not None:
        if task == "classification":
            y_val_labels = y_val.astype(int)
            val_loader = _make_loader(
                X_val, y_val_labels, val_batch_size, shuffle=False
            )
            y_val_one_hot = np.zeros((len(y_val_labels), 10), dtype=np.float32)
            y_val_one_hot[np.arange(len(y_val_labels)), y_val_labels] = 1.0
            y_val_one_hot_t = torch.from_numpy(y_val_one_hot).float().to(device)
        else:
            y_val_labels = y_val.astype(int)
            y_val_one_hot = np.zeros((len(y_val_labels), 10), dtype=np.float32)
            y_val_one_hot[np.arange(len(y_val_labels)), y_val_labels] = 1.0
            dataset = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val_one_hot).float(),
            )
            val_loader = DataLoader(dataset, batch_size=val_batch_size, shuffle=False)
            y_val_one_hot_t = None

        correct_val = 0
        total_val = 0
        mse_accum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                if task == "classification":
                    yb_labels = yb.to(device)
                    logits = backbone_model(xb)
                    pred_labels = torch.argmax(logits, dim=1)
                    correct_val += (pred_labels == yb_labels).sum().item()
                    if y_val_one_hot_t is not None:
                        start = total_val
                        end = total_val + xb.size(0)
                        target_slice = y_val_one_hot_t[start:end]
                        probs = torch.softmax(logits, dim=1)
                        mse_accum += torch.sum((probs - target_slice) ** 2).item()
                else:
                    logits = backbone_model(xb)
                    yb = yb.to(device)
                    pred_labels = torch.argmax(logits, dim=1)
                    true_labels = torch.argmax(yb, dim=1)
                    correct_val += (pred_labels == true_labels).sum().item()
                    probs = torch.softmax(logits, dim=1)
                    mse_accum += torch.sum((probs - yb) ** 2).item()
                total_val += xb.size(0)
        val_acc = correct_val / total_val if total_val > 0 else float("nan")
        val_mse = mse_accum / total_val if total_val > 0 else float("nan")
    return backbone_model, pretrain_acc, val_acc, val_mse


def init_head_posterior(
    model: torch.nn.Module,
    X_init: np.ndarray,
    y_init: np.ndarray,
    device: torch.device,
    num_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    X_labeled = X_init.copy()
    y_labeled_oh = one_hot_labels(y_init, num_classes=num_classes)
    x_t = torch.from_numpy(X_labeled).float().to(device)
    y_t = torch.from_numpy(y_labeled_oh).float().to(device)
    model.compute_posterior(x=x_t, y=y_t)
    return X_labeled, y_labeled_oh


def select_queries(
    model: torch.nn.Module,
    X_pool_run: np.ndarray,
    n_query: int,
    acq_fn: str = "predictive",
) -> np.ndarray:
    n_query = min(n_query, len(X_pool_run))
    if n_query <= 0:
        return np.array([], dtype=int)
    if acq_fn == "predictive":
        variances = model.compute_predictive_variance(X_pool_run)
        return np.argsort(-variances)[:n_query]
    return np.random.choice(range(len(X_pool_run)), size=n_query, replace=False)


def update_sets_after_query(
    X_labeled: np.ndarray,
    y_labeled_oh: np.ndarray,
    X_pool_run: np.ndarray,
    y_pool_oh_run: np.ndarray,
    query_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if query_idx.size == 0:
        return X_labeled, y_labeled_oh, X_pool_run, y_pool_oh_run
    X_labeled = np.concatenate([X_labeled, X_pool_run[query_idx]], axis=0)
    y_labeled_oh = np.concatenate([y_labeled_oh, y_pool_oh_run[query_idx]], axis=0)
    X_pool_run = np.delete(X_pool_run, query_idx, axis=0)
    y_pool_oh_run = np.delete(y_pool_oh_run, query_idx, axis=0)
    return X_labeled, y_labeled_oh, X_pool_run, y_pool_oh_run


def recompute_posterior(
    model: torch.nn.Module,
    X_labeled: np.ndarray,
    y_labeled_oh: np.ndarray,
    device: torch.device,
) -> None:
    x_t = torch.from_numpy(X_labeled).float().to(device)
    y_t = torch.from_numpy(y_labeled_oh).float().to(device)
    model.compute_posterior(x=x_t, y=y_t)


def acquisition_round(
    model: torch.nn.Module,
    X_labeled: np.ndarray,
    y_labeled_oh: np.ndarray,
    X_pool_run: np.ndarray,
    y_pool_oh_run: np.ndarray,
    n_query: int,
    acq_fn: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query_idx = select_queries(model, X_pool_run, n_query=n_query, acq_fn=acq_fn)
    X_labeled, y_labeled_oh, X_pool_run, y_pool_oh_run = update_sets_after_query(
        X_labeled, y_labeled_oh, X_pool_run, y_pool_oh_run, query_idx
    )
    recompute_posterior(model, X_labeled, y_labeled_oh, device)
    return X_labeled, y_labeled_oh, X_pool_run, y_pool_oh_run, query_idx
