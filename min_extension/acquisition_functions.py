from __future__ import annotations

from typing import Tuple

import numpy as np
from torch import nn


def uniform_acq(
    model: nn.Module, X_pool: np.ndarray, n_query: int = 10, **_
) -> Tuple[np.ndarray, np.ndarray]:
    n_query = min(n_query, len(X_pool))
    query_idx = np.random.choice(range(len(X_pool)), size=n_query, replace=False)
    return query_idx, X_pool[query_idx]


def predictive_variance_acq(
    model: nn.Module, X_pool: np.ndarray, n_query: int = 10, **_
) -> Tuple[np.ndarray, np.ndarray]:
    n_query = min(n_query, len(X_pool))
    variances = model.compute_predictive_variance(X_pool)
    query_idx = np.argsort(-variances)[:n_query]
    return query_idx, X_pool[query_idx]
