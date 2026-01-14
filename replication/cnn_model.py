# Custom CNN architecture

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNN(nn.Module):
    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size, stride=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        pooled_h = (img_rows - 2 * kernel_size + 2) // 2
        pooled_w = (img_cols - 2 * kernel_size + 2) // 2
        self.fc1 = nn.Linear(num_filters * pooled_h * pooled_w, dense_layer)
        self.fc2 = nn.Linear(dense_layer, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


class ConvFeatureExtractor(nn.Module):
    """ConvNN backbone up to the first fully-connected layer (fc1)."""

    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size, stride=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        pooled_h = (img_rows - 2 * kernel_size + 2) // 2
        pooled_w = (img_cols - 2 * kernel_size + 2) // 2
        self.feature_dim = dense_layer
        self.fc1 = nn.Linear(num_filters * pooled_h * pooled_w, dense_layer)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return x

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


class BasisFunctionExtractor(nn.Module):
    """Feature extractor that outputs basis functions (frozen after initial training)."""

    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
        num_basis: int = 10,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size, stride=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        pooled_h = (img_rows - 2 * kernel_size + 2) // 2
        pooled_w = (img_cols - 2 * kernel_size + 2) // 2
        self.fc1 = nn.Linear(num_filters * pooled_h * pooled_w, dense_layer)
        self.dropout2 = nn.Dropout(0.5)
        self.basis_output = nn.Linear(dense_layer, num_basis)
        self.num_basis = num_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.basis_output(x)

    def freeze(self) -> None:
        """Freeze all parameters in the basis function extractor."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters in the basis function extractor."""
        for param in self.parameters():
            param.requires_grad = True


class HierarchicalBasisFunctionRegressor(nn.Module, ABC):
    """Base class for hierarchical parametrized basis function regression models."""

    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
        num_basis: int = 10,
        num_outputs: int = 10,
    ) -> None:
        super().__init__()
        self.feature_extractor = BasisFunctionExtractor(
            num_filters=num_filters,
            kernel_size=kernel_size,
            dense_layer=dense_layer,
            img_rows=img_rows,
            img_cols=img_cols,
            num_basis=num_basis,
        )
        self.num_basis = num_basis
        self.num_outputs = num_outputs

    def get_basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """Get the basis function outputs (features) for input x."""
        return self.feature_extractor(x)

    def freeze_basis_functions(self) -> None:
        """Freeze the basis function extractor."""
        self.feature_extractor.freeze()

    def unfreeze_basis_functions(self) -> None:
        """Unfreeze the basis function extractor."""
        self.feature_extractor.unfreeze()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning predictions. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return predictions and predictive variances.
        Returns:
            predictions: torch.Tensor of shape (batch_size, num_outputs)
            variances: torch.Tensor of shape (batch_size, num_outputs) or (batch_size,)
        """
        pass


class AnalyticBasisFunctionRegressor(HierarchicalBasisFunctionRegressor):
    """Analytic inference for basis function regression."""

    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
        num_basis: int = 10,
        num_outputs: int = 10,
        sigma2_likelihood: float = 1.0,
        s2_prior: float = 1.0,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__(
            num_filters=num_filters,
            kernel_size=kernel_size,
            dense_layer=dense_layer,
            img_rows=img_rows,
            img_cols=img_cols,
            num_basis=num_basis,
            num_outputs=num_outputs,
        )
        self.sigma2_likelihood = sigma2_likelihood
        self.s2_prior = s2_prior
        self.jitter = jitter
        self.posterior_cov: Optional[torch.Tensor] = None  # shape (k, k)
        self.posterior_mean: Optional[torch.Tensor] = None  # shape (k, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predictive mean using current posterior (falls back to raw basis if posterior absent)."""
        phi = self.get_basis_functions(x)
        if self.posterior_mean is None:
            return phi
        return phi @ self.posterior_mean

    def predict_with_uncertainty(
        self, x: torch.Tensor, include_likelihood_noise: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predictive mean and variance using closed-form posterior.

        If include_likelihood_noise is False, returns epistemic variance only (phi^T S phi).
        """
        phi = self.get_basis_functions(x)
        if self.posterior_mean is None or self.posterior_cov is None:
            # No posterior yet; return deterministic features with zero epistemic variance.
            base_var = torch.zeros(phi.size(0), device=phi.device, dtype=phi.dtype)
            if include_likelihood_noise:
                base_var = base_var + self.sigma2_likelihood
            pred_var = base_var.unsqueeze(1).expand(-1, self.num_outputs)
            return phi, pred_var

        pred_mean = phi @ self.posterior_mean  # shape (batch, d)
        phi_S = phi @ self.posterior_cov  # (batch, k)
        epistemic = torch.sum(phi_S * phi, dim=1)  # (batch,)
        if include_likelihood_noise:
            total_var = epistemic + self.sigma2_likelihood
        else:
            total_var = epistemic
        pred_var = total_var.unsqueeze(1).expand(-1, self.num_outputs)
        return pred_mean, pred_var

    @torch.no_grad()
    def compute_posterior(self, phi: torch.Tensor, y: torch.Tensor) -> None:
        """Compute closed-form posterior given design matrix phi (n,k) and targets y (n,d)."""
        n, k = phi.shape
        assert k == self.num_basis, "phi dimension mismatch"
        _, d = y.shape
        assert d == self.num_outputs, "target dimension mismatch"

        sigma_inv = 1.0 / self.sigma2_likelihood
        prior_inv = 1.0 / self.s2_prior

        # S_inv = sigma^{-2} phi^T phi + s^{-2} I
        phiT_phi = phi.T @ phi  # (k, k)
        eye = torch.eye(k, device=phi.device, dtype=phi.dtype)
        s_inv = sigma_inv * phiT_phi + prior_inv * eye
        # Add jitter for stability
        s_inv = s_inv + self.jitter * eye
        s_cov = torch.linalg.inv(s_inv)  # (k, k)

        # phi^T y for all outputs at once: (k, d)
        phiT_y = phi.T @ y
        mu = s_cov @ (sigma_inv * phiT_y)  # (k, d)

        self.posterior_cov = s_cov
        self.posterior_mean = mu

    @torch.no_grad()
    def compute_posterior_from_loader(
        self, loader: torch.utils.data.DataLoader, device: torch.device
    ) -> None:
        """Accumulate sufficient statistics over a dataloader and compute posterior."""
        phiT_phi_accum = None
        phiT_y_accum = None
        n_total = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            phi = self.get_basis_functions(batch_x)  # (b, k)
            if phiT_phi_accum is None:
                k = phi.shape[1]
                d = batch_y.shape[1]
                phiT_phi_accum = torch.zeros((k, k), device=device, dtype=phi.dtype)
                phiT_y_accum = torch.zeros((k, d), device=device, dtype=phi.dtype)
            phiT_phi_accum = phiT_phi_accum + phi.T @ phi
            phiT_y_accum = phiT_y_accum + phi.T @ batch_y
            n_total += batch_x.size(0)

        if phiT_phi_accum is None:
            raise ValueError("Empty loader; cannot compute posterior.")
        assert phiT_y_accum is not None

        sigma_inv = 1.0 / self.sigma2_likelihood
        prior_inv = 1.0 / self.s2_prior
        eye = torch.eye(self.num_basis, device=device, dtype=phiT_phi_accum.dtype)

        s_inv = sigma_inv * phiT_phi_accum + prior_inv * eye
        s_inv = s_inv + self.jitter * eye
        s_cov = torch.linalg.inv(s_inv)

        mu = s_cov @ (sigma_inv * phiT_y_accum)

        self.posterior_cov = s_cov
        self.posterior_mean = mu


class BayesianRegressionCNN(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # Freeze all layers except the last one
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # The last layer dim
        self.in_features = base_model.fc2.in_features
        self.out_features = 10  # 10-dim continuous labels

        # Variational parameters for q(W)
        self.mu_w = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.logvar_w = nn.Parameter(torch.randn(self.out_features, self.in_features))

    def forward(self, x):
        phi = self.feature_extractor(x).flatten(1)
        # Apply weights (mean) for the point estimate or sampling for MFVI
        return phi @ self.mu_w.T

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, include_likelihood_noise: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        phi = self.feature_extractor(x)
        mean = self.vi_head(phi)
        var_w = self.vi_head.variance()  # (out, in)
        phi_sq = phi.pow(2)  # (b, in)
        per_dim_var = torch.matmul(phi_sq, var_w.T)  # (b, out)
        if include_likelihood_noise:
            # No explicit noise term specified; keep hook for future.
            pass
        return mean, per_dim_var

    def freeze_feature_extractor(self) -> None:
        self.feature_extractor.freeze()

    def unfreeze_feature_extractor(self) -> None:
        self.feature_extractor.unfreeze()


class MFVIBasisFunctionRegressor(HierarchicalBasisFunctionRegressor):
    """Mean-field variational inference (MFVI) for basis function regression."""

    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
        num_basis: int = 10,
        num_outputs: int = 10,
    ) -> None:
        super().__init__(
            num_filters=num_filters,
            kernel_size=kernel_size,
            dense_layer=dense_layer,
            img_rows=img_rows,
            img_cols=img_cols,
            num_basis=num_basis,
            num_outputs=num_outputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MFVI inference."""
        # Placeholder - implementation details to be specified
        basis_functions = self.get_basis_functions(x)
        return basis_functions

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MFVI predictions with uncertainty."""
        raise NotImplementedError(
            "MFVI basis regressor replaced by MFVIRegressionModel"
        )
