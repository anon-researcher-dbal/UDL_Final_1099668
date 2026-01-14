from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetBasis(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        full_model = models.efficientnet_b0(weights=weights)
        self.features = full_model.features
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x


class BayesianRegressionHead(ABC):
    def __init__(
        self,
        basis_dim: int,
        num_outputs: int,
        likelihood_variance: float = 1.0,
        prior_variance: float = 1.0,
        jitter: float = 1e-6,
    ):
        self.basis_dim = basis_dim
        self.num_outputs = num_outputs
        self.likelihood_variance = likelihood_variance
        self.prior_variance = prior_variance
        self.jitter = jitter
        self.posterior_mean: Optional[torch.Tensor] = None
        self.posterior_cov: Optional[torch.Tensor] = None

    @abstractmethod
    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_with_uncertainty(
        self, phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def compute_posterior(self, phi: torch.Tensor, y: torch.Tensor) -> None:
        pass


class AnalyticalBayesianHead(BayesianRegressionHead):
    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if self.posterior_mean is None:
            return phi
        return phi @ self.posterior_mean

    def predict_with_uncertainty(
        self, phi: torch.Tensor, include_likelihood_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.posterior_mean is None or self.posterior_cov is None:
            base_var = torch.zeros(phi.size(0), device=phi.device, dtype=phi.dtype)
            if include_likelihood_noise:
                base_var = base_var + self.likelihood_variance
            pred_var = base_var.unsqueeze(1).expand(-1, self.num_outputs)
            return phi, pred_var

        pred_mean = phi @ self.posterior_mean
        phi_S = phi @ self.posterior_cov
        epistemic = torch.sum(phi_S * phi, dim=1)
        total_var = (
            epistemic + self.likelihood_variance
            if include_likelihood_noise
            else epistemic
        )
        pred_var = total_var.unsqueeze(1).expand(-1, self.num_outputs)
        return pred_mean, pred_var

    @torch.no_grad()
    def compute_posterior(self, phi: torch.Tensor, y: torch.Tensor) -> None:
        sigma_inv = 1.0 / self.likelihood_variance
        prior_inv = 1.0 / self.prior_variance

        phiT_phi = phi.T @ phi
        eye = torch.eye(phi.shape[1], device=phi.device, dtype=phi.dtype)
        s_inv = sigma_inv * phiT_phi + prior_inv * eye
        s_inv = s_inv + self.jitter * eye
        s_cov = torch.linalg.inv(s_inv)

        phiT_y = phi.T @ y
        mu = s_cov @ (sigma_inv * phiT_y)

        self.posterior_cov = s_cov
        self.posterior_mean = mu


class MFVI_BayesianHead(BayesianRegressionHead):
    def __init__(
        self,
        basis_dim: int,
        num_outputs: int,
        likelihood_variance: float = 1.0,
        prior_variance: float = 1.0,
        jitter: float = 1e-6,
    ):
        super().__init__(
            basis_dim, num_outputs, likelihood_variance, prior_variance, jitter
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if self.posterior_mean is None:
            return phi
        return phi @ self.posterior_mean

    def predict_with_uncertainty(
        self, phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.posterior_mean is None or self.posterior_cov is None:
            base_var = torch.zeros(phi.size(0), device=phi.device, dtype=phi.dtype)
            base_var = base_var + self.likelihood_variance
            pred_var = base_var.unsqueeze(1).expand(-1, self.num_outputs)
            return phi, pred_var
        predictive_mean = phi @ self.posterior_mean
        phi_S = phi @ self.posterior_cov
        epistemic = torch.sum(phi_S * phi, dim=1)
        total_var = epistemic + self.likelihood_variance
        pred_var = total_var.unsqueeze(1).expand(-1, self.num_outputs)

        return predictive_mean, pred_var

    @torch.no_grad()
    def compute_posterior(self, phi: torch.Tensor, y: torch.Tensor) -> None:
        sigma2 = self.likelihood_variance
        prior_var = self.prior_variance
        D = phi.shape[1]

        phiT_phi = phi.T @ phi
        eye = torch.eye(D, device=phi.device, dtype=phi.dtype)

        s_inv = (1.0 / sigma2) * phiT_phi + (1.0 / prior_var) * eye
        s_inv = s_inv + self.jitter * eye
        s_cov = torch.linalg.inv(s_inv)

        phiT_y = phi.T @ y
        mu = s_cov @ ((1.0 / sigma2) * phiT_y)

        diag_vals = torch.diag(s_cov)
        self.posterior_cov = torch.diag(diag_vals).detach()
        self.posterior_mean = mu.detach()


class Laplace_MFVI_BayesianHead(BayesianRegressionHead):
    def __init__(
        self,
        basis_dim: int,
        num_outputs: int,
        prior_variance: float = 1.0,
        jitter: float = 1e-6,
        likelihood_scale: float = 1.0,
        learning_rate: float = 1e-3,
        num_iters: int = 1000,
        batch_size: int = 64,
        **kwargs,
    ):
        super().__init__(
            basis_dim, num_outputs, 2 * (likelihood_scale**2), prior_variance, jitter
        )
        self.likelihood_scale = likelihood_scale
        self.lr = learning_rate
        self.steps = num_iters
        self.batch_size = batch_size

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if self.posterior_mean is None:
            return phi @ self.q_mu
        return phi @ self.posterior_mean

    def predict_with_uncertainty(
        self, phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.posterior_mean is None or self.posterior_cov is None:
            base_var = torch.zeros(phi.size(0), device=phi.device, dtype=phi.dtype)
            base_var = base_var + self.likelihood_variance
            pred_var = base_var.unsqueeze(1).expand(-1, self.num_outputs)
            return phi, pred_var

        predictive_mean = phi @ self.posterior_mean
        var_w = self.posterior_cov
        phi_sq = phi**2
        epistemic_var = phi_sq @ var_w
        aleatoric_var = 2 * (self.likelihood_scale**2)
        predictive_var = epistemic_var + aleatoric_var

        return predictive_mean, predictive_var

    def _kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var)
        prior_var = self.prior_variance
        kl = 0.5 * torch.sum(
            (var / prior_var)
            + (mu**2 / prior_var)
            - 1.0
            - (log_var - math.log(prior_var))
        )
        return kl

    def compute_posterior(self, phi: torch.Tensor, y: torch.Tensor) -> None:
        device = phi.device
        N, D = phi.shape
        num_outputs = y.shape[1]

        q_mu = torch.zeros(D, num_outputs, device=device, requires_grad=True)
        with torch.no_grad():
            q_mu.normal_(0, 0.01)

        q_log_var = torch.ones(D, num_outputs, device=device, requires_grad=True)
        with torch.no_grad():
            init_log_var = math.log(self.prior_variance)
            q_log_var.fill_(init_log_var)

        optimizer = torch.optim.Adam([q_mu, q_log_var], lr=self.lr)
        dataset = torch.utils.data.TensorDataset(phi, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for step in range(self.steps):
            for batch_phi, batch_y in loader:
                optimizer.zero_grad()

                q_log_var_clamped = torch.clamp(q_log_var, min=-10, max=2)
                std = torch.exp(0.5 * q_log_var_clamped)
                epsilon = torch.randn_like(std)
                w_sample = q_mu + std * epsilon

                y_pred = batch_phi @ w_sample
                l1_loss = torch.mean(torch.abs(batch_y - y_pred))
                nll = l1_loss / self.likelihood_scale

                kl = self._kl_divergence(q_mu, q_log_var_clamped)
                loss = nll + kl / N

                loss.backward()
                torch.nn.utils.clip_grad_norm_([q_mu, q_log_var], max_norm=1.0)
                optimizer.step()

        self.posterior_mean = q_mu.detach()
        self.posterior_cov = torch.exp(q_log_var).detach()

    @torch.no_grad()
    def update_likelihood_scale(self, mae_value: float) -> None:
        new_scale = max(float(mae_value), 1e-6)
        self.likelihood_scale = new_scale
        self.likelihood_scale = min(self.likelihood_scale, 2.0)
        self.likelihood_variance = 2 * (new_scale**2)


class BayesianEfficientNetModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_outputs: int = 3,
        head_type: str = "analytical",
        likelihood_variance: float = 1.0,
        prior_variance: float = 1.0,
        likelihood_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_outputs = num_outputs
        self.likelihood_variance = likelihood_variance
        self.head_type = head_type
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            basis = backbone(dummy_input)
            basis_dim = basis.shape[1]
        if head_type == "analytical":
            self.head = AnalyticalBayesianHead(
                basis_dim=basis_dim,
                num_outputs=num_outputs,
                likelihood_variance=likelihood_variance,
                prior_variance=prior_variance,
            )
        elif head_type == "mfvi":
            self.head = MFVI_BayesianHead(
                basis_dim=basis_dim,
                num_outputs=num_outputs,
                likelihood_variance=likelihood_variance,
                prior_variance=prior_variance,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.backbone(x)
        return self.head.forward(phi)

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self.backbone(x)
        return self.head.predict_with_uncertainty(phi)

    def fit_posterior(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        y_mean: Optional[torch.Tensor] = None,
        y_std: Optional[torch.Tensor] = None,
    ) -> None:
        self.backbone.eval()
        with torch.no_grad():
            phi_list = []
            y_list = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                phi = self.backbone(batch_x)
                phi_list.append(phi)
                y_list.append(batch_y)

            phi_all = torch.cat(phi_list, dim=0)
            y_all = torch.cat(y_list, dim=0)

            if y_mean is not None and y_std is not None:
                y_all = (y_all - y_mean) / y_std

        self.head.compute_posterior(phi_all, y_all)


class DeterministicRegressor(nn.Module):
    def __init__(self, backbone: nn.Module, num_outputs: int = 3):
        super().__init__()
        self.backbone = backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            basis = self.backbone(dummy)
            basis_dim = basis.shape[1]
        self.head = nn.Linear(basis_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.backbone(x)
        return self.head(phi)
