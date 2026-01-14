from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np
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


class ConvNNBackbone(nn.Module):
    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        img_rows: int = 28,
        img_cols: int = 28,
        dense_layer: int = 128,
        use_fc2_features: bool = False,
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
        self.use_fc2_features = use_fc2_features
        self.feature_dim = 10 if use_fc2_features else dense_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        if self.use_fc2_features:
            return self.fc2(x)
        else:
            return x

    def get_fc1_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def freeze_except_fc2(self) -> None:
        for name, p in self.named_parameters():
            if "fc2" not in name:
                p.requires_grad = False
            else:
                p.requires_grad = True


class HierarchicalBayesModel(nn.Module):
    def __init__(
        self,
        feature_extractor: Optional[ConvNNBackbone],
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
        num_outputs: int = 10,
        use_fc2: bool = False,
    ) -> None:
        super().__init__()
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = ConvNNBackbone(
                num_filters=num_filters,
                kernel_size=kernel_size,
                dense_layer=dense_layer,
                img_rows=img_rows,
                img_cols=img_cols,
                use_fc2_features=use_fc2,
            )
        self.num_outputs = num_outputs
        self.use_fc2 = use_fc2

    def get_basis_func(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fc2:
            return self.feature_extractor(x)
        else:
            return self.feature_extractor.get_fc1_features(x)

    def freeze_basis_functions(self) -> None:
        self.feature_extractor.freeze()

    def unfreeze_basis_functions(self) -> None:
        self.feature_extractor.unfreeze()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def compute_predictive_variance(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Analytical_HB(HierarchicalBayesModel):
    def __init__(
        self,
        feature_extractor: Optional[ConvNNBackbone],
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
        num_outputs: int = 10,
        likelihood_variance: float = 1.0,
        prior_variance: float = 1.0,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__(
            feature_extractor=feature_extractor,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dense_layer=dense_layer,
            img_rows=img_rows,
            img_cols=img_cols,
            num_outputs=num_outputs,
        )
        self.likelihood_variance = likelihood_variance
        self.prior_variance = prior_variance
        self.jitter = jitter
        self.posterior_cov: Optional[torch.Tensor] = None
        self.posterior_mean: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.get_basis_func(x)
        if self.posterior_mean is None:
            return phi
        return phi @ self.posterior_mean

    def predict_with_uncertainty(
        self, x: torch.Tensor, include_likelihood_noise: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phi = self.get_basis_func(x)
        if self.posterior_mean is None or self.posterior_cov is None:
            base_var = torch.zeros(phi.size(0), device=phi.device, dtype=phi.dtype)
            if include_likelihood_noise:
                base_var = base_var + self.likelihood_variance
            pred_var = base_var.unsqueeze(1).expand(-1, self.num_outputs)
            return phi, pred_var
        pred_mean = phi @ self.posterior_mean
        phi_S = phi @ self.posterior_cov
        epistemic = torch.sum(phi_S * phi, dim=1)
        if include_likelihood_noise:
            total_var = epistemic + self.likelihood_variance
        else:
            total_var = epistemic
        pred_var = total_var.unsqueeze(1).expand(-1, self.num_outputs)
        return pred_mean, pred_var

    @torch.no_grad()
    def compute_posterior(self, x: torch.Tensor, y: torch.Tensor) -> None:
        sigma_inv = 1.0 / self.likelihood_variance
        prior_inv = 1.0 / self.prior_variance
        phi = self.get_basis_func(x)
        phiT_phi = phi.T @ phi
        eye = torch.eye(phi.shape[1], device=phi.device, dtype=phi.dtype)
        s_inv = sigma_inv * phiT_phi + prior_inv * eye
        s_inv = s_inv + self.jitter * eye
        s_cov = torch.linalg.inv(s_inv)
        phiT_y = phi.T @ y
        mu = s_cov @ (sigma_inv * phiT_y)
        self.posterior_cov = s_cov
        self.posterior_mean = mu

    @torch.no_grad()
    def compute_posterior_from_loader(
        self, loader: torch.utils.data.DataLoader, device: torch.device
    ) -> None:
        phiT_phi_accum = None
        phiT_y_accum = None
        n_total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            phi = self.get_basis_func(batch_x)
            if phiT_phi_accum is None:
                k = phi.shape[1]
                d = batch_y.shape[1]
                phiT_phi_accum = torch.zeros((k, k), device=device, dtype=phi.dtype)
                phiT_y_accum = torch.zeros((k, d), device=device, dtype=phi.dtype)
            phiT_phi_accum = phiT_phi_accum + phi.T @ phi
            phiT_y_accum = phiT_y_accum + phi.T @ batch_y
            n_total += batch_x.size(0)
        assert phiT_y_accum is not None
        sigma_inv = 1.0 / self.likelihood_variance
        prior_inv = 1.0 / self.prior_variance
        eye = torch.eye(k, device=device, dtype=phiT_phi_accum.dtype)
        s_inv = sigma_inv * phiT_phi_accum + prior_inv * eye
        s_inv = s_inv + self.jitter * eye
        s_cov = torch.linalg.inv(s_inv)
        mu = s_cov @ (sigma_inv * phiT_y_accum)
        self.posterior_cov = s_cov
        self.posterior_mean = mu

    @torch.no_grad()
    def compute_predictive_variance(self, X: np.ndarray) -> np.ndarray:
        dev = next(self.feature_extractor.parameters()).device
        x_tensor = torch.from_numpy(X).float().to(dev)
        phi = self.get_basis_func(x_tensor)
        if self.posterior_mean is None or self.posterior_cov is None:
            variances = np.zeros(phi.size(0))
            return variances
        var_w = self.posterior_cov
        phi_S = phi @ var_w
        epistemic_var = torch.sum(phi_S * phi, dim=1)
        total_var = epistemic_var + self.likelihood_variance
        return total_var.detach().cpu().numpy()


class MFVI_HB(HierarchicalBayesModel):
    def __init__(
        self,
        feature_extractor: Optional[ConvNNBackbone],
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
        num_outputs: int = 10,
        likelihood_variance: float = 1.0,
        prior_variance: float = 1.0,
        jitter: float = 1e-6,
        elbo_lr: float = 1e-3,
        vi_method: str = "closed",
        **kwargs,
    ) -> None:
        super().__init__(
            feature_extractor=feature_extractor,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dense_layer=dense_layer,
            img_rows=img_rows,
            img_cols=img_cols,
            num_outputs=num_outputs,
        )
        self.likelihood_variance = likelihood_variance
        self.prior_variance = prior_variance
        self.jitter = jitter
        self.elbo_lr = elbo_lr
        self.vi_method = vi_method
        self.posterior_cov: Optional[torch.Tensor] = None
        self.posterior_mean: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.get_basis_func(x)
        if self.posterior_mean is None:
            return phi
        return phi @ self.posterior_mean

    def _compute_A_k(self, phi: torch.Tensor) -> torch.Tensor:
        return torch.sum(phi**2, dim=0)

    def _update_M_closed_form(
        self, phi: torch.Tensor, y: torch.Tensor, sigma2: float
    ) -> torch.Tensor:
        k = phi.shape[1]
        device = phi.device
        dtype = phi.dtype
        phiT_phi = phi.T @ phi
        phiT_y = phi.T @ y
        eye = torch.eye(k, device=device, dtype=dtype)
        lambda_reg = (sigma2 / self.prior_variance) * eye
        M = torch.linalg.solve(phiT_phi + lambda_reg, phiT_y)
        return M

    def _update_variances_closed_form(
        self, A_k: torch.Tensor, sigma2: float
    ) -> torch.Tensor:
        inv_term = (A_k / sigma2) + (1.0 / self.prior_variance)
        var_diag = 1.0 / inv_term
        S = torch.diag(var_diag)
        return S

    def _update_sigma2(
        self, y: torch.Tensor, phi: torch.Tensor, M: torch.Tensor, S: torch.Tensor
    ) -> float:
        n, d = y.shape
        residuals = y - phi @ M
        Res = torch.sum(residuals**2)
        S_diag = torch.diag(S)
        A_k = self._compute_A_k(phi)
        V = torch.sum(A_k * S_diag)
        sigma2_new = float((Res + V) / (n * d))
        return max(sigma2_new, 1e-8)

    def _compute_posterior_coord_ascent(
        self, phi: torch.Tensor, y: torch.Tensor, max_iters: int = 20
    ) -> None:
        max_iterations = max_iters
        tolerance = 1e-4
        sigma2 = self.likelihood_variance
        for iter in range(max_iterations):
            M = self._update_M_closed_form(phi, y, sigma2)
            A_k = self._compute_A_k(phi)
            S = self._update_variances_closed_form(A_k, sigma2)
            sigma2_new = self._update_sigma2(y, phi, M, S)
            sigma2_change = abs(sigma2_new - sigma2)
            if sigma2_change < tolerance:
                break
            sigma2 = sigma2_new
        self.posterior_mean = M.detach()
        self.posterior_cov = S.detach()

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phi = self.get_basis_func(x)
        if self.posterior_mean is None or self.posterior_cov is None:
            base_var = torch.zeros(phi.size(0), device=phi.device, dtype=phi.dtype)
            base_var = base_var + self.likelihood_variance
            pred_var = base_var.unsqueeze(1).expand(-1, self.num_outputs)
            return phi, pred_var
        predictive_mean = phi @ self.posterior_mean
        var_w = self.posterior_cov
        phi_S = phi @ var_w
        epistemic_var = torch.sum(phi_S * phi, dim=1)
        per_dim_var = self.likelihood_variance + epistemic_var
        predictive_var = per_dim_var.unsqueeze(1).expand(-1, self.num_outputs)
        return predictive_mean, predictive_var

    def compute_posterior(self, x: torch.Tensor, y: torch.Tensor) -> None:
        phi = self.get_basis_func(x)

        if self.vi_method == "closed":
            sigma2 = self.likelihood_variance
            M = self._update_M_closed_form(phi, y, sigma2)
            A_k = self._compute_A_k(phi)
            S = self._update_variances_closed_form(A_k, sigma2)
            self.posterior_mean = M.detach()
            self.posterior_cov = S.detach()
            return

        if self.vi_method == "coord":
            self._compute_posterior_coord_ascent(phi, y)
            return

        print("ELBO calculation failed - falling back to gradient descent")
        k = phi.shape[1]
        d = y.shape[1]
        device = phi.device
        dtype = phi.dtype

        mu = torch.zeros((k, d), device=device, dtype=dtype, requires_grad=True)
        log_var_diag = torch.zeros((k,), device=device, dtype=dtype, requires_grad=True)

        optimizer = torch.optim.Adam([mu, log_var_diag], lr=self.elbo_lr)
        sigma_inv = 1.0 / self.likelihood_variance
        prior_inv = 1.0 / self.prior_variance

        for _ in range(50):
            var_diag = torch.exp(log_var_diag)
            eps = torch.randn((k, d), device=device, dtype=dtype)
            w_sample = mu + torch.sqrt(var_diag).unsqueeze(1) * eps

            y_pred = phi @ w_sample
            residuals = y - y_pred

            norm_const = (
                -0.5 * d * phi.size(0) * np.log(2 * np.pi * self.likelihood_variance)
            )
            mse_term = -0.5 * sigma_inv * torch.sum(residuals**2)
            var_contrib = torch.sum((phi @ torch.diag(var_diag)) * phi)
            trace_term = -0.5 * sigma_inv * var_contrib
            exp_log_lik = norm_const + mse_term + trace_term

            kl_var = (
                0.5
                * d
                * torch.sum(-torch.log(var_diag + 1e-8) - 1.0 + prior_inv * var_diag)
            )
            kl_mu = 0.5 * prior_inv * torch.sum(mu**2)
            kl_div = kl_var + kl_mu

            elbo = exp_log_lik - kl_div
            loss = -elbo / phi.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.posterior_mean = mu.detach()
        self.posterior_cov = torch.diag(torch.exp(log_var_diag.detach()))

    @torch.no_grad()
    def compute_predictive_variance(self, X: np.ndarray) -> np.ndarray:
        dev = next(self.feature_extractor.parameters()).device
        x_tensor = torch.from_numpy(X).float().to(dev)
        phi = self.get_basis_func(x_tensor)

        if self.posterior_mean is None or self.posterior_cov is None:
            variances = np.zeros(phi.size(0))
            return variances

        var_w = self.posterior_cov
        phi_S = phi @ var_w
        epistemic_var = torch.sum(phi_S * phi, dim=1)
        total_var = epistemic_var + self.likelihood_variance

        return total_var.detach().cpu().numpy()
