"""Pure loss and ridge-fit functions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.functional import mse_loss

from .model import ExplicitLyapunov
from .vsg_dynamics import BatchMeta


@dataclass
class LossConfig:
    lambda_fit: float = 1.0
    mu_plus: float = 1.0
    mu_minus: float = 1.0
    lambda_U: float = 0.0
    eps_V: float = 1e-3
    eps_dotV: float = 1e-3


def lyapunov_loss(
    model: ExplicitLyapunov,
    teacher,
    x: torch.Tensor,
    p: torch.Tensor,
    meta: BatchMeta,
    config: LossConfig,
) -> dict[str, torch.Tensor]:
    """Evaluate fit, positivity, decrease, and coefficient penalties."""

    z_teacher = teacher.scaffold(x)
    scaffold = model.scaffold_hat(x)
    fit = mse_loss(scaffold, z_teacher)
    value, derivative, x_leaf = model.V_and_Vdot(x, p, meta, teacher)
    norm_squared = x_leaf.square().sum(dim=-1)
    positive_margin = torch.clamp(config.eps_V * norm_squared - value, min=0.0)
    decrease_margin = torch.clamp(
        derivative + config.eps_dotV * norm_squared, min=0.0
    )
    positive = positive_margin.square().mean()
    decrease = decrease_margin.square().mean()
    regularization = model.U.square().sum()
    total = (
        config.lambda_fit * fit
        + config.mu_plus * positive
        + config.mu_minus * decrease
        + config.lambda_U * regularization
    )
    return {
        "total": total,
        "L_fit": fit,
        "L_plus": positive,
        "L_minus": decrease,
        "reg_U": regularization,
        "V": value,
        "dot_V": derivative,
        "hat_z": scaffold,
        "z_teacher": z_teacher,
        "x_norm_sq": norm_squared,
    }


def assert_loss_finite(losses: dict[str, torch.Tensor]) -> None:
    for name in ("total", "L_fit", "L_plus", "L_minus", "reg_U", "V", "dot_V"):
        if not torch.isfinite(losses[name]).all():
            raise FloatingPointError(f"Non-finite loss component: {name}")


def ridge_scaffold_U(
    b_matrix: torch.Tensor,
    z_target: torch.Tensor,
    lam: float = 1e-3,
) -> torch.Tensor:
    """Return the closed-form ridge solution with shape ``(latent, basis)``."""

    if b_matrix.dim() != 2 or z_target.dim() != 2:
        raise ValueError("b_matrix and z_target must both be two-dimensional.")
    basis_dim = b_matrix.shape[1]
    gram = b_matrix.transpose(0, 1) @ b_matrix
    gram = gram + float(lam) * torch.eye(
        basis_dim, dtype=gram.dtype, device=gram.device
    )
    right = b_matrix.transpose(0, 1) @ z_target
    try:
        solved = torch.linalg.solve(gram, right)
    except RuntimeError:
        solved = torch.linalg.lstsq(gram, right).solution
    return solved.transpose(0, 1).contiguous()


def fit_error_ridge(
    b_matrix: torch.Tensor,
    z_target: torch.Tensor,
    coefficients: torch.Tensor,
) -> float:
    predicted = b_matrix @ coefficients.transpose(0, 1)
    return float((z_target - predicted).square().mean().item())
