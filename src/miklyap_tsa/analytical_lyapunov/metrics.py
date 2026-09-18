"""Core analytical-basis and Lyapunov metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .basis import BasisRegistry


def _safe_log10(value: float) -> float:
    return float(np.log10(max(1.0, float(value))))


@dataclass
class MetricsConfig:
    eps_V: float = 1e-3
    eps_dotV: float = 1e-3
    top_k: int = 20
    alpha: float = 1.0
    beta: float = 0.5
    chi: float = 0.01
    w_plus: float = 1.0
    w_minus: float = 1.0


def conditioning_metrics(registry: BasisRegistry, x: torch.Tensor) -> dict:
    matrix = registry.compute_unscaled(x).detach().cpu().double()
    singular = torch.linalg.svdvals(matrix).clamp_min(1e-30)
    raw_condition = float((singular[0] / singular[-1]).item())
    rms = matrix.square().mean(dim=0).sqrt().clamp_min(1e-30)
    scaled_singular = torch.linalg.svdvals(matrix / rms).clamp_min(1e-30)
    scaled_condition = float((scaled_singular[0] / scaled_singular[-1]).item())
    tolerance = max(matrix.shape) * float(scaled_singular[0]) * np.finfo(np.float64).eps
    rank = int((scaled_singular > tolerance).sum().item())
    return {
        "raw_condition_number": raw_condition,
        "scaled_condition_number": scaled_condition,
        "numerical_rank": rank,
        "min_singular_value": float(scaled_singular[-1]),
        "max_singular_value": float(scaled_singular[0]),
        "n_samples": int(matrix.shape[0]),
        "basis_dim": int(matrix.shape[1]),
    }


def complexity_metrics(registry: BasisRegistry) -> dict:
    metadata = registry.to_dict()
    return {
        "term_count": metadata["output_dim"],
        "operation_count": metadata["operation_count"],
        "term_counts_by_type": metadata["term_counts_by_type"],
        "complexity_total": metadata["complexity_total"],
    }


def positivity_metrics(
    values: np.ndarray,
    state_norm_squared: np.ndarray,
    eps_V: float,
    top_k: int,
) -> dict:
    margin = float(eps_V) * state_norm_squared - values
    violated = margin > 0.0
    indices = np.argsort(-margin)[: min(int(top_k), len(margin))]
    return {
        "eps_V": float(eps_V),
        "violation_count": int(violated.sum()),
        "violation_rate": float(violated.mean()) if len(violated) else 0.0,
        "max_violation": float(margin[violated].max()) if violated.any() else 0.0,
        "mean_violation": float(margin[violated].mean()) if violated.any() else 0.0,
        "min_V": float(values.min()),
        "worst_samples": [
            {
                "index": int(index),
                "V": float(values[index]),
                "x_norm_sq": float(state_norm_squared[index]),
                "margin": float(margin[index]),
            }
            for index in indices
        ],
    }


def decrease_metrics(
    derivative: np.ndarray,
    state_norm_squared: np.ndarray,
    eps_dotV: float,
    top_k: int,
    environment_id: np.ndarray | None = None,
) -> dict:
    margin = derivative + float(eps_dotV) * state_norm_squared
    violated = margin > 0.0
    raw_positive = derivative > 0.0
    indices = np.argsort(-margin)[: min(int(top_k), len(margin))]
    output = {
        "eps_dotV": float(eps_dotV),
        "violation_count": int(violated.sum()),
        "violation_rate": float(violated.mean()) if len(violated) else 0.0,
        "raw_positive_count": int(raw_positive.sum()),
        "raw_positive_rate": float(raw_positive.mean()) if len(raw_positive) else 0.0,
        "max_dotV": float(derivative.max()),
        "max_margin_violation": float(margin[violated].max()) if violated.any() else 0.0,
        "mean_violation": float(margin[violated].mean()) if violated.any() else 0.0,
        "margin_only_count": int((violated & ~raw_positive).sum()),
        "margin_only_rate": float((violated & ~raw_positive).mean()) if len(violated) else 0.0,
        "mean_dotV": float(derivative.mean()),
        "dotV_quantiles": {
            str(q): float(np.quantile(derivative, q))
            for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
        "worst_samples": [
            {
                "index": int(index),
                "dotV": float(derivative[index]),
                "x_norm_sq": float(state_norm_squared[index]),
                "margin": float(margin[index]),
            }
            for index in indices
        ],
    }
    if environment_id is not None:
        per_environment = {}
        for value in np.unique(environment_id):
            mask = environment_id == value
            per_environment[int(value)] = {
                "n": int(mask.sum()),
                "violation_count": int(violated[mask].sum()),
                "violation_rate": float(violated[mask].mean()),
                "raw_positive_rate": float(raw_positive[mask].mean()),
            }
        output["per_parameter"] = per_environment
    return output


def composite_score(
    scaffold_relative_l2: float,
    positivity_rate: float,
    decrease_rate: float,
    scaled_condition_number: float,
    operation_count: int,
    config: MetricsConfig,
) -> dict:
    fit_error = float(scaffold_relative_l2)
    lyapunov_error = config.w_plus * float(positivity_rate) + config.w_minus * float(decrease_rate)
    conditioning = _safe_log10(scaled_condition_number)
    simplicity = float(operation_count)
    score = fit_error + config.alpha * lyapunov_error + config.beta * conditioning + config.chi * simplicity
    return {
        "E_fit": fit_error,
        "E_lyap": lyapunov_error,
        "C_cond": conditioning,
        "C_simp": simplicity,
        "S": score,
    }
