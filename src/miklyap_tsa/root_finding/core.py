"""Numerical root-finding and local Lyapunov falsification kernels."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from scipy.optimize import minimize, root
from scipy.stats import qmc


@dataclass
class FalsifierConfig:
    eps_V: float = 1e-3
    eps_dotV: float = 1e-3
    n_seeds: int = 200
    n_origin_seeds: int = 8
    origin_seed_radius: float = 1e-3
    probe_directions: int = 16
    probe_n_steps: int = 14
    probe_step0: float = 1e-4
    probe_step_ratio: float = 2.0
    root_tol: float = 1e-5
    minimize_ftol: float = 1e-16
    minimize_maxiter: int = 80
    flow_maxiter: int = 60
    dedupe_rel_tol: float = 1e-3
    origin_tol: float = 1e-3
    seed: int = 2026


def fibonacci_sphere_directions(n: int, dim: int = 3) -> np.ndarray:
    """Return approximately uniform unit directions."""

    count = max(int(n), 1)
    if int(dim) == 3:
        index = np.arange(count, dtype=float) + 0.5
        golden = np.pi * (1.0 + np.sqrt(5.0))
        theta = golden * index
        cosine = 1.0 - 2.0 * index / count
        sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
        return np.stack(
            [sine * np.cos(theta), sine * np.sin(theta), cosine], axis=-1
        )
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((count, int(dim)))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def generate_seeds(
    lo: Sequence[float],
    hi: Sequence[float],
    n_lhs: int,
    seed: int,
    n_grid_per_dim: int = 0,
    n_origin: int = 0,
    origin_radius: float = 1e-3,
) -> np.ndarray:
    """Generate Latin-hypercube, grid, and near-origin seed points."""

    low = np.asarray(lo, dtype=float)
    high = np.asarray(hi, dtype=float)
    if low.shape != high.shape or np.any(low >= high):
        raise ValueError("lo and hi must have equal shapes with lo < hi.")
    parts: list[np.ndarray] = []
    if int(n_lhs) > 0:
        sampler = qmc.LatinHypercube(d=low.size, seed=int(seed))
        parts.append(qmc.scale(sampler.random(int(n_lhs)), low, high))
    if int(n_grid_per_dim) > 1:
        axes = [np.linspace(low[i], high[i], int(n_grid_per_dim)) for i in range(low.size)]
        mesh = np.meshgrid(*axes, indexing="ij")
        parts.append(np.stack([axis.ravel() for axis in mesh], axis=-1))
    if int(n_origin) > 0:
        rng = np.random.default_rng(int(seed) + 1)
        directions = rng.standard_normal((int(n_origin), low.size))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = float(origin_radius) * np.geomspace(0.1, 1.0, int(n_origin))
        parts.append(np.clip(directions * radii[:, None], low, high))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, low.size))


def cluster_points(points: np.ndarray, tol: float) -> list[list[int]]:
    """Greedily cluster points by Euclidean distance."""

    clusters: list[list[int]] = []
    representatives: list[np.ndarray] = []
    for index, point in enumerate(np.asarray(points, dtype=float)):
        for cluster, representative in zip(clusters, representatives):
            if np.linalg.norm(point - representative) <= float(tol):
                cluster.append(index)
                break
        else:
            clusters.append([index])
            representatives.append(point)
    return clusters


def _level_set_flow(
    value_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    lo: np.ndarray | None,
    hi: np.ndarray | None,
    tol: float,
    maxiter: int,
) -> np.ndarray:
    point = np.asarray(x0, dtype=float).copy()
    for _ in range(int(maxiter)):
        value, gradient = value_and_grad(point)
        value = float(value)
        gradient = np.asarray(gradient, dtype=float)
        if abs(value) <= tol:
            break
        gradient_norm_squared = float(gradient @ gradient)
        if not np.isfinite(value) or not np.isfinite(gradient_norm_squared) or gradient_norm_squared < 1e-30:
            break
        step = (value / gradient_norm_squared) * gradient
        accepted = False
        for _ in range(20):
            candidate = point - step
            if lo is not None and hi is not None:
                candidate = np.clip(candidate, lo, hi)
            candidate_value = float(value_and_grad(candidate)[0])
            if np.isfinite(candidate_value) and abs(candidate_value) < abs(value):
                point = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return point


def find_zero_roots(
    value_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    seeds: np.ndarray,
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
    tol: float = 1e-8,
    dedupe_tol: float = 1e-6,
    ftol: float = 1e-16,
    maxiter: int = 200,
    flow_maxiter: int = 60,
    time_budget_s: float | None = None,
    t_start: float | None = None,
) -> tuple[list[dict], int, bool]:
    """Locate a scalar zero set from multiple seeds using analytic gradients."""

    low = None if lo is None else np.asarray(lo, dtype=float)
    high = None if hi is None else np.asarray(hi, dtype=float)
    bounds = None if low is None or high is None else list(zip(low, high))
    roots: list[np.ndarray] = []
    output: list[dict] = []
    attempted = 0
    partial = False
    for seed_index, seed in enumerate(np.asarray(seeds, dtype=float)):
        if time_budget_s is not None and t_start is not None:
            if time.perf_counter() - t_start > float(time_budget_s):
                partial = True
                break
        attempted += 1
        try:
            flowed = _level_set_flow(
                value_and_grad, seed, low, high, float(tol), int(flow_maxiter)
            )

            def objective(point: np.ndarray) -> tuple[float, np.ndarray]:
                value, gradient = value_and_grad(point)
                value = float(value)
                gradient = np.asarray(gradient, dtype=float)
                return 0.5 * value * value, value * gradient

            result = minimize(
                lambda point: objective(point)[0],
                flowed,
                jac=lambda point: objective(point)[1],
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": float(ftol), "gtol": 1e-14, "maxiter": int(maxiter)},
            )
            point = np.asarray(result.x, dtype=float)
            residual = abs(float(value_and_grad(point)[0]))
        except Exception:
            continue
        if not np.isfinite(residual) or residual > float(tol):
            continue
        if roots and min(np.linalg.norm(point - previous) for previous in roots) <= float(dedupe_tol):
            continue
        roots.append(point)
        output.append({"x": point, "residual": float(residual), "seed_index": seed_index})
    return output, attempted, partial


def find_stationary_points(
    value_and_jacobian: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    seeds: np.ndarray,
    tol: float = 1e-8,
    dedupe_tol: float = 1e-6,
    time_budget_s: float | None = None,
    t_start: float | None = None,
) -> tuple[list[dict], bool]:
    """Find roots of a square vector field with an analytic Jacobian."""

    accepted: list[np.ndarray] = []
    output: list[dict] = []
    partial = False
    for seed_index, seed in enumerate(np.asarray(seeds, dtype=float)):
        if time_budget_s is not None and t_start is not None:
            if time.perf_counter() - t_start > float(time_budget_s):
                partial = True
                break
        try:
            result = root(
                lambda point: np.asarray(value_and_jacobian(point)[0], dtype=float),
                seed,
                jac=lambda point: np.asarray(value_and_jacobian(point)[1], dtype=float),
                method="hybr",
                tol=float(tol),
            )
        except Exception:
            continue
        point = np.asarray(result.x, dtype=float)
        residual = float(np.max(np.abs(np.asarray(result.fun, dtype=float))))
        if not result.success or not np.isfinite(residual) or residual > float(tol):
            continue
        if accepted and min(np.linalg.norm(point - previous) for previous in accepted) <= float(dedupe_tol):
            continue
        accepted.append(point)
        output.append({"x": point, "residual": residual, "seed_index": seed_index})
    return output, partial


def probe_root_neighborhoods(
    margin_fn: Callable[[np.ndarray], np.ndarray],
    roots: Sequence[Sequence[float]],
    directions: np.ndarray,
    steps: Sequence[float],
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
    time_budget_s: float | None = None,
    t_start: float | None = None,
) -> tuple[list[dict], bool]:
    """Probe a batched positive margin around each root."""

    directions = np.asarray(directions, dtype=float)
    steps = np.asarray(steps, dtype=float)
    low = None if lo is None else np.asarray(lo, dtype=float)
    high = None if hi is None else np.asarray(hi, dtype=float)
    hits: list[dict] = []
    partial = False
    for root_index, root_point in enumerate(roots):
        if time_budget_s is not None and t_start is not None:
            if time.perf_counter() - t_start > float(time_budget_s):
                partial = True
                break
        root_point = np.asarray(root_point, dtype=float)
        points = root_point[None, None, :] + directions[:, None, :] * steps[None, :, None]
        flat = points.reshape(-1, points.shape[-1])
        inside = np.ones(flat.shape[0], dtype=bool)
        if low is not None and high is not None:
            inside = np.all((flat >= low) & (flat <= high), axis=-1)
        if not inside.any():
            continue
        try:
            margins = np.asarray(margin_fn(flat[inside]), dtype=float)
        except Exception:
            continue
        for flat_index, margin in zip(np.flatnonzero(inside), margins):
            if not np.isfinite(margin) or margin <= 0.0:
                continue
            direction_index, step_index = divmod(int(flat_index), len(steps))
            hits.append(
                {
                    "x": flat[flat_index],
                    "margin": float(margin),
                    "root_index": root_index,
                    "direction_index": direction_index,
                    "step_index": step_index,
                }
            )
    return hits, partial


def dedupe_counterexamples(hits: list[dict], tol: float) -> list[dict]:
    """Keep the largest margin in each counterexample cluster."""

    if not hits:
        return []
    points = np.stack([np.asarray(hit["x"], dtype=float) for hit in hits])
    used = np.zeros(len(hits), dtype=bool)
    kept: list[dict] = []
    for index in np.argsort([-float(hit["margin"]) for hit in hits]):
        if used[index]:
            continue
        cluster = np.flatnonzero(
            (~used) & (np.linalg.norm(points - points[index], axis=1) <= float(tol))
        )
        used[cluster] = True
        kept.append(hits[int(index)])
    return kept


class DotVField:
    """Autograd-backed ``dot V`` field for one parameter instance.

    ``value_fn`` receives ``(x, p)`` and returns one value per batch row.
    ``rhs_fn`` receives normalized states and returns the physical vector field
    in the same normalized coordinates.
    """

    def __init__(
        self,
        value_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        rhs_fn: Callable[[torch.Tensor], torch.Tensor],
        p: torch.Tensor,
        device: torch.device | str = "cpu",
    ) -> None:
        self.value_fn = value_fn
        self.rhs_fn = rhs_fn
        self.device = torch.device(device)
        self.p = p.detach().to(self.device)
        if self.p.dim() == 1:
            self.p = self.p.unsqueeze(0)

    def _state_leaf(self, x: Sequence[float] | np.ndarray) -> torch.Tensor:
        array = np.asarray(x, dtype=np.float32).reshape(-1, 3)
        return torch.from_numpy(array).to(self.device).detach().clone().requires_grad_(True)

    def _parameter_batch(self, count: int) -> torch.Tensor:
        if self.p.shape[0] == 1:
            return self.p.expand(int(count), -1)
        if self.p.shape[0] != int(count):
            raise ValueError("The parameter batch does not match the state batch.")
        return self.p

    def _value_gradient_derivative(self, state: torch.Tensor, create_graph: bool):
        parameters = self._parameter_batch(state.shape[0])
        value = self.value_fn(state, parameters)
        gradient = torch.autograd.grad(
            value,
            state,
            grad_outputs=torch.ones_like(value),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        derivative = (gradient * self.rhs_fn(state)).sum(dim=-1)
        return value, gradient, derivative

    def dotV_value(self, x: Sequence[float] | np.ndarray) -> float:
        with torch.enable_grad():
            state = self._state_leaf(x)
            return float(self._value_gradient_derivative(state, False)[2][0].item())

    def dotV_value_and_grad(self, x: Sequence[float] | np.ndarray) -> tuple[float, np.ndarray]:
        with torch.enable_grad():
            state = self._state_leaf(x)
            derivative = self._value_gradient_derivative(state, True)[2]
            gradient = torch.autograd.grad(derivative[0], state)[0][0]
            return float(derivative[0].item()), gradient.detach().double().cpu().numpy()

    def gradV_and_hessian(self, x: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.enable_grad():
            state = self._state_leaf(x)
            value, gradient, _ = self._value_gradient_derivative(state, True)
            rows = [torch.autograd.grad(gradient[0, index], state, retain_graph=True)[0][0] for index in range(3)]
            return (
                gradient[0].detach().double().cpu().numpy(),
                torch.stack(rows).detach().double().cpu().numpy(),
            )

    def dotV_batch(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32).reshape(-1, 3)
        with torch.enable_grad():
            state = torch.from_numpy(array).to(self.device).requires_grad_(True)
            derivative = self._value_gradient_derivative(state, False)[2]
        return derivative.detach().double().cpu().numpy()

    def dotV_and_V_batch(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(x, dtype=np.float32).reshape(-1, 3)
        with torch.enable_grad():
            state = torch.from_numpy(array).to(self.device).requires_grad_(True)
            value, _, derivative = self._value_gradient_derivative(state, False)
        return derivative.detach().double().cpu().numpy(), value.detach().double().cpu().numpy()

    def decrease_margin(self, x: Sequence[float] | np.ndarray, eps_dotV: float) -> float:
        point = np.asarray(x, dtype=float)
        return self.dotV_value(point) + float(eps_dotV) * float(point @ point)

    def decrease_margin_batch(self, x: np.ndarray, eps_dotV: float) -> np.ndarray:
        array = np.asarray(x, dtype=float).reshape(-1, 3)
        return self.dotV_batch(array) + float(eps_dotV) * np.sum(array**2, axis=1)

    def positivity_margin(self, x: Sequence[float] | np.ndarray, eps_V: float) -> float:
        point = np.asarray(x, dtype=float)
        _, values = self.dotV_and_V_batch(point.reshape(1, -1))
        return float(eps_V) * float(point @ point) - float(values[0])

    def positivity_margin_batch(self, x: np.ndarray, eps_V: float) -> np.ndarray:
        array = np.asarray(x, dtype=float).reshape(-1, 3)
        _, values = self.dotV_and_V_batch(array)
        return float(eps_V) * np.sum(array**2, axis=1) - values
