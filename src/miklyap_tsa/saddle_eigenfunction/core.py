"""Local unstable Koopman eigenfunction components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp


def numerical_jacobian(
    rhs: Callable[[np.ndarray], np.ndarray],
    state: Sequence[float],
    step: float = 1e-5,
) -> np.ndarray:
    point = np.asarray(state, dtype=float)
    identity = np.eye(point.size)
    return np.column_stack(
        [(rhs(point + step * direction) - rhs(point - step * direction)) / (2.0 * step) for direction in identity]
    )


@dataclass(frozen=True)
class LocalSaddleGeometry:
    state: np.ndarray
    jacobian: np.ndarray
    unstable_eigenvalue: float
    left_unstable: np.ndarray
    stable_basis: np.ndarray


def _real_vector(vector: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector)
    if np.max(np.abs(vector.imag)) > tolerance:
        raise RuntimeError("The local eigendirection is not real.")
    return vector.real.astype(float)


def local_saddle_geometry(
    rhs: Callable[[np.ndarray], np.ndarray],
    state: Sequence[float],
) -> LocalSaddleGeometry:
    """Compute the type-1 hyperbolic saddle geometry at a supplied state."""

    center = np.asarray(state, dtype=float)
    jacobian = numerical_jacobian(rhs, center)
    eigenvalues, right_vectors = np.linalg.eig(jacobian)
    unstable = np.flatnonzero(eigenvalues.real > 1e-7)
    stable = np.flatnonzero(eigenvalues.real < -1e-7)
    if unstable.size != 1 or stable.size != center.size - 1:
        raise RuntimeError("Expected a type-1 hyperbolic saddle.")
    unstable_index = int(unstable[0])
    left_values, left_vectors = np.linalg.eig(jacobian.T)
    left_index = int(np.argmin(np.abs(left_values - eigenvalues[unstable_index])))
    left = _real_vector(left_vectors[:, left_index])
    left /= np.linalg.norm(left)
    if left[np.argmax(np.abs(left))] < 0.0:
        left = -left
    stable_vectors = np.column_stack(
        [_real_vector(right_vectors[:, int(index)]) for index in stable]
    )
    stable_basis, _ = np.linalg.qr(stable_vectors)
    return LocalSaddleGeometry(
        state=center,
        jacobian=jacobian,
        unstable_eigenvalue=float(eigenvalues[unstable_index].real),
        left_unstable=left,
        stable_basis=stable_basis[:, : center.size - 1],
    )


@dataclass(frozen=True)
class BackwardDataset:
    states: np.ndarray
    next_states: np.ndarray
    labels: np.ndarray
    trajectory_ids: np.ndarray
    elapsed_to_seed: np.ndarray


def split_unit_ids(n_units: int, fractions: Sequence[float], seed: int) -> np.ndarray:
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("fractions must contain train, selection, and validation shares.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(int(n_units))
    counts = np.floor(fractions * int(n_units)).astype(int)
    counts[0] += int(n_units) - int(counts.sum())
    labels = np.empty(int(n_units), dtype=np.int8)
    start = 0
    for split, count in enumerate(counts):
        labels[order[start : start + count]] = split
        start += count
    return labels


def sample_backward_koopman_data(
    rhs: Callable[[np.ndarray], np.ndarray],
    geometry: LocalSaddleGeometry,
    *,
    n_seeds: int,
    stable_radii: Sequence[float],
    unstable_radius: float,
    horizon: float,
    samples: int,
    state_bounds: Sequence[Sequence[float]],
    seed: int,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> BackwardDataset:
    """Generate local backward-flow pairs and eigenfunction labels."""

    rng = np.random.default_rng(seed)
    stable_radii = np.asarray(stable_radii, dtype=float)
    bounds = np.asarray(state_bounds, dtype=float)
    if stable_radii.shape != (2,):
        raise ValueError("stable_radii must have shape (2,).")
    if bounds.shape != (3, 2):
        raise ValueError("state_bounds must have shape (3, 2).")
    times = np.linspace(0.0, float(horizon), int(samples))
    states, next_states, labels, trajectory_ids, elapsed = [], [], [], [], []
    for trajectory_id in range(int(n_seeds)):
        stable_coordinates = rng.uniform(-stable_radii, stable_radii)
        unstable_coordinate = rng.uniform(-float(unstable_radius), float(unstable_radius))
        seed_state = (
            geometry.state
            + geometry.stable_basis @ stable_coordinates
            + unstable_coordinate * geometry.left_unstable
        )
        seed_value = float(geometry.left_unstable @ (seed_state - geometry.state))
        solution = solve_ivp(
            lambda _time, point: -rhs(point),
            (0.0, float(horizon)),
            seed_state,
            t_eval=times,
            rtol=rtol,
            atol=atol,
        )
        if not solution.success or solution.y.shape[1] < 2:
            continue
        path = solution.y.T
        valid = np.all(path >= bounds[:, 0], axis=1) & np.all(path <= bounds[:, 1], axis=1)
        for index in range(1, len(path)):
            if not valid[index] or not valid[index - 1]:
                continue
            elapsed_time = float(times[index])
            states.append(path[index])
            next_states.append(path[index - 1])
            labels.append(np.exp(-geometry.unstable_eigenvalue * elapsed_time) * seed_value)
            trajectory_ids.append(trajectory_id)
            elapsed.append(elapsed_time)
    if not states:
        raise RuntimeError("Backward sampling produced no in-bounds states.")
    return BackwardDataset(
        states=np.asarray(states),
        next_states=np.asarray(next_states),
        labels=np.asarray(labels),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
        elapsed_to_seed=np.asarray(elapsed),
    )


DEFAULT_TERMS = (
    "y0",
    "y1",
    "y2",
    "y0^2",
    "y1^2",
    "y2^2",
    "y0*y1",
    "y0*y2",
    "y1*y2",
    "y0^3",
    "y1^3",
    "y2^3",
    "sin(theta)",
    "cos(theta)-1",
    "y1*sin(theta)",
    "y2*sin(theta)",
    "y1*(cos(theta)-1)",
    "y2*(cos(theta)-1)",
)


@dataclass(frozen=True)
class LocalPeriodicBasis:
    """Equilibrium-centered periodic basis for a local eigenfunction."""

    center: np.ndarray
    delta_scale: float
    terms: tuple[str, ...] = DEFAULT_TERMS

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must have shape (3,).")
        unknown = set(self.terms) - set(DEFAULT_TERMS)
        if unknown:
            raise ValueError(f"Unknown local basis terms: {sorted(unknown)}")
        object.__setattr__(self, "center", center)

    def _values_and_gradients(self, states: np.ndarray):
        points = np.atleast_2d(np.asarray(states, dtype=float))
        y0, y1, y2 = (points - self.center.reshape(1, 3)).T
        theta = float(self.delta_scale) * y0
        sine = np.sin(theta)
        cosine = np.cos(theta)
        cosine_minus_one = cosine - 1.0
        zeros = np.zeros_like(y0)
        values = {
            "y0": y0,
            "y1": y1,
            "y2": y2,
            "y0^2": y0**2,
            "y1^2": y1**2,
            "y2^2": y2**2,
            "y0*y1": y0 * y1,
            "y0*y2": y0 * y2,
            "y1*y2": y1 * y2,
            "y0^3": y0**3,
            "y1^3": y1**3,
            "y2^3": y2**3,
            "sin(theta)": sine,
            "cos(theta)-1": cosine_minus_one,
            "y1*sin(theta)": y1 * sine,
            "y2*sin(theta)": y2 * sine,
            "y1*(cos(theta)-1)": y1 * cosine_minus_one,
            "y2*(cos(theta)-1)": y2 * cosine_minus_one,
        }
        gradients = {
            "y0": np.column_stack([np.ones_like(y0), zeros, zeros]),
            "y1": np.column_stack([zeros, np.ones_like(y0), zeros]),
            "y2": np.column_stack([zeros, zeros, np.ones_like(y0)]),
            "y0^2": np.column_stack([2.0 * y0, zeros, zeros]),
            "y1^2": np.column_stack([zeros, 2.0 * y1, zeros]),
            "y2^2": np.column_stack([zeros, zeros, 2.0 * y2]),
            "y0*y1": np.column_stack([y1, y0, zeros]),
            "y0*y2": np.column_stack([y2, zeros, y0]),
            "y1*y2": np.column_stack([zeros, y2, y1]),
            "y0^3": np.column_stack([3.0 * y0**2, zeros, zeros]),
            "y1^3": np.column_stack([zeros, 3.0 * y1**2, zeros]),
            "y2^3": np.column_stack([zeros, zeros, 3.0 * y2**2]),
            "sin(theta)": np.column_stack([self.delta_scale * cosine, zeros, zeros]),
            "cos(theta)-1": np.column_stack([-self.delta_scale * sine, zeros, zeros]),
            "y1*sin(theta)": np.column_stack([self.delta_scale * y1 * cosine, sine, zeros]),
            "y2*sin(theta)": np.column_stack([self.delta_scale * y2 * cosine, zeros, sine]),
            "y1*(cos(theta)-1)": np.column_stack([-self.delta_scale * y1 * sine, cosine_minus_one, zeros]),
            "y2*(cos(theta)-1)": np.column_stack([-self.delta_scale * y2 * sine, zeros, cosine_minus_one]),
        }
        return values, gradients

    def values(self, states: np.ndarray) -> np.ndarray:
        values, _ = self._values_and_gradients(states)
        return np.column_stack([values[name] for name in self.terms])

    def gradients(self, states: np.ndarray) -> np.ndarray:
        _, gradients = self._values_and_gradients(states)
        return np.stack([gradients[name] for name in self.terms], axis=1)

    def generator_rows(
        self,
        states: np.ndarray,
        rhs_values: np.ndarray,
        eigenvalue: float,
    ) -> np.ndarray:
        values = self.values(states)
        gradients = self.gradients(states)
        directional = np.einsum("nbd,nd->nb", gradients, rhs_values)
        return directional - float(eigenvalue) * values


@dataclass(frozen=True)
class BoundaryPair:
    ray_id: int
    boundary_state: np.ndarray
    return_state: np.ndarray
    escape_state: np.ndarray
    bracket_width: float


@dataclass(frozen=True)
class LocalEigenfunctionModel:
    basis: LocalPeriodicBasis
    coefficients: np.ndarray
    eigenvalue: float
    ridge: float
    condition_number: float

    def values(self, states: np.ndarray) -> np.ndarray:
        return self.basis.values(states) @ self.coefficients

    def gradients(self, states: np.ndarray) -> np.ndarray:
        return np.einsum("nbd,b->nd", self.basis.gradients(states), self.coefficients)

    def to_dict(self) -> dict:
        return {
            "kind": "local_saddle_unstable_eigenfunction",
            "center": self.basis.center.tolist(),
            "delta_scale": float(self.basis.delta_scale),
            "terms": list(self.basis.terms),
            "coefficients": self.coefficients.tolist(),
            "eigenvalue": float(self.eigenvalue),
            "ridge": float(self.ridge),
            "condition_number": float(self.condition_number),
        }


def _append_group(
    matrices: list[np.ndarray],
    targets: list[np.ndarray],
    matrix: np.ndarray,
    target: np.ndarray,
    weight: float,
) -> None:
    if matrix.size == 0 or float(weight) <= 0.0:
        return
    factor = np.sqrt(float(weight) / max(1, matrix.shape[0]))
    matrices.append(factor * matrix)
    targets.append(factor * np.asarray(target, dtype=float))


def fit_local_eigenfunction(
    basis: LocalPeriodicBasis,
    geometry: LocalSaddleGeometry,
    backward: BackwardDataset,
    backward_split: np.ndarray,
    boundary_pairs: Sequence[BoundaryPair],
    boundary_split: np.ndarray,
    rhs: Callable[[np.ndarray], np.ndarray],
    *,
    ridge: float,
    weights: dict,
    flow_dt: float,
) -> LocalEigenfunctionModel:
    """Fit the local eigenfunction with trajectory and saddle constraints."""

    train_mask = backward_split[backward.trajectory_ids] == 0
    states = backward.states[train_mask]
    next_states = backward.next_states[train_mask]
    labels = backward.labels[train_mask]
    values = basis.values(states)
    scales = np.maximum(np.sqrt(np.mean(values**2, axis=0)), 1e-8)
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    _append_group(matrices, targets, values / scales, labels, weights.get("path_labels", 1.0))
    flow_rows = (basis.values(next_states) - np.exp(geometry.unstable_eigenvalue * flow_dt) * values) / scales
    _append_group(matrices, targets, flow_rows, np.zeros(flow_rows.shape[0]), weights.get("flow", 1.0))
    rhs_values = np.asarray([rhs(state) for state in states])
    generator_rows = basis.generator_rows(states, rhs_values, geometry.unstable_eigenvalue) / scales
    _append_group(matrices, targets, generator_rows, np.zeros(generator_rows.shape[0]), weights.get("generator", 1.0))

    train_pairs = [
        pair
        for pair in boundary_pairs
        if int(boundary_split[pair.ray_id]) == 0
    ]
    if train_pairs:
        boundary_states = np.asarray([pair.boundary_state for pair in train_pairs])
        _append_group(
            matrices,
            targets,
            basis.values(boundary_states) / scales,
            np.zeros(len(train_pairs)),
            weights.get("boundary", 1.0),
        )
        side_states = np.asarray(
            [state for pair in train_pairs for state in (pair.return_state, pair.escape_state)]
        )
        side_targets = np.asarray(
            [target for pair in train_pairs for target in (-pair.bracket_width / 2.0, pair.bracket_width / 2.0)]
        )
        _append_group(
            matrices,
            targets,
            basis.values(side_states) / scales,
            side_targets,
            weights.get("side", 0.0),
        )

    center_gradient = basis.gradients(geometry.state.reshape(1, 3))[0].T
    _append_group(
        matrices,
        targets,
        center_gradient / scales.reshape(1, -1),
        geometry.left_unstable,
        weights.get("gradient_anchor", 10.0),
    )
    design = np.vstack(matrices)
    target = np.concatenate(targets)
    gram = design.T @ design + float(ridge) * np.eye(design.shape[1])
    scaled_coefficients = np.linalg.solve(gram, design.T @ target)
    coefficients = scaled_coefficients / scales
    return LocalEigenfunctionModel(
        basis=basis,
        coefficients=coefficients,
        eigenvalue=geometry.unstable_eigenvalue,
        ridge=float(ridge),
        condition_number=float(np.linalg.cond(design)),
    )
