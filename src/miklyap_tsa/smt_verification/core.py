"""VSG model and dReal SMT formula construction."""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np


PAPER_BASIS_NAMES = (
    "x1",
    "x2",
    "x3",
    "x1_sq",
    "x2_sq",
    "x3_sq",
    "x1_x2",
    "x1_x3",
    "x2_x3",
    "sin_x1",
    "one_minus_cos_x1",
    "x3_sin_x1",
    "x3_one_minus_cos_x1",
)


def _smt_num(value: float) -> str:
    return format(float(value), ".17g")


def _smt_sum(terms: Sequence[str]) -> str:
    values = [term for term in terms if term not in {"0", "0.0"}]
    if not values:
        return "0.0"
    if len(values) == 1:
        return values[0]
    return "(+ " + " ".join(values) + ")"


def _smt_mul(*terms: str) -> str:
    values = [term for term in terms if term not in {"1", "1.0"}]
    if not values:
        return "1.0"
    if len(values) == 1:
        return values[0]
    return "(* " + " ".join(values) + ")"


@dataclass(frozen=True)
class VSGParameters:
    """Default physical parameters used by the VSG model."""

    fn_hz: float = 50.0
    us_v: float = 311.0
    pref_w: float = 20_000.0
    qref_var: float = 0.0
    inertia_j: float = 5.0
    damping_d: float = 8.0
    kv: float = 12.0
    dq: float = 166.0
    r1_ohm: float = 0.23
    l1_h: float = 8e-3

    def to_pu(self) -> dict[str, float]:
        base_frequency = 2.0 * math.pi * self.fn_hz
        apparent_power_base = self.pref_w
        impedance_base = self.us_v**2 / apparent_power_base
        return {
            "Pref_pu": self.pref_w / apparent_power_base,
            "Qref_pu": self.qref_var / apparent_power_base,
            "R1_pu": self.r1_ohm / impedance_base,
            "X1_pu": base_frequency * self.l1_h / impedance_base,
            "a_P": apparent_power_base / (self.inertia_j * base_frequency**2),
            "a_D": self.damping_d / self.inertia_j,
            "a_Q": apparent_power_base / (self.kv * self.us_v),
            "a_V": self.dq / self.kv,
            "Us_pu": 1.0,
            "wb": base_frequency,
        }


class VSGModel:
    """Pure-real third-order VSG model in normalized coordinates."""

    def __init__(self, parameters: VSGParameters | None = None) -> None:
        self.parameters = parameters or VSGParameters()
        self.pu = self.parameters.to_pu()
        self.scales = np.array(
            [3.5, 15.0 / self.pu["wb"], 25.0 / self.parameters.us_v], dtype=float
        )
        self.x_eq_pu = self._solve_equilibrium()

    def electrical_power_pu(self, delta: float, e_pu: float) -> tuple[float, float]:
        real_voltage = e_pu * math.cos(delta)
        imag_voltage = e_pu * math.sin(delta)
        real_difference = real_voltage - self.pu["Us_pu"]
        imag_difference = imag_voltage
        denominator = self.pu["R1_pu"] ** 2 + self.pu["X1_pu"] ** 2
        current_real = (
            real_difference * self.pu["R1_pu"] + imag_difference * self.pu["X1_pu"]
        ) / denominator
        current_imag = (
            imag_difference * self.pu["R1_pu"] - real_difference * self.pu["X1_pu"]
        ) / denominator
        return (
            real_voltage * current_real + imag_voltage * current_imag,
            imag_voltage * current_real - real_voltage * current_imag,
        )

    def rhs_physical(self, state: Sequence[float]) -> np.ndarray:
        delta, omega_pu, e_pu = (float(value) for value in state)
        active, reactive = self.electrical_power_pu(delta, e_pu)
        return np.array(
            [
                self.pu["wb"] * (omega_pu - 1.0),
                self.pu["a_P"] * (self.pu["Pref_pu"] - active)
                - self.pu["a_D"] * (omega_pu - 1.0),
                self.pu["a_Q"] * (self.pu["Qref_pu"] - reactive)
                - self.pu["a_V"] * (e_pu - self.pu["Us_pu"]),
            ],
            dtype=float,
        )

    def rhs_normalized(self, x: Sequence[float]) -> np.ndarray:
        state = self.x_eq_pu + self.scales * np.asarray(x, dtype=float)
        return self.rhs_physical(state) / self.scales

    def _equilibrium_residual(self, values: np.ndarray) -> np.ndarray:
        return self.rhs_physical([values[0], 1.0, values[1]])[1:]

    def _solve_equilibrium(self) -> np.ndarray:
        values = np.array([0.2, 1.0], dtype=float)
        for _ in range(80):
            residual = self._equilibrium_residual(values)
            if np.linalg.norm(residual, ord=np.inf) < 1e-12:
                return np.array([values[0], 1.0, values[1]], dtype=float)
            jacobian = np.empty((2, 2), dtype=float)
            for index in range(2):
                step = 1e-6 * max(1.0, abs(values[index]))
                plus = values.copy()
                minus = values.copy()
                plus[index] += step
                minus[index] -= step
                jacobian[:, index] = (
                    self._equilibrium_residual(plus)
                    - self._equilibrium_residual(minus)
                ) / (2.0 * step)
            try:
                direction = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError as error:
                raise RuntimeError("Could not solve the default VSG equilibrium.") from error
            accepted = False
            base_norm = np.linalg.norm(residual)
            for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = values + damping * direction
                if np.linalg.norm(self._equilibrium_residual(candidate)) < base_norm:
                    values = candidate
                    accepted = True
                    break
            if not accepted:
                values += 0.1 * direction
        raise RuntimeError("The default VSG equilibrium did not converge.")

    def normalized_jacobian(self) -> np.ndarray:
        jacobian = np.empty((3, 3), dtype=float)
        for index in range(3):
            step = 1e-6
            plus = np.zeros(3, dtype=float)
            minus = np.zeros(3, dtype=float)
            plus[index] = step
            minus[index] = -step
            jacobian[:, index] = (self.rhs_normalized(plus) - self.rhs_normalized(minus)) / (2.0 * step)
        return jacobian


def basis_numeric(x: Sequence[float]) -> np.ndarray:
    x1, x2, x3 = (float(value) for value in x)
    return np.array(
        [
            x1,
            x2,
            x3,
            x1 * x1,
            x2 * x2,
            x3 * x3,
            x1 * x2,
            x1 * x3,
            x2 * x3,
            math.sin(x1),
            1.0 - math.cos(x1),
            x3 * math.sin(x1),
            x3 * (1.0 - math.cos(x1)),
        ],
        dtype=float,
    )


def _basis_index(names: Sequence[str]) -> list[int]:
    unknown = [name for name in names if name not in PAPER_BASIS_NAMES]
    if unknown:
        raise ValueError(f"Unknown paper basis terms: {unknown}")
    return [PAPER_BASIS_NAMES.index(name) for name in names]


def basis_jacobian_numeric(x: Sequence[float]) -> np.ndarray:
    x1, x2, x3 = (float(value) for value in x)
    return np.array(
        [
            [1.0, 0.0, 0.0, 2.0 * x1, 0.0, 0.0, x2, x3, 0.0, math.cos(x1), math.sin(x1), x3 * math.cos(x1), x3 * math.sin(x1)],
            [0.0, 1.0, 0.0, 0.0, 2.0 * x2, 0.0, x1, 0.0, x3, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 2.0 * x3, 0.0, x1, x2, 0.0, 0.0, math.sin(x1), 1.0 - math.cos(x1)],
        ],
        dtype=float,
    )


@dataclass
class AnalyticCandidate:
    """Quadratic form over an ordered subset of the paper basis."""

    matrix: np.ndarray
    basis_names: tuple[str, ...] = PAPER_BASIS_NAMES

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        self.basis_names = tuple(self.basis_names)
        _basis_index(self.basis_names)
        if len(set(self.basis_names)) != len(self.basis_names):
            raise ValueError("Candidate basis names must be unique.")
        dimension = len(self.basis_names)
        if self.matrix.shape != (dimension, dimension):
            raise ValueError(
                f"The candidate matrix must have shape ({dimension}, {dimension})."
            )
        self.matrix = 0.5 * (self.matrix + self.matrix.T)

    def value(self, x: Sequence[float]) -> float:
        basis = basis_numeric(x)[_basis_index(self.basis_names)]
        return float(basis @ self.matrix @ basis)

    def gradient(self, x: Sequence[float]) -> np.ndarray:
        basis = basis_numeric(x)[_basis_index(self.basis_names)]
        derivative_basis = basis_jacobian_numeric(x)[:, _basis_index(self.basis_names)]
        return 2.0 * derivative_basis @ self.matrix @ basis

    def dot_value(self, x: Sequence[float], model: VSGModel) -> float:
        return float(self.gradient(x) @ model.rhs_normalized(x))

    def _nonzero_upper_terms(self) -> list[tuple[int, int, float]]:
        return [
            (row, column, float(self.matrix[row, column]))
            for row in range(self.matrix.shape[0])
            for column in range(row, self.matrix.shape[1])
            if abs(float(self.matrix[row, column])) > 1e-13
        ]

    def smt_value(self, basis: Sequence[str]) -> str:
        terms = []
        for row, column, coefficient in self._nonzero_upper_terms():
            factor = coefficient if row == column else 2.0 * coefficient
            terms.append(_smt_mul(_smt_num(factor), basis[row], basis[column]))
        return _smt_sum(terms)

    def smt_gradient(self, basis: Sequence[str], derivative_basis: Sequence[Sequence[str]]) -> list[str]:
        gradients = []
        for derivative in derivative_basis:
            terms = []
            for row, column, coefficient in self._nonzero_upper_terms():
                if row == column:
                    terms.append(_smt_mul(_smt_num(2.0 * coefficient), basis[row], derivative[row]))
                else:
                    product_derivative = _smt_sum(
                        [
                            _smt_mul(derivative[row], basis[column]),
                            _smt_mul(basis[row], derivative[column]),
                        ]
                    )
                    terms.append(_smt_mul(_smt_num(2.0 * coefficient), product_derivative))
            gradients.append(_smt_sum(terms))
        return gradients


def smt_basis(
    variables: Sequence[str], names: Sequence[str] = PAPER_BASIS_NAMES
) -> list[str]:
    x1, x2, x3 = variables
    full = {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x1_sq": _smt_mul(x1, x1),
        "x2_sq": _smt_mul(x2, x2),
        "x3_sq": _smt_mul(x3, x3),
        "x1_x2": _smt_mul(x1, x2),
        "x1_x3": _smt_mul(x1, x3),
        "x2_x3": _smt_mul(x2, x3),
        "sin_x1": f"(sin {x1})",
        "one_minus_cos_x1": f"(- 1.0 (cos {x1}))",
        "x3_sin_x1": _smt_mul(x3, f"(sin {x1})"),
        "x3_one_minus_cos_x1": _smt_mul(x3, f"(- 1.0 (cos {x1}))"),
    }
    _basis_index(names)
    return [full[name] for name in names]


def smt_basis_derivatives(
    variables: Sequence[str], names: Sequence[str] = PAPER_BASIS_NAMES
) -> list[list[str]]:
    x1, x2, x3 = variables
    full = [
        [
            "1.0",
            "0.0",
            "0.0",
            _smt_mul("2.0", x1),
            "0.0",
            "0.0",
            x2,
            x3,
            "0.0",
            f"(cos {x1})",
            f"(sin {x1})",
            _smt_mul(x3, f"(cos {x1})"),
            _smt_mul(x3, f"(sin {x1})"),
        ],
        [
            "0.0",
            "1.0",
            "0.0",
            "0.0",
            _smt_mul("2.0", x2),
            "0.0",
            x1,
            "0.0",
            x3,
            "0.0",
            "0.0",
            "0.0",
            "0.0",
        ],
        [
            "0.0",
            "0.0",
            "1.0",
            "0.0",
            "0.0",
            _smt_mul("2.0", x3),
            "0.0",
            x1,
            x2,
            "0.0",
            "0.0",
            f"(sin {x1})",
            f"(- 1.0 (cos {x1}))",
        ],
    ]
    indices = _basis_index(names)
    return [[row[index] for index in indices] for row in full]


def smt_dynamics(model: VSGModel, variables: Sequence[str]) -> list[str]:
    x1, x2, x3 = variables
    equilibrium = model.x_eq_pu
    scales = model.scales
    delta = _smt_sum([_smt_num(equilibrium[0]), _smt_mul(_smt_num(scales[0]), x1)])
    omega = _smt_sum([_smt_num(equilibrium[1]), _smt_mul(_smt_num(scales[1]), x2)])
    voltage = _smt_sum([_smt_num(equilibrium[2]), _smt_mul(_smt_num(scales[2]), x3)])
    real_voltage = _smt_mul(voltage, f"(cos {delta})")
    imag_voltage = _smt_mul(voltage, f"(sin {delta})")
    real_difference = f"(- {real_voltage} {_smt_num(model.pu['Us_pu'])})"
    denominator = _smt_num(model.pu["R1_pu"] ** 2 + model.pu["X1_pu"] ** 2)
    current_real = f"(/ {_smt_sum([_smt_mul(real_difference, _smt_num(model.pu['R1_pu'])), _smt_mul(imag_voltage, _smt_num(model.pu['X1_pu']))])} {denominator})"
    current_imag = f"(/ (- {_smt_mul(imag_voltage, _smt_num(model.pu['R1_pu']))} {_smt_mul(real_difference, _smt_num(model.pu['X1_pu']))}) {denominator})"
    active = _smt_sum([_smt_mul(real_voltage, current_real), _smt_mul(imag_voltage, current_imag)])
    reactive = f"(- {_smt_mul(imag_voltage, current_real)} {_smt_mul(real_voltage, current_imag)})"
    omega_minus_one = f"(- {omega} 1.0)"
    active_error = f"(- {_smt_num(model.pu['Pref_pu'])} {active})"
    reactive_error = f"(- {_smt_num(model.pu['Qref_pu'])} {reactive})"
    voltage_error = f"(- {voltage} {_smt_num(model.pu['Us_pu'])})"
    first = _smt_mul(_smt_num(model.pu["wb"]), omega_minus_one)
    second = f"(- {_smt_mul(_smt_num(model.pu['a_P']), active_error)} {_smt_mul(_smt_num(model.pu['a_D']), omega_minus_one)})"
    third = f"(- {_smt_mul(_smt_num(model.pu['a_Q']), reactive_error)} {_smt_mul(_smt_num(model.pu['a_V']), voltage_error)})"
    return [
        f"(/ {first} {_smt_num(scales[0])})",
        f"(/ {second} {_smt_num(scales[1])})",
        f"(/ {third} {_smt_num(scales[2])})",
    ]


def ellipsoid_smt_expression(variables: Sequence[str], radii: Sequence[float]) -> str:
    terms = [
        _smt_mul(_smt_num(1.0 / float(radius) ** 2), variable, variable)
        for variable, radius in zip(variables, radii)
    ]
    return f"(<= {_smt_sum(terms)} 1.0)"


def build_smt_formula(
    model: VSGModel,
    candidate: AnalyticCandidate,
    domain_expression: str,
    margin_name: str = "decrease",
    eps_v: float = 1e-3,
    eps_dot_v: float = 1e-3,
    origin_tol: float = 1e-3,
) -> str:
    """Build a QF_NRA violation-existence query for dReal."""

    variables = ["x1", "x2", "x3"]
    basis = smt_basis(variables, candidate.basis_names)
    derivative_basis = smt_basis_derivatives(variables, candidate.basis_names)
    value = candidate.smt_value(basis)
    gradient = candidate.smt_gradient(basis, derivative_basis)
    dynamics = smt_dynamics(model, variables)
    dot_value = _smt_sum([_smt_mul(gradient[index], dynamics[index]) for index in range(3)])
    norm_squared = _smt_sum([_smt_mul(variable, variable) for variable in variables])
    if margin_name == "positivity":
        margin = f"(- {_smt_mul(_smt_num(eps_v), norm_squared)} {value})"
    elif margin_name == "decrease":
        margin = _smt_sum([dot_value, _smt_mul(_smt_num(eps_dot_v), norm_squared)])
    else:
        raise ValueError("margin_name must be 'positivity' or 'decrease'.")
    assertions = [
        domain_expression,
        f"(>= {norm_squared} {_smt_num(origin_tol**2)})",
        f"(> {margin} 0.0)",
    ]
    return "\n".join(
        [
            "(set-logic QF_NRA)",
            "(declare-fun x1 () Real)",
            "(declare-fun x2 () Real)",
            "(declare-fun x3 () Real)",
            "(assert (and",
            *[f"  {assertion}" for assertion in assertions],
            "))",
            "(check-sat)",
            "(exit)",
            "",
        ]
    )


def classify_dreal_output(stdout: str, stderr: str, returncode: int) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "delta-sat" in text:
        return "delta-sat"
    if re.search(r"\bunsat\b", text):
        return "unsat"
    if re.search(r"\bsat\b", text):
        return "sat"
    return "error" if int(returncode) != 0 else "unknown"


def run_dreal_query(
    solver: str,
    formula: str,
    delta: float = 1e-4,
    timeout: float = 30.0,
    random_seed: int = 2026,
) -> dict:
    """Run one external dReal query and return a compact status record."""

    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".smt2", encoding="utf-8") as handle:
            handle.write(formula)
            handle.flush()
            completed = subprocess.run(
                [
                    solver,
                    "--precision",
                    _smt_num(delta),
                    "--jobs",
                    "1",
                    "--random-seed",
                    str(int(random_seed)),
                    handle.name,
                ],
                capture_output=True,
                text=True,
                timeout=float(timeout),
                check=False,
            )
        output = f"{completed.stdout}\n{completed.stderr}".strip()
        return {
            "status": classify_dreal_output(completed.stdout, completed.stderr, completed.returncode),
            "returncode": int(completed.returncode),
            "solver_ms": 1000.0 * (time.perf_counter() - started),
            "output_tail": output[-300:],
        }
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "returncode": None,
            "solver_ms": 1000.0 * (time.perf_counter() - started),
            "output_tail": str(error)[-300:],
        }
    except FileNotFoundError as error:
        return {
            "status": "solver-not-found",
            "returncode": None,
            "solver_ms": 0.0,
            "output_tail": str(error),
        }
