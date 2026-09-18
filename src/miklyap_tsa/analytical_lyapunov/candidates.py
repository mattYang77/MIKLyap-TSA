"""Staged analytical candidate construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .basis import BasisRegistry
from .losses import fit_error_ridge, ridge_scaffold_U


def _safe_log10(value: float) -> float:
    return float(np.log10(max(1.0, float(value))))


@dataclass
class CandidateSearchConfig:
    mandatory_terms: list[str] = field(default_factory=lambda: ["x1", "x2", "x3"])
    max_terms: int = 8
    beam_width: int = 6
    max_full_candidates: int = 4
    correlation_threshold: float = 0.999
    variance_threshold: float = 1e-10
    condition_threshold: float = 1e8
    ridge_lambda: float = 1e-3
    pareto: bool = True
    w_err: float = 1.0
    w_cond: float = 0.1
    w_simp: float = 0.01


@dataclass
class Candidate:
    terms: list[str]
    complexity: int
    fit_error: float
    fit_error_raw: float
    cond_scaled: float
    pre_score: float
    layer: int
    U_ridge: np.ndarray

    def summary(self) -> dict:
        return {
            "terms": list(self.terms),
            "n_terms": len(self.terms),
            "complexity": self.complexity,
            "fit_error": self.fit_error,
            "fit_error_raw": self.fit_error_raw,
            "cond_scaled": self.cond_scaled,
            "pre_score": self.pre_score,
            "layer": self.layer,
        }


class PrimitiveLibrary:
    """Numerical snapshot of the fixed paper basis."""

    def __init__(self, registry: BasisRegistry, states: torch.Tensor) -> None:
        self.registry = registry
        self.names = registry.names
        self.complexities = [term.complexity for term in registry.terms]
        with torch.no_grad():
            self.B = registry.compute_unscaled(states).double().cpu().numpy()
        self.M, self.n_primitives = self.B.shape

    def column_variance(self) -> np.ndarray:
        return self.B.var(axis=0)

    def correlation(self) -> np.ndarray:
        centered = self.B - self.B.mean(axis=0, keepdims=True)
        standard = centered.std(axis=0)
        standard[standard < 1e-12] = 1.0
        normalized = centered / standard
        return normalized.T @ normalized / max(self.M - 1, 1)


def _condition_number(matrix: np.ndarray) -> float:
    rms = np.sqrt((matrix**2).mean(axis=0))
    rms[rms < 1e-12] = 1.0
    singular = np.linalg.svd(matrix / rms, compute_uv=False)
    singular = np.clip(singular, 1e-30, None)
    return float(singular[0] / singular[-1])


def prune_library(lib: PrimitiveLibrary, config: CandidateSearchConfig) -> list[int]:
    variance = lib.column_variance()
    keep = [i for i, value in enumerate(variance) if value > config.variance_threshold]
    if not keep:
        return []
    correlation = lib.correlation()
    mandatory = [
        lib.names.index(name)
        for name in config.mandatory_terms
        if name in lib.names and lib.names.index(name) in keep
    ]
    optional = sorted(
        [i for i in keep if i not in mandatory],
        key=lambda i: lib.complexities[i],
    )
    survivors = list(mandatory)
    for index in optional:
        if any(abs(correlation[index, kept]) > config.correlation_threshold for kept in survivors):
            continue
        survivors.append(index)
        if len(survivors) >= len(lib.names):
            break
    return survivors


def _pareto_filter(candidates: list[Candidate]) -> list[Candidate]:
    result = []
    for candidate in candidates:
        dominated = any(
            other is not candidate
            and other.fit_error <= candidate.fit_error
            and other.complexity <= candidate.complexity
            and (other.fit_error < candidate.fit_error or other.complexity < candidate.complexity)
            for other in candidates
        )
        if not dominated:
            result.append(candidate)
    return result


def generate_candidates(
    registry: BasisRegistry,
    states: torch.Tensor,
    z_target: torch.Tensor,
    config: CandidateSearchConfig,
) -> dict:
    """Generate staged candidates and return ridge warm starts and summaries."""

    library = PrimitiveLibrary(registry, states)
    target = z_target.detach().double().cpu().numpy()
    kept_indices = prune_library(library, config)
    kept_names = [library.names[index] for index in kept_indices]
    basis = library.B[:, kept_indices]
    name_to_column = {name: column for column, name in enumerate(kept_names)}
    mandatory = [name for name in config.mandatory_terms if name in name_to_column]
    seen: set[frozenset[str]] = set()
    all_candidates: list[Candidate] = []

    def make_candidate(names: list[str], layer: int) -> Candidate | None:
        key = frozenset(names)
        if key in seen or not names:
            return None
        seen.add(key)
        columns = sorted(name_to_column[name] for name in names)
        submatrix = basis[:, columns]
        condition = _condition_number(submatrix)
        if not np.isfinite(condition) or condition > config.condition_threshold:
            return None
        coefficients = ridge_scaffold_U(
            torch.from_numpy(submatrix), torch.from_numpy(target), config.ridge_lambda
        ).numpy()
        raw_error = fit_error_ridge(
            torch.from_numpy(submatrix), torch.from_numpy(target), torch.from_numpy(coefficients)
        )
        normalized_error = raw_error / max(float(np.mean(target**2)), 1e-30)
        complexity = int(sum(library.complexities[kept_indices[column]] for column in columns))
        score = (
            config.w_err * normalized_error
            + config.w_cond * _safe_log10(condition)
            + config.w_simp * complexity
        )
        return Candidate(
            terms=[kept_names[column] for column in columns],
            complexity=complexity,
            fit_error=float(normalized_error),
            fit_error_raw=float(raw_error),
            cond_scaled=float(condition),
            pre_score=float(score),
            layer=layer,
            U_ridge=coefficients,
        )

    first = make_candidate(mandatory, 0)
    beam = [first] if first is not None else []
    all_candidates.extend(beam)
    for layer in range(1, config.max_terms - len(mandatory) + 1):
        expanded: list[Candidate] = []
        for parent in beam:
            for name in kept_names:
                if name in parent.terms:
                    continue
                candidate = make_candidate(parent.terms + [name], layer)
                if candidate is not None:
                    expanded.append(candidate)
        if not expanded:
            break
        expanded.sort(key=lambda candidate: candidate.pre_score)
        beam = expanded[: config.beam_width]
        all_candidates.extend(expanded)

    unique = {}
    for candidate in all_candidates:
        key = frozenset(candidate.terms)
        if key not in unique or candidate.pre_score < unique[key].pre_score:
            unique[key] = candidate
    all_candidates = sorted(unique.values(), key=lambda candidate: candidate.pre_score)
    pareto = _pareto_filter(all_candidates) if config.pareto else []
    final = list(pareto)
    for candidate in all_candidates[: config.max_full_candidates]:
        if candidate not in final:
            final.append(candidate)
    final.sort(key=lambda candidate: candidate.pre_score)

    return {
        "candidates": all_candidates,
        "final_candidates": final[: config.max_full_candidates],
        "pruned_library": {
            "original_primitives": library.names,
            "kept_primitives": kept_names,
            "n_original": library.n_primitives,
            "n_kept": len(kept_names),
        },
        "generation_summary": {
            "n_states_sample": library.M,
            "n_evaluated_candidates": len(all_candidates),
            "n_pareto": len(pareto),
            "n_final": len(final[: config.max_full_candidates]),
            "max_terms": config.max_terms,
            "beam_width": config.beam_width,
            "layers_explored": max((candidate.layer for candidate in all_candidates), default=0),
        },
    }
