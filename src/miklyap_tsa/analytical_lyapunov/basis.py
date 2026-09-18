"""The ordered analytical basis used by the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn


PAPER_BASIS_TERMS = (
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


@dataclass(frozen=True)
class BasisTerm:
    name: str
    complexity: int
    term_type: str
    expression: str
    dependencies: tuple[str, ...]


_Compute = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

_TERM_METADATA = {
    "x1": (1, "linear", "x1", ("x1",)),
    "x2": (1, "linear", "x2", ("x2",)),
    "x3": (1, "linear", "x3", ("x3",)),
    "x1_sq": (2, "quadratic", "x1**2", ("x1",)),
    "x2_sq": (2, "quadratic", "x2**2", ("x2",)),
    "x3_sq": (2, "quadratic", "x3**2", ("x3",)),
    "x1_x2": (2, "cross", "x1*x2", ("x1", "x2")),
    "x1_x3": (2, "cross", "x1*x3", ("x1", "x3")),
    "x2_x3": (2, "cross", "x2*x3", ("x2", "x3")),
    "sin_x1": (3, "trig", "sin(x1)", ("x1",)),
    "one_minus_cos_x1": (3, "trig", "1-cos(x1)", ("x1",)),
    "x3_sin_x1": (4, "coupling", "x3*sin(x1)", ("x1", "x3")),
    "x3_one_minus_cos_x1": (4, "coupling", "x3*(1-cos(x1))", ("x1", "x3")),
}

_COMPUTES: dict[str, _Compute] = {
    "x1": lambda x1, x2, x3: x1,
    "x2": lambda x1, x2, x3: x2,
    "x3": lambda x1, x2, x3: x3,
    "x1_sq": lambda x1, x2, x3: x1 * x1,
    "x2_sq": lambda x1, x2, x3: x2 * x2,
    "x3_sq": lambda x1, x2, x3: x3 * x3,
    "x1_x2": lambda x1, x2, x3: x1 * x2,
    "x1_x3": lambda x1, x2, x3: x1 * x3,
    "x2_x3": lambda x1, x2, x3: x2 * x3,
    "sin_x1": lambda x1, x2, x3: torch.sin(x1),
    "one_minus_cos_x1": lambda x1, x2, x3: 1.0 - torch.cos(x1),
    "x3_sin_x1": lambda x1, x2, x3: x3 * torch.sin(x1),
    "x3_one_minus_cos_x1": lambda x1, x2, x3: x3 * (1.0 - torch.cos(x1)),
}


def _term(name: str) -> BasisTerm:
    complexity, term_type, expression, dependencies = _TERM_METADATA[name]
    return BasisTerm(name, complexity, term_type, expression, dependencies)


class BasisRegistry:
    """Ordered subset of the fixed paper basis."""

    def __init__(self, names: Sequence[str] = PAPER_BASIS_TERMS) -> None:
        names = tuple(str(name) for name in names)
        unknown = [name for name in names if name not in _COMPUTES]
        if unknown:
            raise ValueError(f"Unknown paper basis terms: {unknown}")
        if len(set(names)) != len(names):
            raise ValueError("Basis term names must be unique.")
        self._names = names
        self._terms = tuple(_term(name) for name in names)

    @classmethod
    def paper(cls) -> "BasisRegistry":
        return cls(PAPER_BASIS_TERMS)

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "BasisRegistry":
        return cls(names)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def terms(self) -> list[BasisTerm]:
        return list(self._terms)

    @property
    def output_dim(self) -> int:
        return len(self._names)

    def compute_unscaled(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != 3:
            raise ValueError("The state tensor must have last dimension 3.")
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        return torch.stack(
            [_COMPUTES[name](x1, x2, x3) for name in self._names], dim=-1
        )

    def compute_rms_scales(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        rms = self.compute_unscaled(x).square().mean(dim=0).sqrt()
        return torch.where(rms > eps, rms, torch.ones_like(rms))

    def complexity_total(self) -> int:
        return int(sum(term.complexity for term in self._terms))

    def term_counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for term in self._terms:
            counts[term.term_type] = counts.get(term.term_type, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "terms": [
                {
                    "name": term.name,
                    "complexity": term.complexity,
                    "term_type": term.term_type,
                    "expression": term.expression,
                    "dependencies": list(term.dependencies),
                    "vanishes_at_zero": True,
                }
                for term in self._terms
            ],
            "output_dim": self.output_dim,
            "complexity_total": self.complexity_total(),
            "operation_count": self.complexity_total(),
            "term_counts_by_type": self.term_counts_by_type(),
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_unscaled(x)


class BasisModel(nn.Module):
    """Differentiable basis with optional fixed RMS column scales."""

    def __init__(
        self,
        registry: BasisRegistry | None = None,
        rms_scales: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry or BasisRegistry.paper()
        scales = (
            torch.ones(self.registry.output_dim)
            if rms_scales is None
            else torch.as_tensor(rms_scales).detach().clone()
        )
        if scales.shape != (self.registry.output_dim,):
            raise ValueError("rms_scales must match the basis dimension.")
        self.register_buffer("rms_scales", scales)

    @classmethod
    def paper(cls, rms_scales: torch.Tensor | None = None) -> "BasisModel":
        return cls(BasisRegistry.paper(), rms_scales)

    def fit_rms(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        scales = self.registry.compute_rms_scales(x, eps=eps)
        self.rms_scales.copy_(scales.to(self.rms_scales))
        return self.rms_scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.registry.compute_unscaled(x) / self.rms_scales

    def compute_unscaled(self, x: torch.Tensor) -> torch.Tensor:
        return self.registry.compute_unscaled(x)

    @property
    def output_dim(self) -> int:
        return self.registry.output_dim

    @property
    def names(self) -> list[str]:
        return self.registry.names

    def to_dict(self) -> dict:
        payload = self.registry.to_dict()
        payload["rms_scales"] = self.rms_scales.detach().cpu().tolist()
        return payload
