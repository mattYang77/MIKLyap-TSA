"""Explicit parametric Lyapunov realization."""

from __future__ import annotations

import torch
from torch import nn

from .basis import BasisModel
from .vsg_dynamics import BatchMeta, normalized_rhs


class ExplicitLyapunov(nn.Module):
    """Learnable coefficient matrix over the fixed analytical basis."""

    def __init__(
        self,
        basis: BasisModel,
        latent_dim: int,
        init: str = "xavier",
    ) -> None:
        super().__init__()
        self.basis = basis
        self.latent_dim = int(latent_dim)
        self.basis_dim = basis.output_dim
        coefficients = torch.empty(self.latent_dim, self.basis_dim)
        if init == "xavier":
            nn.init.xavier_uniform_(coefficients)
        elif init == "zeros":
            nn.init.zeros_(coefficients)
        elif init == "small":
            nn.init.normal_(coefficients, std=1e-3)
        else:
            raise ValueError(f"Unknown initialization: {init}")
        self.U = nn.Parameter(coefficients)
        for parameter in self.basis.parameters():
            parameter.requires_grad_(False)

    def b(self, x: torch.Tensor) -> torch.Tensor:
        return self.basis(x)

    def scaffold_hat(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bm,lm->bl", self.b(x), self.U)

    @staticmethod
    def _teacher_transform(teacher, p: torch.Tensor) -> torch.Tensor:
        transform = teacher.transform_matrix(p)
        if transform.dim() == 2:
            transform = transform.unsqueeze(0)
        return transform

    def M(self, p: torch.Tensor, teacher) -> torch.Tensor:
        transform = self._teacher_transform(teacher, p)
        transformed = torch.einsum("bij,jm->bim", transform, self.U)
        mask_fn = getattr(teacher, "stable_mode_mask", None)
        if mask_fn is not None:
            transformed = transformed * mask_fn(p).to(transformed.dtype).unsqueeze(-1)
        return torch.einsum("bim,bin->bmn", transformed, transformed)

    def phi(self, x: torch.Tensor, p: torch.Tensor, teacher) -> torch.Tensor:
        transform = self._teacher_transform(teacher, p)
        scaffold = self.scaffold_hat(x)
        phi = torch.einsum("bij,bj->bi", transform, scaffold)
        mask_fn = getattr(teacher, "apply_mode_mask", None)
        return mask_fn(phi, p) if mask_fn is not None else phi

    def V(self, x: torch.Tensor, p: torch.Tensor, teacher) -> torch.Tensor:
        basis = self.b(x)
        matrix = self.M(p, teacher)
        return torch.einsum("bm,bmn,bn->b", basis, matrix, basis)

    def V_and_Vdot(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        meta: BatchMeta,
        teacher,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``V``, its physical-field derivative, and the leaf state."""

        with torch.enable_grad():
            x_leaf = x.detach().clone().requires_grad_(True)
            value = self.V(x_leaf, p, teacher)
            grad_value = torch.autograd.grad(
                value,
                x_leaf,
                grad_outputs=torch.ones_like(value),
                create_graph=self.training,
                retain_graph=True,
            )[0]
            field = normalized_rhs(x_leaf, meta)
            derivative = (grad_value * field).sum(dim=-1)
        return value, derivative, x_leaf

    def V_at_origin(self, p: torch.Tensor, teacher) -> torch.Tensor:
        origin = torch.zeros(p.shape[0], 3, device=p.device, dtype=p.dtype)
        return self.V(origin, p, teacher)

    def M_symmetry_error(self, p: torch.Tensor, teacher) -> torch.Tensor:
        matrix = self.M(p, teacher)
        return (matrix - matrix.transpose(-1, -2)).abs().amax(dim=(-2, -1))

    def M_min_eigenvalue(self, p: torch.Tensor, teacher) -> torch.Tensor:
        matrix = self.M(p, teacher)
        symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
        try:
            eigenvalues = torch.linalg.eigvalsh(symmetric)
        except RuntimeError:
            eigenvalues = torch.linalg.eigvalsh(symmetric.detach().cpu()).to(symmetric.device)
        return eigenvalues.amin(dim=-1)
