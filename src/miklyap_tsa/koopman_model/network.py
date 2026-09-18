"""Direct parametric Koopman model."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import torch
from torch import nn


def get_activation_function(name: str) -> nn.Module:
    """Return a supported hidden-layer activation."""

    key = str(name).lower()
    activations = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.01),
        "silu": nn.SiLU,
        "gelu": lambda: nn.GELU(approximate="tanh"),
    }
    if key not in activations:
        raise ValueError(f"Unsupported activation function: {name}")
    return activations[key]()


def _parse_indices(value: int | str | Sequence[int] | None, p_dim: int) -> list[int]:
    if value is None:
        indices = list(range(p_dim))
    elif isinstance(value, int):
        indices = [value]
    elif isinstance(value, str):
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        indices = [int(item) for item in value]
    if not indices:
        raise ValueError("At least one parameter-basis index is required.")
    if any(index < 0 or index >= p_dim for index in indices):
        raise ValueError(f"Parameter-basis indices must lie in [0, {p_dim}).")
    return indices


class ParameterBasisLibrary:
    """Polynomial parameter basis used by the structured transform."""

    def __init__(
        self,
        p_dim: int,
        selected_idx: int | str | Sequence[int] | None = None,
        parameter_names: Sequence[str] | str | None = None,
        include_constant: bool = True,
        include_linear: bool = True,
        include_square: bool = False,
        include_pairwise: bool = True,
    ) -> None:
        self.p_dim = int(p_dim)
        self.selected_idx = _parse_indices(selected_idx, self.p_dim)
        if parameter_names is None:
            self.parameter_names = None
        elif isinstance(parameter_names, str):
            self.parameter_names = [item.strip() for item in parameter_names.split(",")]
        else:
            self.parameter_names = [str(item) for item in parameter_names]
        self.include_constant = bool(include_constant)
        self.include_linear = bool(include_linear)
        self.include_square = bool(include_square)
        self.include_pairwise = bool(include_pairwise)

    def selected_names(self) -> list[str]:
        if self.parameter_names is None:
            return [f"p{index}" for index in self.selected_idx]
        if len(self.parameter_names) == len(self.selected_idx):
            return list(self.parameter_names)
        return [
            self.parameter_names[index]
            if index < len(self.parameter_names)
            else f"p{index}"
            for index in self.selected_idx
        ]

    def basis_names(self) -> list[str]:
        names = self.selected_names()
        result: list[str] = []
        if self.include_constant:
            result.append("1")
        if self.include_linear:
            result.extend(names)
        if self.include_square:
            result.extend(f"{name}^2" for name in names)
        if self.include_pairwise:
            result.extend(
                f"{names[i]}*{names[j]}"
                for i, j in combinations(range(len(names)), 2)
            )
        if not result:
            raise ValueError("The parameter basis cannot be empty.")
        return result

    @property
    def output_dim(self) -> int:
        return len(self.basis_names())

    def __call__(self, p: torch.Tensor, return_names: bool = False):
        squeeze = p.dim() == 1
        if squeeze:
            p = p.unsqueeze(0)
        if p.shape[-1] != self.p_dim:
            raise ValueError(f"Expected parameter dimension {self.p_dim}, got {p.shape[-1]}.")

        selected = p[..., self.selected_idx]
        terms: list[torch.Tensor] = []
        if self.include_constant:
            terms.append(torch.ones_like(selected[..., :1]))
        if self.include_linear:
            terms.append(selected)
        if self.include_square:
            terms.append(selected.square())
        if self.include_pairwise:
            pairs = [
                selected[..., i : i + 1] * selected[..., j : j + 1]
                for i, j in combinations(range(selected.shape[-1]), 2)
            ]
            if pairs:
                terms.append(torch.cat(pairs, dim=-1))
        basis = torch.cat(terms, dim=-1)
        if squeeze:
            basis = basis.squeeze(0)
        if return_names:
            return basis, self.basis_names()
        return basis


class StructuredLinearTransform(nn.Module):
    """Parameter-conditioned linear map ``T(p)``."""

    def __init__(
        self,
        latent_dim: int,
        p_dim: int,
        selected_idx: int | str | Sequence[int] | None = None,
        parameter_names: Sequence[str] | str | None = None,
        include_constant: bool = True,
        include_linear: bool = True,
        include_square: bool = False,
        include_pairwise: bool = True,
        init_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.init_scale = float(init_scale)
        self.basis = ParameterBasisLibrary(
            p_dim=p_dim,
            selected_idx=selected_idx,
            parameter_names=parameter_names,
            include_constant=include_constant,
            include_linear=include_linear,
            include_square=include_square,
            include_pairwise=include_pairwise,
        )
        self.transform_bank = nn.Parameter(
            torch.zeros(self.basis.output_dim, self.latent_dim, self.latent_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.transform_bank.zero_()
            eye = torch.eye(
                self.latent_dim,
                device=self.transform_bank.device,
                dtype=self.transform_bank.dtype,
            )
            if self.basis.include_constant:
                self.transform_bank[0].copy_(eye)
                if self.transform_bank.shape[0] > 1:
                    self.transform_bank[1:].copy_(
                        self.init_scale * torch.randn_like(self.transform_bank[1:])
                    )
            else:
                self.transform_bank.copy_(
                    self.init_scale * torch.randn_like(self.transform_bank)
                )

    @property
    def basis_dim(self) -> int:
        return self.basis.output_dim

    def basis_names(self) -> list[str]:
        return self.basis.basis_names()

    def parameter_basis(self, p: torch.Tensor, return_names: bool = False):
        return self.basis(p, return_names=return_names)

    def forward(self, p: torch.Tensor, return_basis: bool = False, return_names: bool = False):
        beta = self.parameter_basis(p)
        if beta.dim() == 1:
            beta = beta.unsqueeze(0)
        transform = torch.einsum("br,rij->bij", beta, self.transform_bank)
        if return_basis and return_names:
            return transform, beta, self.basis_names()
        if return_basis:
            return transform, beta
        return transform

    def apply(
        self,
        z: torch.Tensor,
        p: torch.Tensor | None = None,
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if transform is None:
            if p is None:
                raise ValueError("Provide either p or transform.")
            transform = self.forward(p)
        if z.dim() < 2:
            raise ValueError("The latent tensor must include a batch dimension.")
        if transform.dim() == 2:
            transform = transform.unsqueeze(0)
        if transform.shape[0] == 1 and z.shape[0] > 1:
            transform = transform.expand(z.shape[0], -1, -1)
        if transform.shape[0] != z.shape[0]:
            raise ValueError(
                f"Batch mismatch: latent={tuple(z.shape)}, transform={tuple(transform.shape)}."
            )
        return torch.einsum("bij,b...j->b...i", transform, z)


class Encoder(nn.Module):
    """Shared state encoder for 2-D states or time-window tensors."""

    def __init__(self, layers: Sequence[int], shifts_input: int, activation: str) -> None:
        super().__init__()
        self.shifts_input = int(shifts_input)
        modules: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            modules.append(nn.Linear(int(in_dim), int(out_dim)))
            if index < len(layers) - 2:
                modules.append(get_activation_function(activation))
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.network(x)
        if x.dim() != 3:
            raise ValueError("Encoder input must have shape (B, D) or (B, T, D).")
        if x.shape[1] < self.shifts_input + 1:
            raise ValueError("The input time window is shorter than shifts_input + 1.")
        return torch.stack(
            [self.network(x[:, step, :]) for step in range(self.shifts_input + 1)],
            dim=1,
        )


class Decoder(nn.Module):
    """Shared state decoder."""

    def __init__(self, layers: Sequence[int], activation: str) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            modules.append(nn.Linear(int(in_dim), int(out_dim)))
            if index < len(layers) - 2:
                modules.append(get_activation_function(activation))
        self.network = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class MIKoopmanNet(nn.Module):
    """Direct parameter-conditioned Koopman model used by the paper.

    The model implements

    ``z = encoder(x)``, ``phi = T(p) z``, and ``phi_next = K(p) phi``.
    """

    def __init__(
        self,
        encoder_layer: Sequence[int],
        decoder_layer: Sequence[int],
        act_type: str,
        num_complex_pairs: int,
        num_real: int,
        delta_t: float,
        shifts: int,
        shifts_input: int,
        shifts_pred: int,
        widths_omega_complex: Sequence[int] | Sequence[Sequence[int]],
        widths_omega_real: Sequence[int] | Sequence[Sequence[int]],
        p_dim: int,
        param_basis_indices: int | str | Sequence[int] | None = None,
        param_basis_names: Sequence[str] | str | None = None,
        param_basis_include_constant: bool = True,
        param_basis_include_linear: bool = True,
        param_basis_include_square: bool = False,
        param_basis_include_pairwise: bool = True,
        param_transform_init_scale: float = 1e-2,
        spectral_cond_dim: int = 32,
    ) -> None:
        super().__init__()
        if len(encoder_layer) < 2 or len(decoder_layer) < 2:
            raise ValueError("Encoder and decoder need at least one linear layer.")
        self.num_complex_pairs = int(num_complex_pairs)
        self.num_real = int(num_real)
        self.delta_t = float(delta_t)
        self.shifts = int(shifts)
        self.shifts_input = int(shifts_input)
        self.shifts_pred = int(shifts_pred)
        self.scaffold_dim = int(encoder_layer[-1])
        self.p_dim = int(p_dim)
        self.spectral_cond_dim = int(spectral_cond_dim)

        self.encoder = Encoder(encoder_layer, self.shifts_input, act_type)
        self.decoder = Decoder(decoder_layer, act_type)
        self.transform = StructuredLinearTransform(
            latent_dim=self.scaffold_dim,
            p_dim=self.p_dim,
            selected_idx=param_basis_indices,
            parameter_names=param_basis_names,
            include_constant=param_basis_include_constant,
            include_linear=param_basis_include_linear,
            include_square=param_basis_include_square,
            include_pairwise=param_basis_include_pairwise,
            init_scale=param_transform_init_scale,
        )
        self.spectral_conditioner = self._build_mlp(
            [self.transform.basis_dim, self.spectral_cond_dim, self.spectral_cond_dim],
            act_type,
        )

        complex_widths = self._expand_widths(
            widths_omega_complex, self.num_complex_pairs
        )
        real_widths = self._expand_widths(widths_omega_real, self.num_real)
        self.omega_complex_nets = nn.ModuleList(
            self._build_mlp(
                self._conditioned_widths(widths, self.spectral_cond_dim, 2),
                act_type,
            )
            for widths in complex_widths
        )
        self.omega_real_nets = nn.ModuleList(
            self._build_mlp(
                self._conditioned_widths(widths, self.spectral_cond_dim, 1),
                act_type,
            )
            for widths in real_widths
        )

    @staticmethod
    def _expand_widths(widths, count: int) -> list[list[int]]:
        values = list(widths)
        if not values:
            raise ValueError("A spectral MLP width specification is required.")
        if isinstance(values[0], (int, float)):
            return [list(map(int, values)) for _ in range(count)]
        if len(values) != count:
            raise ValueError(f"Expected {count} spectral width lists, got {len(values)}.")
        return [list(map(int, row)) for row in values]

    @staticmethod
    def _conditioned_widths(widths: Sequence[int], condition_dim: int, output_dim: int) -> list[int]:
        values = list(widths)
        if len(values) < 2:
            raise ValueError("A spectral MLP needs input and output widths.")
        return [int(condition_dim), *map(int, values[1:-1]), int(output_dim)]

    @staticmethod
    def _build_mlp(widths: Iterable[int], activation: str) -> nn.Sequential:
        values = list(widths)
        modules: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(values[:-1], values[1:])):
            modules.append(nn.Linear(int(in_dim), int(out_dim)))
            if index < len(values) - 2:
                modules.append(get_activation_function(activation))
        return nn.Sequential(*modules)

    def encode_scaffold(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def scaffold(self, x: torch.Tensor) -> torch.Tensor:
        """Return the shared scaffold representation."""

        return self.encode_scaffold(x)

    def parameter_basis(self, p: torch.Tensor, return_names: bool = False):
        return self.transform.parameter_basis(p, return_names=return_names)

    def transform_matrix(self, p: torch.Tensor, return_basis: bool = False, return_names: bool = False):
        return self.transform(p, return_basis=return_basis, return_names=return_names)

    def transform_scaffold(
        self,
        z: torch.Tensor,
        p: torch.Tensor | None = None,
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.transform.apply(z, p=p, transform=transform)

    def parameter_condition(self, p: torch.Tensor) -> torch.Tensor:
        beta = self.parameter_basis(p)
        if beta.dim() == 1:
            beta = beta.unsqueeze(0)
        return self.spectral_conditioner(beta)

    def forward_omegas(self, p: torch.Tensor) -> list[torch.Tensor]:
        condition = self.parameter_condition(p)
        return [
            *[network(condition) for network in self.omega_complex_nets],
            *[network(condition) for network in self.omega_real_nets],
        ]

    @staticmethod
    def form_complex_conjugate_block(
        omegas_block: torch.Tensor, delta_t: float
    ) -> torch.Tensor:
        omega = omegas_block[:, 0]
        mu = omegas_block[:, 1]
        scale = torch.exp(mu * delta_t)
        cos_value = torch.cos(omega * delta_t)
        sin_value = torch.sin(omega * delta_t)
        row_one = torch.stack([scale * cos_value, -scale * sin_value], dim=1)
        row_two = torch.stack([scale * sin_value, scale * cos_value], dim=1)
        return torch.stack([row_one, row_two], dim=2)

    def varying_multiply(
        self, y: torch.Tensor, omegas: Sequence[torch.Tensor], delta_t: float
    ) -> torch.Tensor:
        complex_parts: list[torch.Tensor] = []
        for index in range(self.num_complex_pairs):
            pair = y[:, 2 * index : 2 * index + 2]
            block = self.form_complex_conjugate_block(omegas[index], delta_t)
            complex_parts.append(torch.matmul(pair.unsqueeze(1), block).squeeze(1))

        real_parts: list[torch.Tensor] = []
        offset = 2 * self.num_complex_pairs
        for index in range(self.num_real):
            column = y[:, offset + index : offset + index + 1]
            damping = omegas[self.num_complex_pairs + index][:, 0]
            real_parts.append(column * torch.exp(damping * delta_t).unsqueeze(1))

        parts = complex_parts + real_parts
        return torch.cat(parts, dim=1) if parts else y

    def encode_scaffold_and_phi(
        self, x: torch.Tensor, p: torch.Tensor, return_transform: bool = False
    ):
        z = self.encode_scaffold(x)
        transform = self.transform_matrix(p)
        phi = self.transform_scaffold(z, transform=transform)
        if return_transform:
            return z, phi, transform
        return z, phi

    def forward(self, x: torch.Tensor, p: torch.Tensor | None = None):
        if p is None:
            raise ValueError("The direct model requires the parameter vector p.")
        _, phi, _ = self.encode_scaffold_and_phi(x, p, return_transform=True)
        if phi.dim() != 3:
            raise ValueError("forward expects a time-window input with shape (B, T, D).")
        decoded = [self.decoder(phi[:, 0, :])]
        evolved = [phi[:, 0, :]]
        omegas = self.forward_omegas(p)
        current = phi[:, 0, :]
        for _ in range(self.shifts_input):
            current = self.varying_multiply(current, omegas, self.delta_t)
            evolved.append(current)
            decoded.append(self.decoder(current))
        return (
            phi,
            torch.stack(evolved, dim=1),
            torch.stack(decoded, dim=1),
            omegas,
        )

    def forward_once_for_infer(
        self,
        x: torch.Tensor,
        p: torch.Tensor | None = None,
        infer_times: int = 3,
        delta_t: float | None = None,
    ):
        if p is None:
            raise ValueError("The direct model requires the parameter vector p.")
        z = self.encode_scaffold(x)
        if z.dim() == 3:
            z = z[:, 0, :]
        phi = self.transform_scaffold(z, p=p)
        states = [self.decoder(phi)]
        features = [phi]
        omegas = self.forward_omegas(p)
        step = self.delta_t if delta_t is None else float(delta_t)
        for _ in range(int(infer_times)):
            phi = self.varying_multiply(phi, omegas, step)
            features.append(phi)
            states.append(self.decoder(phi))
        return features, states, [omegas for _ in range(int(infer_times))]

    def forward_once_for_gradient(
        self, z: torch.Tensor, deltat: float, p: torch.Tensor | None = None
    ) -> torch.Tensor:
        if p is None:
            raise ValueError("The direct model requires the parameter vector p.")
        omegas = self.forward_omegas(p)
        next_z = self.varying_multiply(z, omegas, float(deltat))
        return (next_z - z) / float(deltat)
