"""Differentiable normalized VSG vector field."""

from __future__ import annotations

from dataclasses import dataclass

import torch


NETWORK_PARAMETER_NAMES = (
    "Pref_pu",
    "Qref_pu",
    "R1_pu",
    "X1_pu",
    "a_P",
    "a_D",
    "a_Q",
    "a_V",
)

PHYSICAL_PARAMETER_NAMES = NETWORK_PARAMETER_NAMES + ("Us_pu", "wb")
STATE_NAMES = ("delta_norm", "omega_pu_norm", "E_norm")


@dataclass
class BatchMeta:
    """Per-sample physical metadata for a batch."""

    Pref_pu: torch.Tensor
    Qref_pu: torch.Tensor
    R1_pu: torch.Tensor
    X1_pu: torch.Tensor
    a_P: torch.Tensor
    a_D: torch.Tensor
    a_Q: torch.Tensor
    a_V: torch.Tensor
    Us_pu: torch.Tensor
    wb: torch.Tensor
    x_eq: torch.Tensor
    scales: torch.Tensor

    def to(self, device: torch.device | str) -> "BatchMeta":
        return BatchMeta(
            Pref_pu=self.Pref_pu.to(device),
            Qref_pu=self.Qref_pu.to(device),
            R1_pu=self.R1_pu.to(device),
            X1_pu=self.X1_pu.to(device),
            a_P=self.a_P.to(device),
            a_D=self.a_D.to(device),
            a_Q=self.a_Q.to(device),
            a_V=self.a_V.to(device),
            Us_pu=self.Us_pu.to(device),
            wb=self.wb.to(device),
            x_eq=self.x_eq.to(device),
            scales=self.scales.to(device),
        )


def electrical_power_pu_torch(
    delta: torch.Tensor,
    e_pu: torch.Tensor,
    r1_pu: torch.Tensor,
    x1_pu: torch.Tensor,
    us_pu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return active and reactive per-unit power using real arithmetic."""

    real_voltage = e_pu * torch.cos(delta)
    imag_voltage = e_pu * torch.sin(delta)
    real_difference = real_voltage - us_pu
    imag_difference = imag_voltage
    denominator = r1_pu.square() + x1_pu.square()
    current_real = (real_difference * r1_pu + imag_difference * x1_pu) / denominator
    current_imag = (imag_difference * r1_pu - real_difference * x1_pu) / denominator
    active = real_voltage * current_real + imag_voltage * current_imag
    reactive = imag_voltage * current_real - real_voltage * current_imag
    return active, reactive


def vsg_rhs_pu_torch(xi: torch.Tensor, meta: BatchMeta) -> torch.Tensor:
    """Evaluate the physical per-unit VSG vector field."""

    delta = xi[:, 0]
    omega_pu = xi[:, 1]
    e_pu = xi[:, 2]
    active, reactive = electrical_power_pu_torch(
        delta, e_pu, meta.R1_pu, meta.X1_pu, meta.Us_pu
    )
    d_delta = meta.wb * (omega_pu - 1.0)
    d_omega = meta.a_P * (meta.Pref_pu - active) - meta.a_D * (omega_pu - 1.0)
    d_e = meta.a_Q * (meta.Qref_pu - reactive) - meta.a_V * (e_pu - meta.Us_pu)
    return torch.stack([d_delta, d_omega, d_e], dim=-1)


def normalized_rhs(x: torch.Tensor, meta: BatchMeta) -> torch.Tensor:
    """Evaluate ``dot_x`` in normalized equilibrium-offset coordinates."""

    physical_state = meta.x_eq + meta.scales * x
    return vsg_rhs_pu_torch(physical_state, meta) / meta.scales
