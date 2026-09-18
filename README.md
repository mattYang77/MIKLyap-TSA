# MIKLyap-TSA

Code for a paper currently under review.

## Scope

This repository contains:

- the direct parametric Koopman model;
- the 13-term analytical Lyapunov basis and core functions;
- numerical root-finding and local decrease-margin probes;
- VSG dynamics, SMT formula construction, and a dReal adapter;
- the local unstable Koopman eigenfunction core.

## Model specification

The model architecture is specified in
[`configs/koopman_model.yaml`](configs/koopman_model.yaml).
The state is

```text
x = [x1, x2, x3] = [delta_norm, omega_pu_norm, E_norm]
```

The network condition vector is

```text
p = [Pref_pu, Qref_pu, R1_pu, X1_pu, a_P, a_D, a_Q, a_V]
```

The physical state is `[delta, omega_pu, E_pu]`. The normalized state is
`x = (state - x_eq) / [3.5, 15 / wb, 25 / Us]`, where `x_eq` is the VSG
equilibrium. The default SI parameter group is
`fn_hz=50`, `Us_v=311`, `Pref_w=20000`, `Qref_var=0`, `J=5`, `D=8`,
`K_v=12`, `D_q=166`, `R1_ohm=0.23`, and `L1_h=0.008`.

The direct parametric Koopman model accepts `x` with shape `(B, T, 3)` and `p` with shape
`(B, 8)`. Its forward outputs are the transformed features with shape
`(B, T, 36)`, the nine-step evolved features, the decoded states, and the
spectral blocks.

The analytical basis is fixed to the following ordered columns:

```text
x1, x2, x3, x1_sq, x2_sq, x3_sq, x1_x2, x1_x3, x2_x3,
sin_x1, one_minus_cos_x1, x3_sin_x1, x3_one_minus_cos_x1
```

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The dReal executable is an external dependency. A macOS build is available at
[`mattYang77/dreal-macos15`](https://github.com/mattYang77/dreal-macos15).

## Reference dReal timing data

The following table records the all-trajectory-point convex-hull comparison
from the source project. Values are raw 20-run solver times in milliseconds,
using `delta=1e-4`, one dReal worker, and a 30 s per-query timeout. The query
was the common decrease-violation query `dot(V) > 0`. The recorded platform
was macOS arm64 with dReal 4.21.06.2.

| Candidate | Status | Mean (ms) | Min (ms) | Max (ms) | Median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Analytic 13-term | delta-sat | 91.77 | 82.64 | 106.00 | 90.94 |
| Huang 2022 [1] `[3,6,1]` | unsat | 523.45 | 492.36 | 545.60 | 523.00 |
| Liu 2024 [2] `[3,3,3,1]` | delta-sat | 1535.23 | 1431.60 | 1873.43 | 1477.52 |
| Liu 2025 [3] `[3,6,6,1]` | unsat | 1248.36 | 1140.38 | 1466.13 | 1198.02 |

Here `delta-sat` means that dReal returned a violation witness at the selected
precision. `unsat` means that no such witness was found for that query at the
selected precision. These are solver-status and encoding-time comparisons,
not comparisons of learned Lyapunov quality.

### References

[1] T. Huang, S. Gao, and L. Xie, "A Neural Lyapunov Approach to Transient Stability Assessment of Power Electronics-Interfaced Networked Microgrids," *IEEE Transactions on Smart Grid*, vol. 13, no. 1, pp. 106-118, Jan. 2022, doi: 10.1109/TSG.2021.3117889.

[2] Y. Liu, J. Zhang, Y. Liu, M. Yang, S. Chen, L. Zhou, and Y. Wang, "An Improved Neural Lyapunov Method for Transient Stability Assessment of Networked Microgrids," *IEEE Transactions on Smart Grid*, vol. 15, no. 2, pp. 1410-1422, Mar. 2024, doi: 10.1109/TSG.2023.3301855.

[3] Y. Liu, L. Zhou, S. Chen, J. Zhang, and Y. Wang, "A Model Ensemble Framework Containing a Novel Trial Solution Structure for the Transient Stability Analysis of VSGs Considering Voltage Dynamics," *IEEE Transactions on Power Systems*, vol. 40, no. 1, pp. 1105-1117, Jan. 2025, doi: 10.1109/TPWRS.2024.3419752.

## File map

- [`docs/FILE_MANIFEST.md`](docs/FILE_MANIFEST.md) lists every file in this repository.
- [`docs/PAPER_CODE_MAPPING.md`](docs/PAPER_CODE_MAPPING.md) records the broad paper correspondence.
