# Paper-to-code mapping

| Manuscript component | Implementation |
| --- | --- |
| Shared encoder, parameter-conditioned transform, and direct spectral evolution | `src/miklyap_tsa/koopman_model/network.py` and `configs/koopman_model.yaml` |
| Ordered analytical structure | `src/miklyap_tsa/analytical_lyapunov/basis.py` |
| Physical Lyapunov realization | `src/miklyap_tsa/analytical_lyapunov/model.py` and `vsg_dynamics.py` |
| Candidate construction | `src/miklyap_tsa/analytical_lyapunov/candidates.py`, `losses.py`, and `metrics.py` |
| Root-guided numerical falsification | `src/miklyap_tsa/root_finding/core.py` |
| SMT verification | `src/miklyap_tsa/smt_verification/core.py` |
| Local unstable Koopman eigenfunction | `src/miklyap_tsa/saddle_eigenfunction/core.py` |

The code uses the paper state convention

```text
x = [x1, x2, x3] = [delta_norm, omega_pu_norm, E_norm]
```

The 13 basis columns are kept in the paper order. Root finding is a numerical
falsification component; an `unsat` result from a dReal query is the SMT-side
result at the selected precision, not a root-finding result.
