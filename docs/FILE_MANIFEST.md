# File manifest

The repository contains the following components.

The root metadata files (`README.md`, `requirements.txt`, `pyproject.toml`,
`.gitignore`, and the files under `docs/`) contain installation and usage
information for these components.

| Path | Content | Scope |
| --- | --- | --- |
| `configs/koopman_model.yaml` | Parametric Koopman model architecture and parameter-basis specification | Koopman model |
| `src/miklyap_tsa/koopman_model/network.py` | Direct parametric Koopman model, encoder, decoder, parameter-conditioned transform, and spectral blocks | Koopman model |
| `src/miklyap_tsa/analytical_lyapunov/basis.py` | Ordered 13-term analytical basis | Analytical Lyapunov |
| `src/miklyap_tsa/analytical_lyapunov/model.py` | Explicit parametric Lyapunov model and physical-field derivative | Analytical Lyapunov |
| `src/miklyap_tsa/analytical_lyapunov/vsg_dynamics.py` | Differentiable normalized VSG vector field and batch metadata contract | Analytical Lyapunov |
| `src/miklyap_tsa/analytical_lyapunov/candidates.py` | Staged candidate construction and ridge pre-scoring | Analytical Lyapunov |
| `src/miklyap_tsa/analytical_lyapunov/losses.py` | Pure loss and closed-form ridge functions | Analytical Lyapunov |
| `src/miklyap_tsa/analytical_lyapunov/metrics.py` | Basis conditioning, complexity, positivity, decrease, and composite metrics | Analytical Lyapunov |
| `src/miklyap_tsa/root_finding/core.py` | Root localization, polishing, neighborhood probing, and counterexample deduplication | Numerical falsification |
| `src/miklyap_tsa/smt_verification/core.py` | VSG model, analytical candidate, SMT formula construction, and dReal adapter | SMT verification |
| `src/miklyap_tsa/saddle_eigenfunction/core.py` | Local unstable eigenfunction basis, geometry, sampling, and fitting | Local saddle component |
