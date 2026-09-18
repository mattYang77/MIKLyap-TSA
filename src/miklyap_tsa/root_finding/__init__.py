"""Framework-free numerical root-finding and local falsification kernels."""

from .core import (
    FalsifierConfig,
    DotVField,
    cluster_points,
    dedupe_counterexamples,
    fibonacci_sphere_directions,
    find_stationary_points,
    find_zero_roots,
    generate_seeds,
    probe_root_neighborhoods,
)

__all__ = [
    "FalsifierConfig",
    "DotVField",
    "cluster_points",
    "dedupe_counterexamples",
    "fibonacci_sphere_directions",
    "find_stationary_points",
    "find_zero_roots",
    "generate_seeds",
    "probe_root_neighborhoods",
]
