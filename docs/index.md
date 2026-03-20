# Linear Optics Toolkit

This repository provides a numerical framework to evaluate **hierarchical invariants** of photonic states under passive linear optics.

The implementation supports:

- global Jordan hierarchy invariants,
- Schur-sector-local hierarchy invariants,
- multiplicity-local hierarchy invariants (basis-convention dependent but deterministic in this code).
- lifted one-body observable measurement (expectation, variance, distribution, sampling).
- a bosonic symmetric Fock backend for `gram=1` state construction/performance.

## Scope

Current prototype focus:

- first-quantized external space `(C^m)^{\otimes n}`,
- exact Schur projector construction for general `n` (computationally heavier as `n` grows),
- reproducible decomposition for small-scale numerical checks accompanying theory papers.

## Why This Is Useful For Papers

- turns abstract hierarchy statements into executable checks,
- makes assumptions explicit (scope, basis convention, normalization),
- provides a compact, test-backed artifact for supplementary material.

See:

- [Quick Start](quickstart.md)
- [API Guide](api.md)
- [Theory Notes](theory.md)
- [Performance](performance.md)
- [Reproducibility](reproducibility.md)
- [References](references.md)
