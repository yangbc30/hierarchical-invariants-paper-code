# Photonic Jordan Framework

This repository provides a numerical framework to evaluate **hierarchical invariants** of photonic states under passive linear optics.

The implementation supports:

- global Jordan hierarchy invariants,
- Schur-sector-local hierarchy invariants,
- multiplicity-local hierarchy invariants (basis-convention dependent but deterministic in this code).
- lifted one-body observable measurement (expectation, variance, distribution, sampling).

## Scope

Current prototype focus:

- first-quantized external space `(C^m)^{\otimes n}`,
- exact support for Schur projectors at `n=2,3`,
- reproducible decomposition for small-scale numerical checks accompanying theory papers.

## Why This Is Useful For Papers

- turns abstract hierarchy statements into executable checks,
- makes assumptions explicit (scope, basis convention, normalization),
- provides a compact, test-backed artifact for supplementary material.

See:

- [Quick Start](quickstart.md)
- [API Guide](api.md)
- [Theory Notes](theory.md)
- [Reproducibility](reproducibility.md)
- [References](references.md)
