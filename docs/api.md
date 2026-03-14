# API Guide

The package is organized by domain (inspired by scientific toolboxes such as `qoptcraft`).

## Main Entry Point

- `photonic_jordan.PhotonicSystem`

Core user workflows:

1. Construct a system (`m_ext`, `n_particles`).
2. Build states from Gram models or explicit density matrices.
3. Evaluate invariants globally or on a local scope.

## State Layer

- `photonic_jordan.StateBuilder`
- `photonic_jordan.PhotonicState`
- `photonic_jordan.StateInvariantView`
- `photonic_jordan.InvariantReport`

Common methods:

- `StateBuilder.from_modes_and_gram`
- `PhotonicState.density_matrix`
- `PhotonicState.project_jordan`
- `StateInvariantView.I_exact`
- `StateInvariantView.I_cumulative`
- `PhotonicState.analyze`

Representation note:

- `PhotonicState` does not store a fixed representation label.
- Use `rho.density_matrix(rep='tensor'|'schur')` to retrieve matrix forms of the same state.

## Decomposition Layer

- `photonic_jordan.SchurWeylDecomposition`
- `photonic_jordan.SymmetricGroupProjectors`

Common methods:

- `SchurWeylDecomposition.sector_projector`
- `SchurWeylDecomposition.multiplicity_projector`
- `PhotonicSystem.sector_projector`
- `PhotonicSystem.multiplicity_projector`

## Hierarchy Layer

- `photonic_jordan.JordanFiltration`
- `photonic_jordan.InvariantEngine`

Common methods:

- `JordanFiltration.build`
- `JordanFiltration.apply_projector_layer`
- `JordanFiltration.apply_projector_cumulative`
- `InvariantEngine.commutator_error_layer`
- `InvariantEngine.commutator_error_cumulative`

## Documentation Conventions

Public classes and methods use a unified docstring structure where applicable:

- `Parameters`
- `Returns`
- `Notes`
- `References`

This is designed so paper supplementary readers can map code calls to precise definitions.
