# API Guide

The package is organized by domain (inspired by scientific toolboxes such as `qoptcraft`).

## Main Entry Point

- `photonic_jordan.PhotonicSystem`

Core user workflows:

1. Construct a system (`m_ext`, `n_particles`).
2. Build states from Gram models or explicit density matrices.
3. Evaluate invariants globally or on a local scope.
4. Measure lifted one-body observables.

Caching options on construction:

- `PhotonicSystem(..., auto_cache=True, cache_dir=...)`
- `PhotonicSystem.configure_global_cache(enabled=..., cache_dir=...)`

Useful Fock backend helpers on system:

- `PhotonicSystem.total_unitary_fock_from_single_particle`
- `PhotonicSystem.evolve_density_fock`
- `PhotonicSystem.save_fock_cache`
- `PhotonicSystem.load_fock_cache`

## State Layer

- `photonic_jordan.StateBuilder`
- `photonic_jordan.PhotonicState`
- `photonic_jordan.StateInvariantView`
- `photonic_jordan.StateMeasurementView`
- `photonic_jordan.InvariantReport`

Common methods:

- `StateBuilder.from_modes_and_gram`
- `PhotonicState.density_matrix`
- `PhotonicState.project_jordan`
- `StateInvariantView.I_exact`
- `StateInvariantView.I_cumulative`
- `StateMeasurementView.expectation`
- `StateMeasurementView.distribution`
- `PhotonicState.analyze`

Representation note:

- `PhotonicState` does not store a fixed representation label.
- Use `rho.density_matrix(rep='tensor'|'fock'|'schur')` to retrieve matrix forms of the same state.
- For `gram=1`, `StateBuilder.from_modes_and_gram` uses a symmetric Fock backend first.

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

Fock backend note:

- When a state is currently backed by `fock` and query scope is global, `I_exact/I_cumulative` are evaluated in the symmetric hierarchy (`lambda=(n,)` sector-local equivalent).

## Measurement Layer

- `photonic_jordan.ObservableFactory`
- `photonic_jordan.SingleParticleObservable`
- `photonic_jordan.ObservableDistribution`

Common methods:

- `ObservableFactory.from_matrix`
- `ObservableFactory.number`
- `ObservableFactory.sigma_x`
- `ObservableFactory.sigma_y`
- `ObservableFactory.sigma_z`
- `SingleParticleObservable.expectation`
- `SingleParticleObservable.variance`
- `SingleParticleObservable.distribution`
- `SingleParticleObservable.sample`

Scope semantics are shared with invariants:

- global
- `sector=lambda`
- `multiplicity=(lambda, a)` (or legacy alias `copy=...`)

Scoped distribution behavior:

- `conditional=False`: probabilities sum to `Tr(Q rho)`
- `conditional=True`: probabilities sum to `1`

Current non-goal:

- generic POVM support is intentionally not included in this iteration.

## Space Layer

- `photonic_jordan.LabeledTensorSpace`
- `photonic_jordan.BosonicFockSpace`

The Fock backend is used for fully indistinguishable (`gram=1`) bosonic input states and can be requested explicitly via `rho.density_matrix(rep='fock')`.
It also supports native lifted evolution in symmetric space.

## Documentation Conventions

Public classes and methods use a unified docstring structure where applicable:

- `Parameters`
- `Returns`
- `Notes`
- `References`

This is designed so paper supplementary readers can map code calls to precise definitions.
