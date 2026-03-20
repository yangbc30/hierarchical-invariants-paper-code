# Linear Optics Toolkit (Research Prototype)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangbc30/hierarchical-invariants-paper-code/blob/main/notebooks/demo_colab.ipynb)

Numerical framework for checking hierarchical Jordan invariants of multiphoton states under passive linear optics, with global/sector/multiplicity scopes.

## Project Goal

- Keep a simple user API centered on `PhotonicSystem`, `PhotonicState`, and `InvariantReport`.
- Provide reproducible numerical checks for theory statements used in papers.
- Support Schur-aware decomposition and local Jordan hierarchies.
- Support one-body observable measurement through `system.observable` and `state.measure`.

## Core Design Contracts

### Scope conventions

Exactly one scope is active per invariant query:

1. global
2. sector-local: `sector=lambda`
3. multiplicity-local: `multiplicity=(lambda, a)`

### Canonical vs basis-dependent objects

Canonical (basis-independent):

- partitions `lambda`
- sector projectors `Q_lambda`
- sector-local Jordan spaces

Basis-dependent:

- multiplicity labels `a`
- multiplicity projectors `Q_{lambda,a}`
- multiplicity-local Jordan spaces/invariants

### Jordan naming

- cumulative space: `J_{<=j}`
- exact layer: `Delta J_j`
- API: `I_cumulative(order, ...)` and `I_exact(order, ...)`

## Mathematical Contracts (Implemented)

- Passive linear optics acts as `U = S^{\otimes n}` on the labeled external space.
- Global hierarchy uses one-body generators `E_st = sum_i |s><t|_i`.
- Local hierarchies are built from projected generators:
  - sector-local: `Q_lambda E_st Q_lambda`
  - multiplicity-local: `Q_{lambda,a} E_st Q_{lambda,a}`
- For reduced external states from permutation-symmetric totals, Schur blocks are the natural decomposition.

## Most Direct Workflow (State-First)

### 1) State Construction

```python
from photonic_jordan import Fock, FockMixed, from_fock_density, from_modes_and_gram

rho = from_modes_and_gram([0, 1, 0], gram=0.5, m_ext=2)
rho_fock = Fock(2, 1, 0)
rho_mix = FockMixed((0.7, 2, 1, 0), (0.3, 1, 2, 0))
# rho_adv = from_fock_density(rho_fock_matrix, m_ext=3, n_particles=3)
```

### 2) State Algebra (Classical Mix / Coherent Superposition)

```python
import numpy as np
from photonic_jordan import Fock, from_modes_and_gram

rho_a = from_modes_and_gram([0, 1, 0], gram=0.4, m_ext=2)
rho_b = from_modes_and_gram([0, 1, 0], gram=0.2, system=rho_a.system)
rho_classical = rho_a.mix(rho_b, weight=0.4)

rho_p = Fock(1, 0)
rho_q = Fock(0, 1, system=rho_p.system)
rho_coherent = rho_p.superpose(
    rho_q,
    alpha=np.sqrt(0.8),
    beta=np.sqrt(0.2) * np.exp(1j * np.pi / 3),
)
```

### 3) State Evolution

```python
S = rho_a.system.unitary.haar(seed=123)
rho_out = rho_a.evolve(S)
```

### 4) Invariant Computation

```python
print(rho_out.invariant.I_exact(2))
print(rho_out.invariant.I_exact(2, sector=(2, 1)))
print(rho_out.invariant.I_exact(2, multiplicity=((2, 1), 0)))
print(rho_out.analyze(max_order=3))
```

### 5) Observable Measurement

```python
obs = rho_out.system.observable.sigma_z(modes=(0, 1))
print(rho_out.measure.expectation(obs))
print(rho_out.measure.variance(obs))
print(rho_out.measure.distribution(obs))
```

Auto-cache behavior:

- `PhotonicSystem(..., auto_cache=True, cache_dir=...)` controls per-instance automatic Fock + Schur cache persistence.
- default is enabled, stored under `~/.cache/linear_optics_toolkit/`.
- global default can be changed via `PhotonicSystem.configure_global_cache(...)`.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Test

```bash
.venv/bin/python -m pytest -q
```

## Package Layout

- `photonic_jordan/math/`: linear algebra helpers.
- `photonic_jordan/spaces/`: labeled tensor space, bosonic Fock backend, and symmetric-group projectors.
- `photonic_jordan/hierarchy/`: Jordan filtration and invariant diagnostics.
- `photonic_jordan/measurement/`: one-body observable construction and measurement views.
- `photonic_jordan/state/`: state factory, state object, user-facing builders.
- `photonic_jordan/system/`: top-level `PhotonicSystem` orchestrator.
- `photonic_jordan/schur/`: Schur-Weyl decomposition and multiplicity projectors.

## Documentation

- Docs source: `docs/`
- MkDocs config: `mkdocs.yml`
- Local preview:

```bash
.venv/bin/mkdocs serve
```

- Local build:

```bash
.venv/bin/mkdocs build
```

GitHub Pages will deploy docs automatically from the workflow in `.github/workflows/docs-pages.yml`.

## Performance Guidance

- The dense labeled-space dimension is `d = m_ext ** n_particles`.
- Memory/time grow quickly with `d` (and generator-heavy workflows scale with `m_ext^2 * d^2` storage).
- For fully indistinguishable inputs (`gram=1`), state construction now uses a bosonic Fock backend of dimension `C(m_ext+n_particles-1, n_particles)` and avoids eager tensor-space density construction.
- Fock-backed states now evolve with a native lifted unitary in symmetric space (`rho.evolve(S)` stays in Fock cache unless tensor is explicitly requested).
- In this Fock path, global `rho.invariant.*` values correspond to the symmetric-sector local hierarchy (the `lambda=(n,)` block in tensor Schur language).
- See `docs/performance.md` for measured examples up to mode/particle counts near `6`.

## Online Demo

- `notebooks/demo_colab.ipynb`

## Citation

- Citation metadata: `CITATION.cff`
- Repository: https://github.com/yangbc30/hierarchical-invariants-paper-code
