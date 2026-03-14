# Photonic Jordan Framework (Research Prototype)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangbc30/hierarchical-invariants-paper-code/blob/main/notebooks/demo_colab.ipynb)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/yangbc30/hierarchical-invariants-paper-code)

Numerical framework for checking hierarchical Jordan invariants of multiphoton states under passive linear optics, with global/sector/multiplicity scopes.

## Project Goal

- Keep a simple user API centered on `PhotonicSystem`, `PhotonicState`, and `InvariantReport`.
- Provide reproducible numerical checks for theory statements used in papers.
- Support Schur-aware decomposition and local Jordan hierarchies.

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

## Canonical Workflow

```python
import numpy as np
from photonic_jordan import PhotonicSystem

sys = PhotonicSystem(m_ext=3, n_particles=3, particle_type="boson")
rho = sys.state.from_modes_and_gram([0, 1, 2], gram="indistinguishable")

S = sys.unitary.haar(seed=123)
rho_out = rho.evolve(S)

print(rho_out.invariant.I_exact(2))
print(rho_out.invariant.I_exact(2, sector=(2, 1)))
print(rho_out.invariant.I_exact(2, multiplicity=((2, 1), 0)))
print(rho_out.analyze(max_order=3))

# Same physical state in different matrix representations
rho_tensor = rho_out.density_matrix(rep="tensor")
rho_schur = rho_out.density_matrix(rep="schur")
```

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
- `photonic_jordan/spaces/`: labeled tensor space and symmetric-group projectors.
- `photonic_jordan/hierarchy/`: Jordan filtration and invariant diagnostics.
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

## Online Demo

- `notebooks/demo_colab.ipynb`
- `notebooks/demo_codespaces.ipynb`

## Citation

- Citation metadata: `CITATION.cff`
- Repository: https://github.com/yangbc30/hierarchical-invariants-paper-code
