# Quick Start

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Run Tests

```bash
.venv/bin/python -m pytest
```

## Minimal Example

```python
import numpy as np
from photonic_jordan import PhotonicSystem

sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(0))
rho = sys.state.from_modes_and_gram([0, 1, 0], gram=0.4)

# Global hierarchy
print(rho.invariant.I_exact(1))
print(rho.invariant.I_cumulative(2))

# Sector-local hierarchy
lam = (2, 1)
print(rho.invariant.I_exact(1, sector=lam))

# Multiplicity-local hierarchy
print(rho.invariant.I_exact(1, multiplicity=(lam, 0)))

# Same state in different representations
rho_tensor = rho.density_matrix(rep=\"tensor\")
rho_schur = rho.density_matrix(rep=\"schur\")
print(rho_tensor.shape, rho_schur.shape)

# Observable measurement
obs = sys.observable.sigma_z(modes=(0, 1))
print(rho.measure.expectation(obs))
print(rho.measure.variance(obs))
print(rho.measure.distribution(obs))
```

## Measurement Scope Behavior

- `conditional=False` (default): scoped statistics are unnormalized scope contributions.
- `conditional=True`: scoped density is renormalized inside the selected scope.

For distributions:

- global query sums to `1`
- scoped query with `conditional=False` sums to `Tr(Q rho)`
- scoped query with `conditional=True` sums to `1`

## Not Included Yet

- generic POVM classes
- detector noise / dark counts
- post-measurement state update maps

## Online Notebooks

- `notebooks/demo_colab.ipynb`
