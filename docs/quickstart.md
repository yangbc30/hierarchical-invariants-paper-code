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
```

## Online Notebooks

- `notebooks/demo_colab.ipynb`
