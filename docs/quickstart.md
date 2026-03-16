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

sys = PhotonicSystem(m_ext=2, n_particles=3, rng=np.random.default_rng(0), auto_cache=True)
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
rho_fock = rho.density_matrix(rep=\"fock\")  # bosonic systems
rho_schur = rho.density_matrix(rep=\"schur\")
print(rho_tensor.shape, rho_fock.shape, rho_schur.shape)

# Observable measurement
obs = sys.observable.sigma_z(modes=(0, 1))
print(rho.measure.expectation(obs))
print(rho.measure.variance(obs))
print(rho.measure.distribution(obs))

# Fully indistinguishable shortcut (fast Fock backend)
rho_sym = sys.state.from_modes_and_gram([0, 1, 0], gram=1)
print(rho_sym.has_rep(\"fock\"), rho_sym.has_rep(\"tensor\"))
print(rho_sym.density_matrix(rep=\"fock\").shape)

# Native Fock evolution for fock-backed states
S = sys.unitary.haar(seed=7)
rho_sym_out = rho_sym.evolve(S)
print(rho_sym_out.has_rep(\"fock\"), rho_sym_out.has_rep(\"tensor\"))

# Optional: save/load expensive Fock caches
sys.save_fock_cache(\"cache_m2_n3.npz\", max_order=3, include_generators=True)
sys.load_fock_cache(\"cache_m2_n3.npz\")
```

By default, automatic cache persistence is enabled and stored under
`~/.cache/linear_optics_toolkit/`. You can override per instance with `cache_dir=...`
or set global defaults via `PhotonicSystem.configure_global_cache(...)`.

For `gram=1` inputs, `from_modes_and_gram` stores the state in the symmetric Fock backend first.
Global invariant calls on such states use the symmetric-sector local hierarchy (`lambda=(n,)`).

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
