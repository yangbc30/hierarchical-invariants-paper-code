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

## 1) State Construction

```python
from photonic_jordan import Fock, FockMixed, from_fock_density, from_modes_and_gram

rho = from_modes_and_gram([0, 1, 0], gram=0.4, m_ext=2)
rho_fock = Fock(2, 1, 0)
rho_mix = FockMixed((0.6, 2, 1, 0), (0.4, 1, 2, 0))
# rho_adv = from_fock_density(rho_fock_matrix, m_ext=3, n_particles=3)
```

## 2) State Algebra

```python
import numpy as np
from photonic_jordan import Fock, from_modes_and_gram

rho_a = from_modes_and_gram([0, 1, 0], gram=0.4, m_ext=2)
rho_b = from_modes_and_gram([0, 1, 0], gram=0.2, system=rho_a.system)

# classical mixture
rho_classical = rho_a.mix(rho_b, weight=0.3)

# coherent superposition (pure states only)
rho_p = Fock(1, 0)
rho_q = Fock(0, 1, system=rho_p.system)
rho_coherent = rho_p.superpose(rho_q, alpha=1.0, beta=1j)
```

## 3) State Evolution

```python
S = rho.system.unitary.haar(seed=7)
rho_out = rho.evolve(S)
```

## 4) Invariant Computation

```python
print(rho_out.invariant.I_exact(1))
print(rho_out.invariant.I_cumulative(2))
print(rho_out.invariant.I_exact(1, sector=(2, 1)))
print(rho_out.invariant.I_exact(1, multiplicity=((2, 1), 0)))
```

## 5) Observable Measurement

```python
obs = rho_out.system.observable.sigma_z(modes=(0, 1))
print(rho_out.measure.expectation(obs))
print(rho_out.measure.variance(obs))
print(rho_out.measure.distribution(obs))
```

## Optional: Representations and Cache

```python
rho_tensor = rho_out.density_matrix(rep="tensor")
rho_fock = rho_out.density_matrix(rep="fock")
rho_schur = rho_out.density_matrix(rep="schur")

rho_out.system.save_fock_cache("cache_m2_n3.npz", max_order=3, include_generators=True)
rho_out.system.load_fock_cache("cache_m2_n3.npz")
rho_out.system.save_schur_cache("schur_cache_m2_n3.npz", include_multiplicity=True)
rho_out.system.load_schur_cache("schur_cache_m2_n3.npz")
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
