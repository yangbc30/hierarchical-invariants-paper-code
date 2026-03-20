# Performance Notes

This project uses dense matrices on the labeled external Hilbert space of dimension
`d = m_ext ** n_particles`.

For fully indistinguishable inputs (`gram=1`), the code now has a bosonic Fock
backend with symmetric-space dimension
`d_sym = C(m_ext + n_particles - 1, n_particles)`.

## Complexity Expectations

For dense operators:

- one density matrix costs about `16 * d^2` bytes (`complex128`)
- one lifted generator `E_st` also costs about `16 * d^2` bytes
- full generator cache costs about `m_ext^2 * 16 * d^2` bytes

So scaling is fundamentally exponential in particle number through `d = m^n`.

In the symmetric Fock backend, the same dense formulas use `d_sym` instead of
`d`, which can reduce costs by orders of magnitude.

## Practical Consequences

- state construction and basic evolution can still be usable for moderate `d`
- hierarchy/invariant builds are much more expensive because they use all one-body generators and operator-space linear algebra
- cases near `(m_ext, n_particles) = (5, 5)` are already large in dense form
- for `gram=1`, state build and observable workflows run through Fock first
  and avoid eager full-tensor density construction

## Suggested Larger Examples (still practical)

```python
import numpy as np
from photonic_jordan import PhotonicSystem

# Example A: many particles, few modes
sys = PhotonicSystem(m_ext=2, n_particles=6, rng=np.random.default_rng(1))
rho = sys.state.from_modes_and_gram([0, 1, 0, 1, 0, 1], gram=1)

# Example B: balanced medium size
sys = PhotonicSystem(m_ext=4, n_particles=4, rng=np.random.default_rng(2))
rho = sys.state.from_modes_and_gram([0, 1, 2, 3], gram=1)

# Example C: larger mode count
sys = PhotonicSystem(m_ext=6, n_particles=4, rng=np.random.default_rng(3))
rho = sys.state.from_modes_and_gram([0, 1, 2, 3], gram=1)
```

## Measured Runtime Snapshot (local machine)

This snapshot is for:

```python
sys = PhotonicSystem(m_ext=m, n_particles=n)
rho = sys.state.from_modes_and_gram([...], gram=1)
```

| `(m,n)` | `d=m^n` | `rho` memory (MB) | generator cache (MB) | init + state (s) |
|---|---:|---:|---:|---:|
| `(2,6)` | 64 | 0.06 | 0.25 | 0.041 |
| `(3,5)` | 243 | 0.90 | 8.11 | 0.004 |
| `(4,4)` | 256 | 1.00 | 16.00 | 0.002 |
| `(5,4)` | 625 | 5.96 | 149.01 | 0.012 |
| `(6,4)` | 1296 | 25.63 | 922.64 | 0.033 |
| `(5,5)` | 3125 | 149.01 | 3725.29 | 0.216 |

Notes:

- `PhotonicSystem` initialization is now lazy for generators/Jordan global filtration.
- heavy cost appears when first calling hierarchy/invariant operations.
- for example, `rho.invariant.I_exact(1)` at `(m,n)=(3,5)` took about `22.8s` in this environment.

## Symmetric Fock Scale Reference

| `(m,n)` | `m^n` | `C(m+n-1,n)` | matrix-size reduction factor `(m^n / C)^2` |
|---|---:|---:|---:|
| `(5,5)` | 3125 | 126 | 615.12 |
| `(6,5)` | 7776 | 252 | 952.16 |
| `(5,6)` | 15625 | 210 | 5536.07 |
| `(6,6)` | 46656 | 462 | 10198.38 |

Notes:

- The Fock backend currently targets bosonic fully indistinguishable input paths.
- Global invariant queries on Fock-backed states correspond to the symmetric-sector local hierarchy (`lambda=(n,)` in tensor/Schur language).
- If you repeatedly use fixed `(m_ext, n_particles)`, persist caches with
  `save_fock_cache(...)`/`load_fock_cache(...)` and
  `save_schur_cache(...)`/`load_schur_cache(...)` to avoid rebuilding
  generators, hierarchy bases, and Schur decomposition data.
- Automatic cache persistence is enabled by default:
  `PhotonicSystem(..., auto_cache=True, cache_dir=...)`.

## Recommendation

- for routine iteration, stay around `d <= 10^3` unless you specifically profile your workload
- use observable/statistics workflows first, and run hierarchy builds selectively
- avoid dense hierarchy runs near `d ~ 3000+` unless you have large memory headroom
