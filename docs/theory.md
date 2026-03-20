# Theory Notes

## Hierarchy Construction

In this implementation, operator-space hierarchy is constructed numerically:

- `J_{<=0} = span{seed}` (default seed is identity or local projector),
- `J_{<=j}` is generated from `J_{<=j-1}` via right multiplication by one-body generators,
- orthogonalization is done in Hilbert-Schmidt geometry.

## Scope Types

Exactly one scope can be active for local invariants:

- global scope (default),
- sector scope: `sector=lambda`,
- multiplicity scope: `multiplicity=(lambda, a)`.

Multiplicity basis is not unique mathematically. This package fixes labels by a deterministic seeded commutant diagonalization convention.

## Physics Interpretation

- `I_j`: weight in the `j`-th exact layer (`Delta J_j`),
- `I_{<=j}`: total accessible weight up to order `j`,
- invariance under passive linear optics is checked through commutator diagnostics and equality of invariant values before/after evolution.

## Limitations

- Schur projector construction scales through permutation sums and is therefore expensive at larger `n`,
- prototype is optimized for numerical verification and reproducibility, not large-scale production workloads.
