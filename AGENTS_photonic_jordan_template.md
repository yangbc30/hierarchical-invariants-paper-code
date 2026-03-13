# AGENTS.md template for Photonic Jordan repo

## Project goal

Implement and maintain a photonic Jordan framework for permutation-symmetric / partially distinguishable multiphoton states, passive linear optics, and global/sector/copy-local Jordan invariants.

## Primary public API

The main user-facing API must remain centered on:

- `PhotonicSystem`
- `PhotonicState`
- `PhotonicUnitary`
- `InvariantReport`

Avoid exposing internal engines as the normal user workflow.

## Scope conventions

There are three distinct scopes:

1. global
2. sector-local: `sector=λ`
3. copy-local: `copy=(λ, a)`

Never merge these concepts.

## Canonical vs basis-dependent objects

Canonical / basis-independent:

- partitions `λ`
- sector projectors `Q_λ`
- sector-local Jordan spaces

Basis-dependent:

- multiplicity basis labels `a`
- copy projectors `Q_{λ,a}`
- copy-local Jordan spaces / invariants

Any code or docs touching copy-local objects must state the basis dependence clearly.

## Jordan naming rules

Keep cumulative and exact-layer objects separate.

Use names like:

- `J_{<=j}` / cumulative
- `ΔJ_j` / exact layer
- `I_cumulative(j, ...)`
- `I_exact(j, ...)`

Do not use ambiguous names like `layer_weight` for cumulative quantities.

## Engineering rules

- Prefer small, reviewable changes.
- Add tests with every mathematical feature.
- Do not change public API unless explicitly asked.
- Keep notebook ergonomics short and direct.
- Use lazy caching for Schur decomposition, projectors, and Jordan hierarchies.

## Test commands

Fill these in for the actual repo:

- `pytest -q`
- `pytest tests/test_api_smoke.py -q`
- formatting command
- lint command

## Done criteria

A task is done only if:

- relevant tests pass,
- new code paths have unit tests,
- notebook-style smoke usage still works,
- public API semantics remain consistent,
- any copy-local feature documents basis dependence.
