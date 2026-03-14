# Reproducibility

## Environment

Use the pinned ranges in:

- `pyproject.toml`
- `requirements.txt`

Install with:

```bash
.venv/bin/pip install -e '.[dev]'
```

## Tests

Run:

```bash
.venv/bin/python -m pytest
```

Test suite includes:

- decomposition and projector consistency,
- sector/multiplicity-local invariance checks,
- hierarchy monotonicity and projection identities,
- Gram validation and physicality checks,
- multiplicity-label determinism for fixed model size.

## Citation Metadata

Citable metadata is provided in `CITATION.cff`.
