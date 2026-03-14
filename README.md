# Photonic Jordan Framework (Prototype)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/yangbc30/lonunify)

This repository provides a numerical prototype for checking Schur/multiplicity-scoped Jordan invariants in multiphoton systems.

## Quick Start in GitHub Codespaces

1. Click **Open in GitHub Codespaces**.
2. Wait for container setup (dependencies are installed automatically via `.devcontainer/devcontainer.json`).
3. Open `notebooks/demo_codespaces.ipynb`.
4. Run all cells.

## Local Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

## Demo Notebook

- `notebooks/demo_codespaces.ipynb`
- Shows:
  - sector weights,
  - sector/multiplicity scoped `I_exact`,
  - invariance check under passive unitary evolution.
