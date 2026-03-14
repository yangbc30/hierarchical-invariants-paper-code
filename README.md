# Photonic Jordan Framework (Prototype)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangbc30/hierarchical-invariants-paper-code/blob/main/notebooks/demo_colab.ipynb)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/yangbc30/hierarchical-invariants-paper-code)

This repository provides a numerical prototype for checking Schur/multiplicity-scoped Jordan invariants in multiphoton systems.

## Quick Start in Colab

1. Click **Open in Colab**.
2. Open `notebooks/demo_colab.ipynb` if not already open.
3. Run all cells.

Note: for external users to open this notebook via link, the repository must be public.

## Quick Start in GitHub Codespaces (optional)

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
- `notebooks/demo_colab.ipynb`
- Shows:
  - sector weights,
  - sector/multiplicity scoped `I_exact`,
  - invariance check under passive unitary evolution.
