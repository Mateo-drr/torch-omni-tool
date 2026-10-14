# Getting Started

## Installation

Install in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

For building documentation locally:

```bash
pip install -e ".[docs]"
```

For optional Weights & Biases logging (only needed if `Config.wb` is enabled):

```bash
pip install -e ".[wandb]"
```

Requires Python >= 3.12.

## Activate Environment

If using the repo's virtual environment:

```bash
. .venv/bin/activate
```

## Quick Verification

Check configuration defaults:

```bash
python -m mltools.config
```

Run the training scaffold (uses intra-package imports correctly as a module):

```bash
python -m mltools.training
```

> Note: Always run as `python -m mltools.*` from the repo root. Running scripts directly (e.g. `python mltools/training.py`) may fail due to intra-package imports. 

## Development Tools

```bash
pytest             # Run tests (currently none)
mypy mltools       # Type check (strict mode)
pylint mltools     # Linting
```
