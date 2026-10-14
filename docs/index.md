# torch-omni-tool

A small PyTorch library of reusable network blocks plus a minimal training scaffold
(config, dataloaders, train/eval loops, logging helpers).

Everything is plain `nn.Module` code with type hints and no framework dependencies, so
blocks can be copied into any project or imported from here.

## Installation

```bash
pip install -e ".[dev]"      # runtime + pytest / pylint / mypy / black / types-tqdm
pip install -e ".[docs]"     # docs dependencies
pip install -e ".[wandb]"    # adds the optional Weights & Biases logging dependency
```

Requires Python >= 3.12. `wandb` is optional and only imported when `Config.wb` is enabled,
so installing it is not needed to use the blocks.

## Quick Start

Run the training scaffold:

```bash
python -m mltools.training
```

Inspect configuration defaults:

```bash
python -m mltools.config
```

## Features

- Reusable PyTorch blocks (ConvNeXt V2, GRN, RepMoDE 1D/2D/3D, RRDB 2D/3D, SEBlock, ViT embeddings)
- Activations (Snake)
- Normalizations (LayerNorm variants)
- Minimal training scaffold with AMP and GradScaler support
- Typed with `py.typed` (PEP 561)

See the [Usage](usage.md) and [Architecture](architecture.md) pages for details.
