# torch-omni-tool

A small PyTorch library of reusable network blocks plus a minimal training scaffold
(config, dataloaders, train/eval loops, logging helpers).

Every block is a plain `nn.Module` with type hints and no framework dependencies, so you can
import it from here or copy it into any project.

## Install

```bash
pip install -e .
```

Optional extras:

| Extra | Adds |
| --- | --- |
| `.[dev]` | `pytest`, `pylint`, `mypy`, `black`, `types-tqdm` |
| `.[docs]` | MkDocs, Material theme and mkdocstrings |
| `.[wandb]` | Weights & Biases logging (only imported when `Config.wb` is enabled) |

Requires Python >= 3.12.

## Quick Start

Use a block:

```python
import torch
from torch_omni_tool.blocks import GRN

x = torch.randn(2, 64, 56, 56)
y = GRN(64)(x)          # [2, 64, 56, 56]
```

Run the training scaffold:

```bash
python -m torch_omni_tool.training
```

Print the default configuration:

```bash
python -m torch_omni_tool.config
```

## What is inside

| Module | Contents |
| --- | --- |
| `torch_omni_tool.blocks` | ConvNeXt V2, GRN, RepMoDE 1D/2D/3D, RRDB 2D/3D, SEBlock, ViT embeddings |
| `torch_omni_tool.activations` | `Snake` |
| `torch_omni_tool.normalizations` | `LayerNorm` variants |
| `torch_omni_tool.datasets` | Dataset / DataLoader factory and 2D/3D augmentations |
| `torch_omni_tool.config` | `Config` dataclass and `config` singleton |
| `torch_omni_tool.model` | `SampleNet`, a placeholder autoencoder-style model |
| `torch_omni_tool.loops` | `run_model`, `train_loop`, `eval_loop` |
| `torch_omni_tool.utils` | Timing, metric formatting and epoch bookkeeping helpers |
| `torch_omni_tool.training` | Entry point that wires the pieces together |

## Next Steps

- [Getting Started](getting-started.md) - install and verify your setup
- [Usage](usage.md) - configuration, training and data loading
- [Architecture](architecture.md) - package layout and training flow
- [API Reference](api/index.md) - full module documentation
- [Changelog](changelog.md) - release history
