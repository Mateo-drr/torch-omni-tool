# torch-omni-tool

A small PyTorch library of reusable network blocks and layers.

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

Requires Python >= 3.12.

## Quick Start

```python
import torch
from torch_omni_tool.blocks import GRN

x = torch.randn(2, 64, 56, 56)
y = GRN(64)(x)          # [2, 64, 56, 56]
```

## What is inside

| Module | Contents |
| --- | --- |
| `torch_omni_tool.blocks` | ConvNeXt V2, GRN, RepMoDE 1D/2D/3D, RRDB 2D/3D, SEBlock, ViT embeddings |
| `torch_omni_tool.activations` | `Snake` |
| `torch_omni_tool.normalizations` | `LayerNorm` variants |
| `torch_omni_tool.datasets` | Dataset / DataLoader factory and 2D/3D augmentations |
| `torch_omni_tool.utils` | Timing and metric formatting helpers |

## Next Steps

- [Getting Started](getting-started.md) - install and verify your setup
- [Usage](usage.md) - imports, blocks, datasets and layers
- [Architecture](architecture.md) - package layout and conventions
- [API Reference](api/index.md) - full module documentation
- [Changelog](changelog.md) - release history
