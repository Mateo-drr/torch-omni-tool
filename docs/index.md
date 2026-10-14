# torch-omni-tool

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: Non-Commercial](https://img.shields.io/badge/license-Non--Commercial-orange?style=flat-square)](https://github.com/Mateo-drr/torch-omni-tool/blob/main/LICENSE)

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

<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg .middle } __Blocks__

    ---

    ConvNeXt V2, GRN, RepMoDE 1D/2D/3D, RRDB 2D/3D, SEBlock and ViT positional embeddings.

    [:octicons-arrow-right-24: Reference](api/torch_omni_tool.blocks.md)

-   :material-tune-variant:{ .lg .middle } __Normalizations__

    ---

    `LayerNorm` for `[b, c, h, w]` and `[b, h, w, c]` inputs.

    [:octicons-arrow-right-24: Reference](api/torch_omni_tool.normalizations.md)

-   :material-sine-wave:{ .lg .middle } __Activations__

    ---

    The periodic `Snake` activation for channel-wise use.

    [:octicons-arrow-right-24: Reference](api/torch_omni_tool.activations.md)

-   :material-database-outline:{ .lg .middle } __Datasets__

    ---

    Dataset / DataLoader factory and 2D/3D augmentations.

    [:octicons-arrow-right-24: Reference](api/torch_omni_tool.datasets.md)

-   :material-tools:{ .lg .middle } __Utilities__

    ---

    Timing and metric formatting helpers.

    [:octicons-arrow-right-24: Reference](api/torch_omni_tool.utils.md)

</div>

## Next Steps

- [Getting Started](getting-started.md) - install and verify your setup
- [Usage](usage.md) - imports, blocks, datasets and layers
- [Architecture](architecture.md) - package layout and conventions
- [API Reference](api/index.md) - full module documentation
- [Changelog](changelog.md) - release history
