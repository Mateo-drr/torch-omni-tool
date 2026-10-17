# Usage

## Imports

Every subpackage is reachable straight off the root package:

```python
import torch_omni_tool as omni

omni.activations.Snake(64)      # [b, c, h, w]
omni.blocks.GRN(64)             # [b, c, h, w]
omni.normalizations.LayerNorm(64)
omni.datasets.make_dl
omni.utils.format_time
```

The same public names are available through direct imports:

```python
from torch_omni_tool.blocks import SEBlock, GRN
from torch_omni_tool.normalizations import LayerNorm
from torch_omni_tool.activations import Snake
from torch_omni_tool.datasets import make_dl
```

Leaf-module paths also work if needed. Be consistent with the existing naming (some files use
`2D` vs `3d` casing).

> Note: The normalization module filename is `layern_norm_2d.py` (typo). The import path is
> `torch_omni_tool.normalizations.layern_norm_2d`.

## Blocks

Every block is a standalone `nn.Module`, so it composes like any other layer:

```python
import torch
from torch_omni_tool.blocks import ConvNeXtBlock, SEBlock

x = torch.randn(2, 64, 56, 56)
x = SEBlock(64)(x)          # [2, 64, 56, 56]
x = ConvNeXtBlock(64)(x)    # [2, 64, 56, 56]
```

Many blocks come in dimension-specific variants (RepMoDE 1D/2D/3D, RRDB 2D/3D). Pick the
variant that matches your spatial dimensions; see [Architecture](architecture.md) for the full
list.

## Datasets and Augmentations

`torch_omni_tool.datasets` exposes the dataset / dataloader factory and the augmentations:

```python
from torch_omni_tool.datasets import (
    CustomDataset,
    make_dl,
    cutmix,
    random_local_rotation_2d,
    random_local_rotation_3d,
)
```

## Normalizations and Activations

```python
from torch_omni_tool.normalizations import LayerNorm
from torch_omni_tool.activations import Snake
```

`LayerNorm` accepts `[b, c, h, w]` and `[b, h, w, c]` inputs. `Snake` is a periodic activation
applied channel-wise.
