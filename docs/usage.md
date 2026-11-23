# Usage

## Imports

Import paths mirror the folder tree. Every name is exported by the folder that contains its
file, and parent packages only expose their subfolders:

```python
from torch_omni_tool.activations import Snake
from torch_omni_tool.normalizations import LayerNorm2d
from torch_omni_tool.blocks.ConvNeXtV2 import ConvNeXtBlock, GRN2d
from torch_omni_tool.blocks.RepMoDE import MoDEConv3d
from torch_omni_tool.blocks.RRDB import RRDB2d
from torch_omni_tool.datasets import make_dl
from torch_omni_tool.datasets.CutMix import cutmix
```

The same paths work as attributes off the root package:

```python
import torch_omni_tool as omni

omni.activations.Snake(64)
omni.blocks.SEB.SEBlock(64)
omni.blocks.RRDB.RRDB3d(nf=32)
omni.datasets.RandomLocalRotation.random_local_rotation_2d
```

## Blocks

Every block is a standalone `nn.Module`, so it composes like any other layer:

```python
import torch
from torch_omni_tool.blocks.ConvNeXtV2 import ConvNeXtBlock
from torch_omni_tool.blocks.SEB import SEBlock

x = torch.randn(2, 64, 56, 56)
x = SEBlock(64)(x)  # [2, 64, 56, 56]
x = ConvNeXtBlock(64)(x)  # ConvNeXt V2 block, [2, 64, 56, 56]
x = ConvNeXtBlock(64, version=1)(x)  # original ConvNeXt block
```

Classes with dimension variants end in `1d`, `2d` or `3d`, like `nn.Conv2d`: `GRN1d` /
`GRN2d` / `GRN3d`, `MoDEConv1d` / `MoDEConv2d` / `MoDEConv3d`, `RRDB2d` / `RRDB3d`. Pick the one
that matches your spatial dimensions; see [Architecture](architecture.md) for the full list.

## Datasets and Augmentations

`torch_omni_tool.datasets` holds the dataset scaffold, and each augmentation has its own folder:

```python
import torch
from torch_omni_tool.datasets import CustomDataset, make_dl
from torch_omni_tool.datasets.CutMix import cutmix
from torch_omni_tool.datasets.RandomLocalRotation import random_local_rotation_2d

images, labels = torch.rand(4, 3, 64, 64), torch.randint(0, 5, (4, 64, 64))
images, labels = random_local_rotation_2d(images, labels, radius=16, p=0.5)
```

## Normalizations and Activations

```python
import torch
from torch_omni_tool.activations import Snake
from torch_omni_tool.normalizations import LayerNorm2d

x = torch.randn(2, 64, 56, 56)
x = LayerNorm2d(64, data_format="chan_first")(x)
x = Snake(64)(x)
```

`LayerNorm2d` accepts `[b, c, h, w]` (`chan_first`) and `[b, h, w, c]` (`chan_last`) inputs.
`Snake` is a periodic activation with a learnable frequency per channel.
