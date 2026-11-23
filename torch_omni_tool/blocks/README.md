# Blocks

Reusable `nn.Module` blocks, one folder per block family. Classes are exported by their folder,
so import paths mirror the tree:

```python
from torch_omni_tool.blocks.RRDB import RRDB3d
from torch_omni_tool.blocks.RepMoDE import MoDEConv3d
from torch_omni_tool.blocks.SEB import SEBlock
```

## Available blocks

| Folder | Classes | Data | Source |
| --- | --- | --- | --- |
| [ConvNeXtV2/](ConvNeXtV2/README.md) | `ConvNeXtBlock`, `GRN1d`, `GRN2d`, `GRN3d` | 2D (GRN: 1D, 2D, 3D) | ConvNeXt ([paper](https://arxiv.org/abs/2201.03545), [code](https://github.com/facebookresearch/ConvNeXt)), ConvNeXt V2 ([paper](https://arxiv.org/abs/2301.00808), [code](https://github.com/facebookresearch/ConvNeXt-V2)) |
| [RepMoDE/](RepMoDE/README.md) | `MoDE`, `MoDEConv{1d,2d,3d}`, `MoDE{Encoder,Decoder}Block{1d,3d}`, `MoDESubNet2Conv{1d,3d}` | 1D, 2D, 3D | RepMode ([paper](https://arxiv.org/abs/2212.10066), [code](https://github.com/correr-zhou/RepMode)) |
| [RRDB/](RRDB/README.md) | `RRDB2d`, `RRDB3d`, `ResidualDenseBlock_5C_2d`, `ResidualDenseBlock_5C_3d` | 2D, 3D | ESRGAN ([paper](https://arxiv.org/abs/1809.00219), [code](https://github.com/xinntao/ESRGAN)) |
| [SEB/](SEB/README.md) | `SEBlock` | 2D | SENet ([paper](https://arxiv.org/abs/1709.01507), [code](https://github.com/hujie-frank/SENet)) |
| [ViT/](ViT/README.md) | `PositionEmbeddingSine` | 2D | DETR ([paper](https://arxiv.org/abs/2005.12872), [code](https://github.com/facebookresearch/detr)) |

Each folder README lists the full citation and how the implementation differs from the
reference code.

## Things to know

- **Pick the variant that matches your data.** Classes with dimension variants end in `1d`,
  `2d` or `3d` (like `nn.Conv2d`), files in `_1d.py`, `_2d.py` or `_3d.py`.
