# Normalizations

Normalization layers.

| Class | File | Input |
| --- | --- | --- |
| `LayerNorm2d` | [`layer_norm_2d.py`](layer_norm_2d.py) | `[B, C, H, W]` or `[B, H, W, C]` |

## Source

- **Paper:** Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, *Layer Normalization*, 2016 —
  [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- **Reference code:** the `LayerNorm` class of the official ConvNeXt implementation,
  [`facebookresearch/ConvNeXt` `models/convnext.py`](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py)
  (Liu et al., *A ConvNet for the 2020s*, CVPR 2022 —
  [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)). The only change is the naming of the
  formats: `chan_first` / `chan_last` here, `channels_first` / `channels_last` upstream.

## `LayerNorm2d`

```python
LayerNorm2d(
    normalized_shape: int,
    eps: float = 1e-6,
    data_format: Literal["chan_first", "chan_last"] = "chan_first",
)
```

Normalizes over the channel dimension with a learnable per-channel `weight` (ones) and `bias`
(zeros). `normalized_shape` is the number of channels.

- `data_format="chan_last"`: input `[B, H, W, C]`, delegated to `F.layer_norm` over the last dim.
- `data_format="chan_first"`: input `[B, C, H, W]`, mean and variance computed over dim 1 and
  the scale and bias broadcast as `weight[:, None, None]`.

```python
import torch
from torch_omni_tool.normalizations import LayerNorm2d

norm = LayerNorm2d(64, data_format="chan_first")
y = norm(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
```

## Caveats

- Any other `data_format` raises `NotImplementedError` at construction.
- `chan_first` assumes exactly two trailing spatial dims; there are no 1D/3D variants.
