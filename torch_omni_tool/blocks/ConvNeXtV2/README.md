# ConvNeXtV2

The ConvNeXt block, in its V2 (default) or original V1 form, and Global Response
Normalization (GRN), the layer that ConvNeXt V2 adds to it.

| Class | File | Input |
| --- | --- | --- |
| `ConvNeXtBlock` | [`ConvNeXtV2.py`](ConvNeXtV2.py) | `[B, C, H, W]` |
| `GRN1d`, `GRN2d`, `GRN3d` | [`GRN.py`](GRN.py) | `[B, C, L]`, `[B, C, H, W]`, `[B, C, D, H, W]` (or channel last) |

## Source

- **ConvNeXt (`ConvNeXtBlock`, `version=1`):** Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer,
  Trevor Darrell, Saining Xie, *A ConvNet for the 2020s*, CVPR 2022 —
  [arXiv:2201.03545](https://arxiv.org/abs/2201.03545). Reference code: the `Block` class in
  [`facebookresearch/ConvNeXt` `models/convnext.py`](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py).
- **ConvNeXt V2 (`GRN*`, `ConvNeXtBlock` with `version=2`):** Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang
  Liu, In So Kweon, Saining Xie, *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
  Autoencoders*, CVPR 2023 — [arXiv:2301.00808](https://arxiv.org/abs/2301.00808). Reference
  code: the `GRN` class in
  [`facebookresearch/ConvNeXt-V2` `models/utils.py`](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py)
  and the `Block` class in
  [`models/convnextv2.py`](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py).
  Upstream GRN only takes 2D channel-last `[N, H, W, C]` tensors and stores `gamma` / `beta` as
  `[1, 1, 1, C]`; here it comes in 1d/2d/3d, takes either layout through `data_format` and
  stores the parameters as `[C]` (upstream checkpoints still load, see below).

## `ConvNeXtBlock`

```python
ConvNeXtBlock(dim: int, layer_scale_init_value: float = 1e-6, version: Literal[1, 2] = 2)
```

Input and output are `[B, dim, H, W]`:

1. Depthwise 7x7 convolution (`groups=dim`), then permute to `[B, H, W, C]`.
2. `LayerNorm2d` (`chan_last`) from [`normalizations`](../../normalizations/README.md).
3. Inverted bottleneck: `Linear(dim, 4 * dim)` → GELU → (V2 only: `GRN2d(4 * dim)`) →
   `Linear(4 * dim, dim)`.
4. V1 only: LayerScale, a learnable per-channel `gamma` initialized to
   `layer_scale_init_value` (a value `<= 0` disables it). V2 drops LayerScale, so
   `layer_scale_init_value` is ignored.
5. Permute back and add the residual.

```python
import torch
from torch_omni_tool.blocks.ConvNeXtV2 import ConvNeXtBlock

x = torch.randn(1, 64, 56, 56)
y_v2 = ConvNeXtBlock(dim=64)(x)  # ConvNeXt V2 block, [1, 64, 56, 56]
y_v1 = ConvNeXtBlock(dim=64, version=1)(x)  # original ConvNeXt block
```

## `GRN1d` / `GRN2d` / `GRN3d`

```python
GRN2d(dim: int, data_format: Literal["chan_first", "chan_last"] = "chan_first")
```

Global Response Normalization, `dim == C`. The math lives in one base class that works for any
number of spatial dims; the `1d` / `2d` / `3d` classes only fix the expected input rank, like
`nn.BatchNorm1d/2d/3d`:

| Class | `chan_first` | `chan_last` |
| --- | --- | --- |
| `GRN1d` | `[B, C, L]` | `[B, L, C]` |
| `GRN2d` | `[B, C, H, W]` | `[B, H, W, C]` |
| `GRN3d` | `[B, C, D, H, W]` | `[B, D, H, W, C]` |

1. `gx = ||x||_2` over all the spatial dims.
2. `nx = gx / (mean_C(gx) + 1e-6)`, normalizing the channel responses against each other.
3. `y = gamma * (x * nx) + beta + x`.

`gamma` and `beta` have shape `[C]` and are zero-initialized, so a new GRN is the identity.
Checkpoints that store them pre-shaped (`[1, C, 1, 1]` from older versions of this layer,
`[1, 1, 1, C]` from the official ConvNeXt V2 code) are reshaped on load. A wrong input rank or
channel count raises `ValueError`. The tested behaviour is listed in
[`tests/`](../../../tests/README.md).

```python
import torch
from torch_omni_tool.blocks.ConvNeXtV2 import GRN1d, GRN2d, GRN3d

y1 = GRN1d(64)(torch.randn(2, 64, 100))  # [2, 64, 100]
y2 = GRN2d(64)(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
y3 = GRN3d(64, data_format="chan_last")(torch.randn(2, 8, 8, 8, 64))  # [2, 8, 8, 8, 64]
```

## Caveats

- Stochastic depth (`DropPath`) is not implemented.
