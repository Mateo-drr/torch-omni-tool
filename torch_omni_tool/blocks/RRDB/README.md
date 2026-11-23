# RRDB

Residual-in-Residual Dense Blocks from ESRGAN, in 2D and 3D. Useful as the trunk of
super-resolution or restoration networks.

| Class | File | Input |
| --- | --- | --- |
| `ResidualDenseBlock_5C_2d`, `RRDB2d` | [`RRDB_2d.py`](RRDB_2d.py) | `[B, C, H, W]` |
| `ResidualDenseBlock_5C_3d`, `RRDB3d` | [`RRDB_3d.py`](RRDB_3d.py) | `[B, C, D, H, W]` |

## Source

- **Paper:** Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy,
  Yu Qiao, Xiaoou Tang, *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*,
  ECCV Workshops 2018 — [arXiv:1809.00219](https://arxiv.org/abs/1809.00219)
- **Reference code:** `ResidualDenseBlock_5C` and `RRDB` in
  [`xinntao/ESRGAN` `RRDBNet_arch.py`](https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py).
  The 2D blocks follow it directly (renamed with a `2d` suffix); the 3D blocks are the same
  design with `Conv3d`.

## `ResidualDenseBlock_5C_2d` / `ResidualDenseBlock_5C_3d`

```python
ResidualDenseBlock_5C_2d(num_features: int, growth_channels: int = 32, bias: bool = True, device=None, dtype=None)
ResidualDenseBlock_5C_3d(num_features: int, growth_channels: int = 32, bias: bool = True, device=None, dtype=None)
```

Five convolutions (3x3 in 2D, 3x3x3 in 3D) with dense connections and `LeakyReLU(0.2)`.
Convolutions 1-4 output `growth_channels` channels and see the concatenation of the input with every
previous output; convolution 5 maps back to `num_features` channels. The result is residual-scaled:

```
out = 0.2 * conv5(cat(x, x1, x2, x3, x4)) + x
```

## `RRDB2d` / `RRDB3d`

```python
RRDB2d(num_features: int, growth_channels: int = 32, device=None, dtype=None)
RRDB3d(num_features: int, growth_channels: int = 32, device=None, dtype=None)
```

Three dense blocks in sequence plus a residual around the stack, again scaled by `0.2`.

```python
import torch
from torch_omni_tool.blocks.RRDB import RRDB2d, RRDB3d

y2d = RRDB2d(num_features=64, growth_channels=32)(torch.randn(1, 64, 32, 32))  # [1, 64, 32, 32]
y3d = RRDB3d(num_features=32, growth_channels=16)(torch.randn(1, 32, 8, 8, 8))  # [1, 32, 8, 8, 8]
```

## Caveats

- All convolutions use `stride=1, padding=1`, so shapes are preserved and blocks stack freely.
- Only the building blocks are provided: no upsampling, pixel shuffle or discriminator.
