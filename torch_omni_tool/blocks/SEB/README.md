# SEB

Squeeze-and-Excitation channel gating.

| Class | File | Input |
| --- | --- | --- |
| `SEBlock` | [`SEB.py`](SEB.py) | `[B, C, H, W]` |

## Source

- **Paper:** Jie Hu, Li Shen, Gang Sun, *Squeeze-and-Excitation Networks*, CVPR 2018 —
  [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
- **Reference code:** the official release is the Caffe implementation at
  [`hujie-frank/SENet`](https://github.com/hujie-frank/SENet). This block follows the usual
  PyTorch formulation (bias-free linear layers), as in
  [`moskomule/senet.pytorch`](https://github.com/moskomule/senet.pytorch).

## `SEBlock`

```python
SEBlock(num_features: int, squeeze_channels: int = 16, device=None, dtype=None)
```

1. **Squeeze:** `AdaptiveAvgPool2d(1)` reduces the input to `[B, C]`.
2. **Excite:** `Linear(C, squeeze_channels)` → ReLU → `Linear(squeeze_channels, C)` → Sigmoid gives a
   per-channel gate in `[0, 1]`.
3. **Scale:** the input is multiplied by the gate.

```python
import torch
from torch_omni_tool.blocks.SEB import SEBlock

y = SEBlock(num_features=64, squeeze_channels=16)(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
```

## Caveats

- `squeeze_channels` is the absolute bottleneck width (same as torchvision's
  `SqueezeExcitation`), while the paper uses a reduction ratio `r`
  (bottleneck `C // r`, `r = 16` by default). Pass
  `squeeze_channels=num_features // 16` to match it.
