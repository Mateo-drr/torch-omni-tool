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
SEBlock(channels: int, reduce_dim: int = 16)
```

1. **Squeeze:** `AdaptiveAvgPool2d(1)` reduces the input to `[B, C]`.
2. **Excite:** `Linear(C, reduce_dim)` → ReLU → `Linear(reduce_dim, C)` → Sigmoid gives a
   per-channel gate in `[0, 1]`.
3. **Scale:** the input is multiplied by the gate.

```python
import torch
from torch_omni_tool.blocks.SEB import SEBlock

y = SEBlock(channels=64, reduce_dim=16)(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
```

## Caveats

- `reduce_dim` is the absolute bottleneck width, while the paper uses a reduction ratio `r`
  (bottleneck `C // r`, `r = 16` by default). Pass `reduce_dim=channels // 16` to match it.
