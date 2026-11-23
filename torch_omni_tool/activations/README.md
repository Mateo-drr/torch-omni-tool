# Activations

Activation layers.

| Class | File | Input |
| --- | --- | --- |
| `Snake` | [`snake.py`](snake.py) | `[B, C, H, W]` |

## Source

- **Paper:** Liu Ziyin, Tilman Hartwig, Masahito Ueda, *Neural Networks Fail to Learn Periodic
  Functions and How to Fix It*, NeurIPS 2020 — [arXiv:2006.08195](https://arxiv.org/abs/2006.08195)
- **Reference code:** written from the paper's formula. NVIDIA BigVGAN's
  [`activations.py`](https://github.com/NVIDIA/BigVGAN/blob/main/activations.py) is a widely
  used PyTorch implementation of the same activation (and of the `SnakeBeta` variant).

## `Snake`

```python
Snake(num_features: int, eps: float = 1e-6, device=None, dtype=None)
```

`snake_a(x) = x + (1 / (a + eps)) * sin(a * x) ** 2`, applied elementwise. `a` is a learnable
per-channel frequency of shape `[1, C, 1, 1]` initialized to `0.5`; `eps` keeps the division
finite when `a` approaches zero. The function is periodic and monotonic, which helps models
regress and extrapolate periodic signals.

```python
import torch
from torch_omni_tool.activations import Snake

y = Snake(64)(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
```

## Caveats

- The `[1, C, 1, 1]` parameter only broadcasts over 4D inputs; `[B, C, L]` inputs need a
  reshaped parameter.
- Only the plain `Snake` is implemented, not `SnakeBeta` (separate learnable magnitude).
