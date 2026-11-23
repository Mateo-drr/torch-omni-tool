# ViT

Positional embeddings for vision transformers.

| Class | File | Input |
| --- | --- | --- |
| `PositionEmbeddingSine` | [`position_embedding_sine.py`](position_embedding_sine.py) | `[B, C, H, W]` |

## Source

- **Paper:** Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander
  Kirillov, Sergey Zagoruyko, *End-to-End Object Detection with Transformers* (DETR), ECCV 2020
  — [arXiv:2005.12872](https://arxiv.org/abs/2005.12872). The 2D encoding applies the 1D
  sinusoidal encoding of Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
  ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)) to each axis.
- **Reference code:** `PositionEmbeddingSine` in
  [`facebookresearch/detr` `models/position_encoding.py`](https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py).
  Upstream takes `num_pos_feats` as an argument and builds the coordinates from a padding mask;
  this version derives the feature count from `C` and uses the full grid (no mask).

## `PositionEmbeddingSine`

```python
PositionEmbeddingSine(temperature: float = 10000, normalize: bool = True, scale: float | None = None)
```

Returns an embedding with the same shape as the input, which is only used for its shape and
device, so the result can be added to the feature map directly:

1. 1-based coordinate grids for `y` and `x`.
2. With `normalize=True` each grid is divided by its last value and multiplied by `scale`
   (default `2 * pi`). Passing `scale` with `normalize=False` raises `ValueError`.
3. Each coordinate is divided by `temperature ** (2 * (i // 2) / num_pos_feats)` with
   `num_pos_feats = (C + 1) // 2` and encoded as interleaved sine/cosine.
4. The `y` and `x` encodings are concatenated on the channel axis (`y` first) and cut to `C`.

```python
import torch
from torch_omni_tool.blocks.ViT import PositionEmbeddingSine

pos = PositionEmbeddingSine()(torch.randn(2, 128, 56, 56))  # [2, 128, 56, 56]
```

## Caveats

- `(C + 1) // 2` must be even, otherwise `torch.stack` raises (e.g. `C = 6` or `C = 10`).
  Use a channel count that is a multiple of 4.
- With `normalize=False` the coordinates are raw pixel indices, so the encoding changes with
  the image size.
