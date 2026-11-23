# CutMix

Swap a random region between two training samples, labels included.

| Function | File | Input |
| --- | --- | --- |
| `cutmix` | [`CutMix_3d.py`](CutMix_3d.py) | chunks `[b, 1, d, h, w]`, labels `[b, h, w]` |

## Source

- **Paper:** Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon
  Yoo, *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*,
  ICCV 2019 — [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
- **Reference code:** [`clovaai/CutMix-PyTorch`](https://github.com/clovaai/CutMix-PyTorch).
  This version is adapted to volumes with dense labels: it swaps the region in both directions
  and does not mix the targets by patch area.

## `cutmix`

```python
cutmix(chunks, lbls, mask_size=32)
```

Picks two adjacent samples of the batch and swaps a random `mask_size` cube between them, and the
matching `[h, w]` window between their labels. Both tensors are modified in place.

```python
import torch
from torch_omni_tool.datasets.CutMix import cutmix

chunks, labels = cutmix(torch.rand(4, 1, 64, 64, 64), torch.randint(0, 2, (4, 64, 64)))
```

## Caveats

- Needs a batch of at least 2.
- Only adjacent samples are mixed, and a single pair per call.
