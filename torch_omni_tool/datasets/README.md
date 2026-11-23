# Datasets

A placeholder dataset with its dataloader factory, and data augmentations, one folder per
augmentation. Names are exported by the folder that contains their file:

```python
from torch_omni_tool.datasets import make_dl
from torch_omni_tool.datasets.CutMix import cutmix
from torch_omni_tool.datasets.RandomLocalRotation import random_local_rotation_2d
```

## Available augmentations

| Folder | Functions | Data | Source |
| --- | --- | --- | --- |
| [RandomLocalRotation/](RandomLocalRotation/README.md) | `random_local_rotation_2d`, `random_local_rotation_3d` | 2D, 3D | Alomar et al. ([paper](https://doi.org/10.3390/jimaging9020046)) |
| [CutMix/](CutMix/README.md) | `cutmix` | 3D | CutMix ([paper](https://arxiv.org/abs/1905.04899), [code](https://github.com/clovaai/CutMix-PyTorch)) |

Each folder README lists the full citation and how the implementation differs from the
reference code.

## Dataset scaffold

[`cstm_ds.py`](cstm_ds.py) holds `CustomDataset(config)`, which returns samples as
`{"data": tensor}` as expected by [`loops.run_model`](../loops.py) (`samples["data"]` is both
input and target). It holds a single random `[8, 16]` tensor: replace the `self.data`
construction with real loading logic and keep returning `{"data": ...}`.

`make_dl(config, split)` wraps it in a `DataLoader` with `batch_size=config.batch`,
`num_workers=config.num_workers` and `pin_memory=True`. Only `"train"` shuffles and sets
`prefetch_factor` (skipped when `num_workers == 0`).

```python
from torch_omni_tool.config import config
from torch_omni_tool.datasets import make_dl

samples = next(iter(make_dl(config, "train")))  # {"data": tensor}
```
