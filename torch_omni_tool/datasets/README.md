# Datasets

Dataset / dataloader factory and 2D/3D augmentations. The dataset here is a placeholder that
returns random tensors; swap its body for real data and the rest of the scaffold keeps
working.

## `cstm_ds.py`

### `CustomDataset`

```python
CustomDataset(config: Config)
```

A `torch.utils.data.Dataset` whose samples are dicts with the tensor under the `"data"` key,
which is what [`..`](../loops.py) expects (`samples["data"]` is both the input
and the autoencoder target). It currently holds a single random `[8, 16]` tensor, so
`len(dataset) == 1` and `dataset[0]` returns `{"data": tensor of shape [8, 16]}`.

To use real data, replace the `self.data` construction with whatever loading logic you need
(file reads, memory-mapped arrays, a HuggingFace dataset, ...) and keep returning
`{"data": ...}`. `config` is stored on the instance, so batch/worker settings stay
reachable.

### `make_dl`

```python
make_dl(config: Config, split: Literal["train", "valid", "test"]) -> DataLoader[dict[str, Any]]
```

Wraps `CustomDataset` in a `DataLoader` with `batch_size=config.batch`,
`pin_memory=True` and `num_workers=config.num_workers`. Only the training split shuffles,
and only it sets `prefetch_factor` (skipped when `num_workers == 0`, since the option is
invalid there). Any other split raises `NotImplementedError`.

```python
from torch_omni_tool.config import config
from torch_omni_tool.datasets.cstm_ds import make_dl

loader = make_dl(config, "train")
samples = next(iter(loader))  # {"data": tensor}
```

## `augmentations_2d.py`

Augmentations for 2D data, for use with 2D convolutional models. Uses Python's `random`
and `torchvision.transforms.functional` (`torchvision` is a core dependency).

### `random_local_rotation_2d`

```python
random_local_rotation_2d(in_tensor, label=None, radius=16, p=0.5) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]
```

Rotates a random circular patch together with its label, so the two stay aligned. Pass
`label=None` (the default) to augment only the input tensor, in which case just the rotated
tensor is returned. Each sample of the batch is drawn independently: it has a chance `p` of
being augmented, and its own random square position and angle in `[20, 340]`. Unlike the 3D
version, the label is batched and the function is a no-op for samples that are not selected.

Args:

- `in_tensor`: image batch `[b, c, h, w]`.
- `label`: optional label batch `[b, h, w]`; integer labels are kept integer (rotated with
  nearest interpolation), float labels keep their dtype.
- `radius`: radius of the circular patch that gets rotated.
- `p`: per-sample probability of applying the augmentation.

Caveats:

- Raises `ValueError` if `in_tensor` is not 4D, `label` is provided but not 3D, or
  `2 * radius` does not fit in the spatial dimensions.
- Both tensors are modified in place.

Reference: [Local Rotation Augmentation for 3D data](https://www.mdpi.com/2313-433X/9/2/46).

## `augmentations_3d.py`

Augmentations for volumetric data. Both take and return `(chunk, label)` pairs, mutate or
copy their inputs as noted below, and use Python's `random` plus `Pillow` (not declared in
`pyproject.toml`) — `Pillow` is only needed if you import this module.

### `random_local_rotation_3d`

```python
random_local_rotation_3d(in_tensor, tlbl=None, radius=16, p=0.5) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]
```

Applies the augmentation with probability `p`. A random square of side `2 * radius` is
picked, a circular mask is drawn inside it with `Pillow`, a random angle in `[20, 340]` is
drawn, and the masked circular region is rotated — **image chunk and label mask together**,
so labels stay aligned. Pass `tlbl=None` (the default) to augment only the input chunk, in
which case just the rotated chunk is returned.

Caveats:

- Expects a 4D chunk `[b, 1, h, w]` and a 2D label `[h, w]`; a 5D volume raises
  `RuntimeError` (the slices assume 2D).
- The chunk is modified in place.

Reference: [Local Rotation Augmentation for 3D data](https://www.mdpi.com/2313-433X/9/2/46)
(as cited in the function docstring).

### `cutmix`

```python
cutmix(chunks, lbls, mask_size=32) -> tuple[torch.Tensor, torch.Tensor]
```

Picks two adjacent samples of the batch (`idx`, `idx + 1`), chooses a random
`mask_size`-cube position and swaps that cube between them, chunks and labels alike. It
mutates `chunks` and `lbls` in place.

Reference: [CutMix: Regularized Strategy for Mixup on Image Classification](https://arxiv.org/abs/1905.04899).
