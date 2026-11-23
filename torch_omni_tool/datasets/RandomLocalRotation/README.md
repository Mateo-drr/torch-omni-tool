# RandomLocalRotation

Random local rotation (RLR): rotate a randomly placed circular region of an image instead of
the whole image, so no empty corners appear. Labels are rotated with the same patch and stay
aligned.

| Function | File | Input |
| --- | --- | --- |
| `random_local_rotation_2d` | [`RandomLocalRotation_2d.py`](RandomLocalRotation_2d.py) | `[b, c, h, w]`, optional label `[b, h, w]` |
| `random_local_rotation_3d` | [`RandomLocalRotation_3d.py`](RandomLocalRotation_3d.py) | `[b, 1, h, w]`, optional label `[h, w]` |

## Source

- **Paper:** Khaled Alomar, Halil Ibrahim Aysel, Xiaohao Cai, *Data Augmentation in
  Classification and Segmentation: A Survey and New Strategies*, Journal of Imaging 9(2), 46,
  2023 — [doi:10.3390/jimaging9020046](https://doi.org/10.3390/jimaging9020046)
- **Reference code:** the paper does not link any code, both functions are written from its
  description.

## `random_local_rotation_2d`

```python
random_local_rotation_2d(in_tensor, label=None, radius=16, p=0.5)
```

Every sample is selected independently with probability `p` and gets its own patch position and
angle in `[20, 340]` degrees. Returns the tensor, or `(tensor, label)` when a label is given.

- Integer labels are rotated with nearest interpolation and stay integer.
- Raises `ValueError` for a non-4D input, a non-3D label or a patch larger than the image.
- Both tensors are modified in place.

```python
import torch
from torch_omni_tool.datasets.RandomLocalRotation import random_local_rotation_2d

images, labels = torch.rand(4, 3, 64, 64), torch.randint(0, 5, (4, 64, 64))
images, labels = random_local_rotation_2d(images, labels, radius=16, p=0.5)
```

## `random_local_rotation_3d`

```python
random_local_rotation_3d(in_tensor, tlbl=None, radius=16, p=0.5)
```

With probability `p` one patch position and angle is drawn and applied to every sample of the
chunk and to the label. The chunk and label are modified in place.

## Caveats

- `random_local_rotation_3d` works on 2D slices despite its name; a 5D volume fails.
- `random_local_rotation_3d` does not check that the patch fits, an oversized `radius` fails
  inside `random.randint`.
