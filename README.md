# torch-omni-tool

A small PyTorch library of reusable network blocks and layers.

Everything is plain `nn.Module` code with type hints and no framework dependencies, so
blocks can be copied into any project or imported from here.

## Documentation

Full documentation is available at [GitHub Pages](https://Mateo-drr.github.io/torch-omni-tool/).

## Installation

```bash
pip install -e ".[dev]"      # runtime + pytest / pylint / mypy / black / types-tqdm
pip install -e ".[docs]"     # docs dependencies
```

Requires Python >= 3.12.

## What is inside

| Path | Contents |
| --- | --- |
| `torch_omni_tool.blocks` | Network blocks: ConvNeXt V2, GRN, RepMoDE, RRDB, Squeeze-and-Excite, ViT positional embeddings |
| `torch_omni_tool.activations` | Activation layers: the periodic `Snake` activation |
| `torch_omni_tool.normalizations` | `LayerNorm2d` for `[b, c, h, w]` and `[b, h, w, c]` |
| `torch_omni_tool.datasets` | `Dataset` / `DataLoader` factory and augmentations (random local rotation, CutMix) |
| `torch_omni_tool.utils` | Timing and metric formatting helpers |
| [`tests/`](tests/README.md) | Pytest suite (GRN block and local rotation augmentations) |

Import paths mirror the folder tree: every name is exported (with `__all__`) by the folder that
contains its file, and parent packages only expose their subfolders:

```python
import torch_omni_tool as omni

omni.blocks.SEB.SEBlock(64)
omni.activations.Snake(64)

from torch_omni_tool.blocks.ConvNeXtV2 import ConvNeXtBlock, GRN2d
from torch_omni_tool.blocks.RRDB import RRDB3d
from torch_omni_tool.datasets.CutMix import cutmix
from torch_omni_tool.normalizations import LayerNorm2d
```

The root `torch_omni_tool` package stays thin (`import torch_omni_tool` exposes the
subpackages, not the classes). A `py.typed` marker is shipped (PEP 561), so type checkers
and IDEs use the inline type hints.

## Blocks at a glance

| Block | Input | Import |
| --- | --- | --- |
| `ConvNeXtBlock` | `[B, C, H, W]` | `torch_omni_tool.blocks.ConvNeXtV2` |
| `GRN1d`, `GRN2d`, `GRN3d` | `[B, C, L]`, `[B, C, H, W]`, `[B, C, D, H, W]` | `torch_omni_tool.blocks.ConvNeXtV2` |
| `MoDE`, `MoDEConv2d` | `[B, C, H, W]` | `torch_omni_tool.blocks.RepMoDE` |
| `MoDEConv1d`, `MoDEConv3d`, encoder/decoder blocks `1d`/`3d` | `[B, C, L]` and `[B, C, D, H, W]` | `torch_omni_tool.blocks.RepMoDE` |
| `RRDB2d`, `RRDB3d` | `[B, C, H, W]`, `[B, C, D, H, W]` | `torch_omni_tool.blocks.RRDB` |
| `SEBlock` | `[B, C, H, W]` | `torch_omni_tool.blocks.SEB` |
| `PositionEmbeddingSine` | `[B, C, H, W]` | `torch_omni_tool.blocks.ViT` |
| `Snake` | `[B, C, H, W]` | `torch_omni_tool.activations` |

Classes with dimension variants end in `1d`, `2d` or `3d`, like `nn.Conv2d`.

```python
import torch
from torch_omni_tool.blocks.ConvNeXtV2 import GRN2d

y = GRN2d(64)(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
```

## Development

```bash
pytest                                  # tests/
mypy                                    # strict, over torch_omni_tool/
black torch_omni_tool tests scripts     # formatting, line length 100
pylint torch_omni_tool
zensical serve                          # docs preview
```

CI (`.github/workflows/ci.yml`) runs black, a line length check, pylint (not blocking yet),
mypy and pytest on Python 3.12.

Code style: black with a 100 column limit, and Google style docstrings that start on the line
after the opening quotes and have no blank lines (`Args:` / `Returns:` items indented by 4 spaces,
`name (type): description`). The docs build parses them through
[`scripts/griffe_compact.py`](scripts/griffe_compact.py).

## Known issues

- `PositionEmbeddingSine` fails when `(C + 1) // 2` is odd (e.g. `C = 6`).
- `MoDEConv2d` only works with `kernel_size=5` and `stride=1`.
- The causal `MoDEConv1d` with `conv_type="normal"` leaks future steps through
  `InstanceNorm1d`.
- `RepMoDE_1d.one_hot_task_embedding` and `RepMoDE_3d.one_hot_task_embedding` take `self` and
  cannot be called as plain functions.
- `random_local_rotation_3d` works on `[b, 1, h, w]` slices only; `cutmix` needs a batch of at
  least 2.

See the folder READMEs for block specific caveats.

## References

The blocks are re-implementations of published work. Each folder README has the full citation
and the differences from the reference code.

| Implementation | Paper | Reference code |
| --- | --- | --- |
| `ConvNeXtBlock` (`version=1`), `LayerNorm2d` | Liu et al., [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), CVPR 2022 | [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt) |
| `GRN1d` / `GRN2d` / `GRN3d`, `ConvNeXtBlock` (`version=2`) | Woo et al., [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808), CVPR 2023 | [facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2) |
| `MoDE*` (RepMoDE) | Zhou et al., [RepMode: Learning to Re-parameterize Diverse Experts for Subcellular Structure Prediction](https://arxiv.org/abs/2212.10066), CVPR 2023 | [correr-zhou/RepMode](https://github.com/correr-zhou/RepMode) |
| `RRDB2d`, `RRDB3d` | Wang et al., [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219), ECCV Workshops 2018 | [xinntao/ESRGAN](https://github.com/xinntao/ESRGAN) |
| `SEBlock` | Hu et al., [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507), CVPR 2018 | [hujie-frank/SENet](https://github.com/hujie-frank/SENet) (Caffe) |
| `PositionEmbeddingSine` | Carion et al., [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020 | [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| `Snake` | Liu Ziyin et al., [Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195), NeurIPS 2020 | [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN) (third party) |
| `random_local_rotation_2d`, `random_local_rotation_3d` | Alomar et al., [Data Augmentation in Classification and Segmentation: A Survey and New Strategies](https://doi.org/10.3390/jimaging9020046), J. Imaging 2023 | not linked in the paper |
| `cutmix` | Yun et al., [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899), ICCV 2019 | [clovaai/CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch) |

## License

See `LICENSE`.
