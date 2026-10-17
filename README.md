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
| `torch_omni_tool.normalizations` | `LayerNorm` for `[b, c, h, w]` and `[b, h, w, c]` |
| `torch_omni_tool.datasets` | `Dataset` / `DataLoader` factory and 2D/3D augmentations |
| `torch_omni_tool.utils` | Timing and metric formatting helpers |
| [`tests/`](tests/README.md) | Pytest suite (currently the GRN block) |

Each subpackage `__init__.py` re-exports its public API (with `__all__`), so blocks are imported
from their subpackage, or reached straight off the root package:

```python
import torch_omni_tool as omni

omni.blocks.SEBlock(64)
omni.activations.Snake(64)

from torch_omni_tool.blocks import SEBlock, GRN
from torch_omni_tool.normalizations import LayerNorm
from torch_omni_tool.blocks.ConvNeXtV2.GRN import GRN  # leaf path, also valid
```

The root `torch_omni_tool` package stays thin (`import torch_omni_tool` exposes the
subpackages, not the classes). A `py.typed` marker is shipped (PEP 561), so type checkers
and IDEs use the inline type hints.

Note the inconsistent casing in the paths (`RRDB_2D.py` vs `RRDB_3d.py`) and the typo in
`normalizations/layern_norm_2d.py`.

## Blocks at a glance

| Block | Input | Import |
| --- | --- | --- |
| `ConvNeXtBlock` | `[B, C, H, W]` | `torch_omni_tool.blocks.ConvNeXtV2` |
| `GRN` | `[B, C, H, W]` | `torch_omni_tool.blocks.ConvNeXtV2` |
| `MoDE`, `MoDEConv2D` | `[B, C, H, W]` | `torch_omni_tool.blocks.RepMoDE` |
| `MoDEConv1D`, `MoDEConv3D`, encoder/decoder blocks | `[B, C, L]` and `[B, C, D, H, W]` | `torch_omni_tool.blocks.RepMoDE` |
| `RRDB`, `RRDB_3D` | `[B, C, H, W]`, `[B, C, D, H, W]` | `torch_omni_tool.blocks.RRDB` |
| `SEBlock` | `[B, C, H, W]` | `torch_omni_tool.blocks.SEB` |
| `PositionEmbeddingSine` | `[B, C, H, W]` | `torch_omni_tool.blocks.ViT` |
| `Snake` | `[B, C, H, W]` | `torch_omni_tool.activations` |

The RepMoDE 1D/3D variants share class names in their leaf modules, so each subpackage
re-exports them with dimensional aliases: `MoDEConv1D` / `MoDEConv3D`,
`MoDEDecoderBlock1D` / `MoDEDecoderBlock3D`, `MoDEEncoderBlock1D` / `MoDEEncoderBlock3D`,
`MoDESubNet2Conv1D` / `MoDESubNet2Conv3D`.

```python
import torch
from torch_omni_tool.blocks import GRN

y = GRN(64)(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
```

## Development

```bash
pytest                      # tests/ (testpaths set in pyproject.toml)
mypy torch_omni_tool        # strict mode over torch_omni_tool/
pylint torch_omni_tool
```

CI (`.github/workflows/ci.yml`) runs Python 3.12 on Ubuntu: install, pylint, mypy and
pytest. Mypy is clean in strict mode; pylint is marked `continue-on-error` until the
remaining warnings are fixed.

## Known issues

- `RepMoDE_1d.one_hot_task_embedding` and `RepMoDE_3d.one_hot_task_embedding` are module
  level functions that take `self: Any` and cannot be called.
- `MoDEConv(..., causal=True)` in `RepMoDE_1d.py` pads the channel dim instead of the time
  dim.
- `PositionEmbeddingSine` needs an even `(channels + 1) // 2`; odd channel counts raise
  from `torch.stack`.
- `augmentations_3d.random_local_rotation_3d` expects a 4D `[b, 1, H, W]` chunk and a 2D
  label; 5D volumes raise `RuntimeError`.
- In `RepMoDE_3d.MoDEConv` (and `RepMoDE_1d.MoDEConv` in `eval()` mode), only the first
  task's expert weights are used for the whole batch.
- See the folder READMEs for block specific caveats.

## References

The blocks are re-implementations of published architectures. Original sources:

| Block | Source |
| --- | --- |
| `ConvNeXtBlock` | [ConvNeXt V2: ConvNet Modality Masking Pretraining](https://arxiv.org/abs/2301.03524) — [repo](https://github.com/facebookresearch/ConvNeXt-V2) |
| `GRN`, `LayerNorm` | [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03597) — [repo](https://github.com/facebookresearch/ConvNeXt) |
| `MoDE*` (RepMoDE) | [RepMode: Learning to Re-parameterize Diverse Experts for Subcellular Structure Prediction (CVPR 2023)](https://arxiv.org/abs/2212.10066) — [repo](https://github.com/correr-zhou/RepMode) |
| `RRDB`, `RRDB_3D` | [ESRGAN: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1809.00219) — [repo](https://github.com/xinntao/ESRGAN) |
| `SEBlock` | [Squeeze-and-Excitation Networks (CVPR 2018)](https://arxiv.org/abs/1709.01507) |
| `PositionEmbeddingSine` | [End-to-End Object Detection with Transformers (DETR)](https://arxiv.org/abs/2005.12872) — [repo](https://github.com/facebookresearch/detr) |
| `Snake` | [Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195) |
| `random_local_rotation_2d` | [Local Rotation Augmentation for 3D data](https://www.mdpi.com/2313-433X/9/2/46) |
| `random_local_rotation_3d` | [Local Rotation Augmentation for 3D data](https://www.mdpi.com/2313-433X/9/2/46) |
| `cutmix` | [CutMix: Regularized Strategy for Mixup on Image Classification](https://arxiv.org/abs/1905.04899) |

## License

See `LICENSE`.
