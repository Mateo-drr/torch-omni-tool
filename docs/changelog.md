# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CONTRIBUTING.md` with the code conventions, and `tests/test_conventions.py`, which enforces
  them in CI (exports, constructor arguments, naming, docstrings, folder READMEs).
- Module docstrings for every subpackage.
- Tests for every block, layer and augmentation and for the training scaffold, including
  regression tests for the fixed bugs; unfixed known issues are `xfail(strict=True)` tests.
- Root package now exposes its subpackages, so `import torch_omni_tool as omni` gives
  `omni.activations.Snake`, `omni.blocks.GRN` and the rest of the public API.

### Changed
- `ConvNeXtBlock` takes a `version` argument and defaults to the ConvNeXt V2 block (GRN after
  the GELU, no LayerScale). Pass `version=1` for the previous behaviour; V1 checkpoints do not
  load into a V2 block.
- Constructor arguments follow torch naming and order, like `nn.Conv2d(in_channels,
  out_channels, ..., device=None, dtype=None)`:
  - `in_channels` / `out_channels` first: `MoDEConv*`, `MoDE*Block*`, `MoDESubNet2Conv*` and
    `MoDE` (previously `num_experts, num_tasks, in_chan, out_chan` / `in_chans, out_chans`);
    `num_experts` moved after `num_tasks` and defaults to 5.
  - Layers and blocks that keep the channel count take `num_features`, like `nn.BatchNorm2d`:
    `ConvNeXtBlock` (was `dim`), `SEBlock` (was `channels`, and `reduce_dim` is now
    `squeeze_channels` as in torchvision), `RRDB2d` / `RRDB3d` / `ResidualDenseBlock_5C_*`
    (were `nf`, `gc` is now `growth_channels`; no longer defaults to 64), `GRN1d/2d/3d` (was
    `dim`) and `Snake` (was `channels`). `LayerNorm2d` keeps `normalized_shape` like
    `nn.LayerNorm`.
  - Every layer accepts `device` and `dtype`. Outputs are unchanged for the same seed.
- Import paths now mirror the folder tree: `torch_omni_tool.blocks` and `torch_omni_tool.datasets`
  only expose their subfolders, and names are imported from the folder that holds them, e.g.
  `from torch_omni_tool.blocks.RRDB import RRDB3d` instead of `from torch_omni_tool.blocks import
  RRDB_3D`. The flat re-exports are removed.
- Dimension variants renamed with a lowercase suffix: `MoDEConv1d` / `MoDEConv2d` /
  `MoDEConv3d`, `MoDE{Encoder,Decoder}Block{1d,3d}`, `MoDESubNet2Conv{1d,3d}` (previously
  `MoDEConv` in both the 1D and 3D modules, exported as `...1D` / `...3D`), `RRDB2d` / `RRDB3d`
  and `ResidualDenseBlock_5C_2d` / `ResidualDenseBlock_5C_3d` (previously `RRDB` / `RRDB_3D`
  and `ResidualDenseBlock_5C` / `ResidualDenseBlock_5C_3D`). `RRDB_2D.py` is now `RRDB_2d.py`.
- Augmentations moved to one folder per augmentation, like `blocks`:
  `datasets/RandomLocalRotation/RandomLocalRotation_{2d,3d}.py` and `datasets/CutMix/CutMix_3d.py`.
  The old `datasets.augmentations_2d` and `datasets.augmentations_3d` module paths are gone.
- `GRN` is split into `GRN1d`, `GRN2d` and `GRN3d` on top of one dimension-agnostic base (like
  `nn.BatchNorm1d/2d/3d`), takes a `data_format` argument (`chan_first` default, or
  `chan_last`) and validates the input rank and channel count. `gamma` / `beta` are now stored
  as `[C]`; checkpoints with the old `[1, C, 1, 1]` shape or the official ConvNeXt V2
  `[1, 1, 1, C]` shape are reshaped on load.
- Documentation is built with Zensical (the successor to Material for MkDocs) from the
  existing `mkdocs.yml`, keeping the auto-generated mkdocstrings API reference.

### Fixed
- With mixed precision, gradient clipping ran on gradients still multiplied by the loss scale;
  `train_loop` now unscales them first.
- The training script built a learning rate scheduler but never stepped it.
- `MoDEConv1d` / `MoDEConv3d` used the first sample's task kernel for the whole batch in eval
  mode; every sample now uses its own kernel.
- `MoDEConv2d` failed on non-contiguous inputs (`view` instead of `reshape`).

## [0.1.2] - 2026-10-15

### Added
- Documentation site (MkDocs with the Material theme) including a full, auto-generated
  API reference, deployed to GitHub Pages on push to `main`.

### Changed
- Updated the `random_local_rotation_3d` local rotation augmentation.
- Style cleanup across the package.

## [0.1.1] - 2026-10-12

### Added
- Initial release of torch-omni-tool
- Core blocks, training scaffold, utilities
- Type hints and py.typed marker
