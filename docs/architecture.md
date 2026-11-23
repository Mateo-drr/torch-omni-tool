# Architecture

## Package Layout

```
torch_omni_tool/
├── activations/        Snake
├── normalizations/     LayerNorm2d
├── blocks/
│   ├── ConvNeXtV2/     ConvNeXtBlock, GRN1d, GRN2d, GRN3d
│   ├── RepMoDE/        MoDE, MoDEConv{1d,2d,3d}, MoDE{Encoder,Decoder}Block{1d,3d}, MoDESubNet2Conv{1d,3d}
│   ├── RRDB/           RRDB2d, RRDB3d, ResidualDenseBlock_5C_2d, ResidualDenseBlock_5C_3d
│   ├── SEB/            SEBlock
│   └── ViT/            PositionEmbeddingSine
├── datasets/           CustomDataset, make_dl
│   ├── CutMix/         cutmix
│   └── RandomLocalRotation/   random_local_rotation_2d, random_local_rotation_3d
└── utils.py            timing and metric formatting helpers
```

`config.py`, `model.py`, `loops.py` and `training.py` hold a sample training setup, run it with
`python -m torch_omni_tool.training`.

## Conventions

- **Imports mirror the tree.** Every name is exported (with `__all__`) by the folder that
  contains its file; parent packages only expose their subfolders. The root package exposes
  the subpackages, not the classes.
- **One folder per block or augmentation.** Each holds the implementation files, an
  `__init__.py` exporting the public names and a README with the paper, the reference code and
  the differences from it.
- **Dimension variants** end in `1d`, `2d` or `3d` (like `nn.Conv2d`), and their files in
  `_1d.py`, `_2d.py` or `_3d.py`. Layers without a kernel, such as GRN, keep the math in one
  base class and the variants only check the input rank, like `nn.BatchNorm1d/2d/3d`.
- **Channel layout:** layers that support both layouts take
  `data_format="chan_first" | "chan_last"`.
- **Typing and style:** full type hints checked by `mypy` in strict mode, black with a 100
  column limit, and Google style docstrings without blank lines (the docs parse them through
  `scripts/griffe_compact.py`).

## Known Gotchas

- `PositionEmbeddingSine` fails when `(C + 1) // 2` is odd (e.g. `C = 6`); use a multiple of 4.
- `MoDEConv2d` only works with `kernel_size=5` and `stride=1`; `MoDEConv1d` / `MoDEConv3d`
  ignore `stride`.
- With `causal=True`, `MoDEConv1d` takes and returns `[B, L, C]`. Use `conv_type="final"` for
  strict causality, the `InstanceNorm1d` of `conv_type="normal"` sees future steps.
- `RepMoDE_1d.one_hot_task_embedding` and `RepMoDE_3d.one_hot_task_embedding` take `self` and
  cannot be called as plain functions; build the one-hot tensor with `F.one_hot`.
- `random_local_rotation_3d` works on `[b, 1, h, w]` slices only, and `cutmix` needs a batch of
  at least 2.
