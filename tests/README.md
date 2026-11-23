# Tests

Pytest suite. `testpaths = ["tests"]` and `addopts = "-v"` are set in `pyproject.toml`, so a
bare `pytest` from the repository root runs everything here.

```bash
pytest
pytest tests/test_grn.py -k identity
pytest tests/test_repmode.py -k "3d and eval"
```

`conftest.py` is currently empty; fixtures and helpers live next to the tests that use them. How
tests are written (plain functions, `parametrize` for variants, seeded inputs, reference
implementations, regression and known-issue tests) is described in
[`CONTRIBUTING.md`](../CONTRIBUTING.md#tests), and `test_conventions.py` enforces the parts that
can be checked.

## Files

| File | Covers |
| --- | --- |
| `test_conventions.py` | The rules of `CONTRIBUTING.md` over the whole package: exports mirror the folder tree, constructor arguments follow torch, lowercase `1d` / `2d` / `3d` suffixes, docstring layout with `Args:` / `Returns:` matching the signature, a `## Source` section linking the paper in every folder README, and the test style of this folder. A second set of tests feeds each check broken examples. |
| `test_grn.py` | `GRN1d` / `GRN2d` / `GRN3d` against a reference for any number of spatial dims: formula, `gamma` / `beta` behaviour, identity at init, gradients, dtypes, non-contiguous inputs, input validation, `chan_last` vs `chan_first`, `state_dict` round trip and loading pre-shaped checkpoints |
| `test_convnext.py` | `ConvNeXtBlock`: V2 by default with a channel-last `GRN2d` and no LayerScale, V1 with LayerScale and no GRN, shapes, invalid versions, `dtype` |
| `test_layers.py` | `Snake` (formula, monotonic, learnable frequency), `LayerNorm2d` (matches `F.layer_norm`, both layouts agree), `SEBlock` (matches a reference, bottleneck width, per-channel gate), `PositionEmbeddingSine` (matches a DETR style reference for several channel counts) |
| `test_rrdb.py` | Dense blocks and RRDB in 2d / 3d: shapes, identity when the last convolutions are zero, the `0.2` residual scaling, growth channels, `bias` |
| `test_repmode.py` | `MoDEConv1d/2d/3d`: every sample uses its own task kernel in train and eval (regression), batched equals single samples, shapes, task dependence, gate softmax, non-contiguous inputs (regression); causal layout and causality; encoder / subnet / decoder shapes; the `MoDE` wrapper's global task swap and task weights; `dtype` |
| `test_augmentations.py` | Random local rotation, 2d and 3d through the same parametrized tests: optional label, label not changing the input's rotation, in place, `p=0` / `p=1`, label alignment, dtypes; 2d integer labels, per-sample draws and input validation; 3d using one patch for the whole chunk |
| `test_cutmix.py` | One `mask_size` cube swapped in both directions between adjacent samples with the matching label window, in place, batch content preserved |
| `test_training.py` | Dataset and `make_dl`, `run_model`, `train_loop` (loss decreases, AMP clips unscaled gradients, scheduler steps; regressions), `eval_loop`, `utils`, and the training script stepping its scheduler (regression) |

## Known issues

Bugs that are not fixed yet are `xfail(strict=True)` tests, so fixing one makes its test fail as
an unexpected pass until the marker is removed:

- `PositionEmbeddingSine` with an odd `(C + 1) // 2` (`test_layers.py`)
- `MoDEConv2d` with `kernel_size != 5`, and the causal `MoDEConv1d` normalization seeing future
  steps (`test_repmode.py`)
- `eval_loop` ignoring `eval_name`, and the reversed loss arguments (`test_training.py`)
