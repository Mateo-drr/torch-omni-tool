# Tests

Pytest suite. `testpaths = ["tests"]` and `addopts = "-v"` are set in `pyproject.toml`, so a
bare `pytest` from the repository root runs everything here.

```bash
pytest
pytest tests/test_grn.py -k identity
```

`conftest.py` is currently empty; fixtures live next to the tests that use them.

## `test_grn.py`

Tests for [`GRN1d` / `GRN2d` / `GRN3d`](../torch_omni_tool/blocks/ConvNeXtV2/README.md).
Inputs are generated with a fixed-seed `torch.Generator` so runs are reproducible, and the
block is cross-checked against a straightforward re-implementation of the GRN formula
(`_reference_grn`) rather than against hardcoded numbers.

What is covered:

- **Shapes** — output matches the input for `[1, 3, 1, 1]`, `[2, 8, 5, 7]` and
  `[4, 16, 32, 32]`; non-contiguous inputs; float32/float64 dtype preservation; a channel
  count that does not match `dim` and an input of the wrong rank both raise `ValueError`.
- **Formula** — equality with the reference implementation, norms computed over `H, W`
  before being normalized across `C`, `gamma` scaling and `beta` shifting per channel,
  `beta` acting as a pure additive shift, positive homogeneity when `beta == 0`.
- **Initialization and parameters** — `gamma`/`beta` shapes, both zero-initialized, block is
  the identity at init, `state_dict` round-trip, all-zero input does not produce `NaN`.
- **Variants and layout** — `GRN1d` and `GRN3d` match an N-d reference, `chan_last` matches
  `chan_first` for every variant, the channel check follows `data_format`, and checkpoints
  with pre-shaped `[1, C, 1, 1]` / `[1, 1, 1, C]` parameters load.
- **Autograd** — gradients reach the input and both parameters, input and parameter
  gradients match an autograd reference, `beta` receives a constant gradient, and train and
  eval mode agree.

## `test_convnext.py`

Tests for [`ConvNeXtBlock`](../torch_omni_tool/blocks/ConvNeXtV2/README.md): V2 is the default
and uses a channel-last `GRN2d` without LayerScale, V1 keeps LayerScale (disabled when
`layer_scale_init_value <= 0`) without GRN, both preserve the input shape, and an invalid
`version` raises `ValueError`.

## `test_augmentations.py`

23 tests for the local rotation augmentations in
[`../torch_omni_tool/datasets`](../torch_omni_tool/datasets/README.md). Inputs come from a
fixed-seed `torch.Generator` and the `random` module is seeded per test, so runs are
reproducible.

What is covered:

- **Optional label** — omitting the label (or passing `None` explicitly) returns a bare
  `torch.Tensor`, while passing a label returns `(tensor, label)`; the input is rotated
  identically in both cases because rotating a label consumes no randomness.
- **In-place** — the augmented tensor is the same object that was passed in.
- **Probability** — `p=0` is a no-op for both tensor and label, `p=1` changes every sample.
- **Alignment** — a shared float plane in the input and label stays equal after rotation.
- **Dtypes** — integer labels stay integer and in range, float labels keep their dtype.
- **Validation** — a non-4D input, a non-3D label and an oversized radius raise `ValueError`.

## Adding tests

Keep tests deterministic (seed the generator), assert against a reference implementation
where one is cheap to write, and prefer `torch.testing.assert_close` over manual
`tolerances`. Type hints are required — mypy runs in strict mode and CI runs both tools.
