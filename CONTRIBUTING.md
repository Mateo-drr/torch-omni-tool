# Contributing

The rules that keep `torch_omni_tool` consistent. Most of them are checked automatically, see
[How the rules are enforced](#how-the-rules-are-enforced); the rest are judgment calls for review.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,docs]"
```

Before pushing, run what CI runs:

```bash
black torch_omni_tool tests scripts
pylint torch_omni_tool tests scripts --disable=all --enable=line-too-long
mypy
pytest
zensical build    # when docs or docstrings changed
```

## Formatting

- [black](https://black.readthedocs.io) with `line-length = 100` (set in `pyproject.toml`).
- Black does not wrap docstrings, comments or long strings: wrap them by hand within 100
  columns, as late as possible.
- Absolute imports only (`from torch_omni_tool.blocks.RRDB import RRDB2d`).

## Typing

`mypy` runs in strict mode: every function has full annotations, including the return type.
Pass keyword arguments explicitly instead of unpacking a `**dict` of mixed types, which strict
mypy rejects.

## Docstrings

Google style, with three project rules: the text starts on the line after the opening `"""`,
there are no blank lines inside, and section items are indented by 4 spaces (8 for continuation
lines).

```python
def scale(x: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """
    Multiply the input by a constant
    Args:
        x (torch.Tensor): Input tensor with shape [B, C, H, W]
        factor (float): Value every element is multiplied by, a description that is too long for
            one line continues with an eight space indent
    Returns:
        torch.Tensor: Scaled tensor with the same shape as the input
    """
```

- Every module, class and function has a docstring (`@overload` stubs excepted).
- `Args:` lists every parameter except `self` / `cls`, in signature order, as
  `name (type): description` with the type written exactly as in the annotation.
- `Returns:` is present when the function returns something, as `type: description`.
- Docstrings with code examples (`::` or `>>>`) may keep the blank lines and indentation the
  examples need.

The docs parser (griffe) normally needs a blank line before `Args:` / `Returns:`;
[`scripts/griffe_compact.py`](scripts/griffe_compact.py) inserts it at docs build time. If you
use a section name it does not know yet, add it to `_SECTION_NAMES` there.

## Package layout and exports

```
torch_omni_tool/
├── blocks/<Name>/       one folder per block family
├── datasets/<Name>/     one folder per augmentation
├── activations/, normalizations/
```

- **One folder per block or augmentation**, holding the implementation files, an `__init__.py`
  with `__all__`, and a README (see [Adding a block](#adding-a-block-or-augmentation)).
- **Import paths mirror the tree.** A name is exported only by the folder that contains its
  file, and parent packages only expose their subfolders. No flat re-exports, no aliases:
  the exported name always equals the object's `__name__`.
- **A class never shadows its own file:** don't export a class with the same name as a module
  in the same folder (e.g. a class `GRN` next to `GRN.py`).

## Naming

- **Dimension variants** end in a lowercase `1d` / `2d` / `3d` like `nn.Conv2d`
  (`MoDEConv3d`, `RRDB2d`, `ResidualDenseBlock_5C_3d`), and their files in `_1d.py` /
  `_2d.py` / `_3d.py`.
- **Layers without a kernel** (norms and similar) keep the math in one dimension-agnostic base
  class, and the `1d` / `2d` / `3d` classes only fix the expected input rank, like
  `nn.BatchNorm1d/2d/3d` (see `GRN1d/2d/3d`).
- Folder names use the paper's name in CamelCase (`ConvNeXtV2`, `RandomLocalRotation`).

## Constructor arguments

Follow torch's names and order:

| Layer | Arguments | Torch precedent |
| --- | --- | --- |
| Changes the channel count | `in_channels, out_channels, ...` first | `nn.Conv2d` |
| Keeps the channel count (blocks, norms, activations) | `num_features, ...` first | `nn.BatchNorm2d` |
| LayerNorm style | `normalized_shape, ...` first | `nn.LayerNorm` |
| Other channel counts | `<role>_channels` (`growth_channels`, `squeeze_channels`) | torchvision |
| Every layer with parameters or buffers | `device=None, dtype=None` last | all of `torch.nn` |

- `device` and `dtype` are passed to every parameter, buffer and sublayer the layer creates.
- Layers that accept both channel layouts take
  `data_format: Literal["chan_first", "chan_last"] = "chan_first"`.
- No abbreviations such as `nf`, `gc`, `dim`, `in_chan`, `n_out`.

## Adding a block or augmentation

1. Create `blocks/<Name>/` (or `datasets/<Name>/`) with the implementation files, named after
   the dimension variants they hold.
2. Export the public names in `<Name>/__init__.py` with `__all__`, and add `<Name>` to the
   parent package's imports and `__all__`.
3. Write `<Name>/README.md` with the shared layout: one-line description, a table of
   classes / files / input shapes, then a `## Source` section with
   - **Paper:** authors, title, venue and year, with a link (arXiv or DOI);
   - **Reference code:** the official repository, linked to the exact file when possible,
     or a statement that none was published;
   - how this implementation differs from the reference code,

   followed by usage with a runnable example, and caveats.
4. Add the row to the parent README's table (`blocks/README.md` or `datasets/README.md`) and to
   the references table in the root `README.md`.
5. Add tests in `tests/` (see below).
6. Add an entry under `## [Unreleased]` in [`docs/changelog.md`](docs/changelog.md).

Verify citations against the paper and the repository: titles, arXiv IDs and file paths.

## Tests

Layout and naming:

- One test module per area, `tests/test_<area>.py` (`test_grn.py`, `test_training.py`).
- Plain test functions named after the behaviour they check, `test_<what>_<expected>`
  (`test_identity_at_init`, `test_rejects_other_ranks`). No test classes: shared setup goes in
  a fixture or a small helper. A `NamedTuple` describing test cases is fine.
- Module-level helpers start with an underscore (`_randn`, `_reference`) and have a docstring.
- Test modules are fully typed; `mypy` checks `tests/` in strict mode.

Writing tests:

- One behaviour per test. Use `@pytest.mark.parametrize` for input variants (shapes,
  dimensions, dtypes, layouts, options) instead of copies of the same test or a loop inside it,
  and give it readable `ids` when the values are not (`ids=["1d", "2d", "3d"]`). Stack
  `parametrize` decorators to cross independent variants.
- When a behaviour exists for several dimension variants, parametrize over all of them; keep
  variant-specific behaviour in its own test, named after the variant (`test_2d_...`).
- Deterministic: build inputs from a seeded `torch.Generator`, and seed `random` when the code
  under test uses it.
- Compare against a straightforward reference implementation where one is cheap to write,
  rather than hard-coded numbers, and use `torch.testing.assert_close` instead of manual
  tolerances.
- Test what the code does, not what torch does: every test calls the code under test.
- Every fixed bug gets a regression test with a `# regression: ...` comment explaining what
  used to go wrong.
- A known bug that is not fixed yet gets a test marked
  `@pytest.mark.xfail(strict=True, reason="known issue: ...")`. It documents the bug, and fails
  as an unexpected pass once the bug is fixed, so the marker gets removed with the fix.
- Silence expected warnings on the test that triggers them with
  `@pytest.mark.filterwarnings`, with a comment saying why.

## Changelog and versions

- Changes go under `## [Unreleased]` in [`docs/changelog.md`](docs/changelog.md), following
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
- [Semantic Versioning](https://semver.org/spec/v2.0.0.html): while the version is `0.x`, a
  breaking change (renamed argument, moved import path, changed default) bumps the minor
  version.
- Pushing to `main` deploys the docs; publishing a GitHub Release publishes to PyPI.

## How the rules are enforced

| Rule | Checked by |
| --- | --- |
| Formatting | `black --check` in CI |
| 100 column limit, docstrings and comments included | pylint `line-too-long` in CI |
| Type hints, package and tests | `mypy` (strict) in CI |
| Exports mirror the tree, exported name equals `__name__`, no shadowed files | `tests/test_conventions.py` |
| `device` / `dtype` last, `in_channels` / `out_channels` / `num_features`, no abbreviations | `tests/test_conventions.py` |
| Lowercase `1d` / `2d` / `3d` suffixes on classes and files | `tests/test_conventions.py` |
| Docstring layout, `Args:` matching the signature and annotations, `Returns:` present | `tests/test_conventions.py` |
| Every block / augmentation folder has a README with a `## Source` section and links | `tests/test_conventions.py` |
| Tests are plain functions (no `Test*` classes), helpers start with an underscore and have a docstring | `tests/test_conventions.py` |
| Citations are correct, differences from the reference are documented, tests are meaningful | review |
