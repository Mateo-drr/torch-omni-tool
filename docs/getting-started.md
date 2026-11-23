# Getting Started

## Installation

Install in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

For building documentation locally:

```bash
pip install -e ".[docs]"
```

Requires Python >= 3.12.

## Quick Verification

Import a block and run a forward pass:

```python
import torch
from torch_omni_tool.blocks.ConvNeXtV2 import GRN2d

x = torch.randn(2, 64, 56, 56)
y = GRN2d(64)(x)
print(y.shape)  # torch.Size([2, 64, 56, 56])
```

The subpackages are also reachable off the root package, so with
`import torch_omni_tool as omni` the same layer is `omni.blocks.ConvNeXtV2.GRN2d(64)`.

## Development Tools

```bash
pytest                                  # run the tests
mypy                                    # type check (strict, over torch_omni_tool/)
black torch_omni_tool tests scripts     # format (line length 100)
pylint torch_omni_tool                  # lint
zensical serve                          # preview these docs
```

CI runs black, a line length check, pylint (not blocking yet), mypy and pytest on every push.
