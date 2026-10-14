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

## Activate Environment

If using the repo's virtual environment:

```bash
. .venv/bin/activate
```

## Quick Verification

Import a block and run a forward pass:

```python
import torch
from torch_omni_tool.blocks import GRN

x = torch.randn(2, 64, 56, 56)
y = GRN(64)(x)
print(y.shape)          # torch.Size([2, 64, 56, 56])
```

## Development Tools

```bash
pytest                       # Run tests
mypy torch_omni_tool         # Type check (strict mode)
pylint torch_omni_tool       # Linting
```
