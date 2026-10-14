# Usage

## Configuration

`torch_omni_tool.config.Config` holds all hyperparameters and options for a run. The module provides a `config` singleton that training code uses. There is no CLI or config file parsing; override attributes on the singleton or edit defaults as needed.

Print defaults:

```bash
python -m torch_omni_tool.config
```

## Training

The scaffold wires together config, data loading, model, and loops. Run it with:

```bash
python -m torch_omni_tool.training
```

Flow: `training.py` → `make_dl()` (from `torch_omni_tool.datasets`) → `SampleNet` → `loops.train_loop()` / `loops.eval_loop()` → `utils.finish_epoch()`. 

Key details:
- Uses `torch.amp.autocast(device_type=config.device)` and `torch.amp.GradScaler` when `config.half_p` is enabled.
- Loss is autoencoder-style: `criterion(samples["data"], output)` so the target is the input.
- Training body is guarded by `if __name__ == "__main__":` so importing has no side effects.
- Scheduler is created in `training.py` but not passed to `train_loop` (see Known Issues in README context) — be aware if modifying.

## Data Loading

`torch_omni_tool.datasets.cstm_ds` provides `make_dl()`. The dataset returns samples with key `data` (e.g. `{'data': tensor}`), which is what `loops.run_model` expects.

## Imports

Preferred: import from public subpackages.

```python
from torch_omni_tool.blocks import SEBlock, GRN
from torch_omni_tool.normalizations import LayerNorm
from torch_omni_tool.activations import Snake
```

Leaf paths also work if needed. Be consistent with existing naming (some files use `2D` vs `3d` casing).

> Note: The normalization module filename is `layern_norm_2d.py` (typo). The import path is `torch_omni_tool.normalizations.layern_norm_2d` as used in code.
