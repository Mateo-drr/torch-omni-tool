# Architecture

## Package Layout

- `torch_omni_tool.config` – `Config` dataclass and module-level `config` singleton
- `torch_omni_tool.model` – `SampleNet` (placeholder autoencoder-style model)
- `torch_omni_tool.training` – entry point wiring pieces together
- `torch_omni_tool.loops` – `run_model`, `train_loop`, `eval_loop`
- `torch_omni_tool.utils` – timing, metric formatting, epoch bookkeeping helpers
- `torch_omni_tool.blocks` – network blocks (ConvNeXt V2, RepMoDE 1D/2D/3D, RRDB 2D/3D, SEB, ViT)
- `torch_omni_tool.normalizations` – `LayerNorm` implementations (note filename: `layern_norm_2d.py`)
- `torch_omni_tool.datasets` – dataset/dataloader factory and augmentations
- `torch_omni_tool.activations` – `Snake`

## Training Flow

1. Load config (`torch_omni_tool.config`)
2. Build dataloader via `torch_omni_tool.datasets.make_dl`
3. Instantiate `torch_omni_tool.model.SampleNet`
4. Run loops in `torch_omni_tool.loops` (train/eval) with utilities in `torch_omni_tool.utils`
5. Log to wandb if enabled

## Blocks & Variants

Many blocks come in dimension-specific variants. Pick the variant matching your spatial dims:

| Block | Dims | Notes |
|---|---|---|
| RepMoDE | 1D, 2D, 3D | Variants in `torch_omni_tool.blocks.RepMoDE`. Public API re-exports dimensional aliases. |
| RRDB | 2D, 3D | `RRDB_2D.py`, `RRDB_3d.py` (casing differs). |

## Imports & Conventions

- Mixed import styles exist in the codebase. Prefer running as module (`python -m torch_omni_tool.training`) or using `torch_omni_tool.*` imports when modifying.
- `ConvNeXtV2.ConvNeXtV2` imports `LayerNorm` as `from torch_omni_tool.normalizations.layern_norm_2d import LayerNorm`.
- Subpackages re-export public APIs via `__all__`; root package stays thin.
- Type hints are present; `mypy` runs in strict mode. Few docstrings currently.

## Known Gotchas (from codebase)

- `RepMoDE_1d.one_hot_task_embedding` and `RepMoDE_3d.one_hot_task_embedding` are module-level functions taking `self: Any` and cannot be called.
- `MoDEConv(..., causal=True)` in `RepMoDE_1d.py` pads the channel dim instead of the time dim.
- `PositionEmbeddingSine` needs even `(channels + 1) // 2`.
- `augmentations_3d.random_local_rotation_3d` expects specific shapes; 5D volumes may raise.
- `training.py` builds a scheduler but passes `scheduler=None` to `train_loop`.
- In RepMoDE 3D/1D `MoDEConv` eval mode, only first task's expert weights are used for the whole batch.
- Filename typo: `layern_norm_2d.py` (keep import paths consistent if renaming).
- Casing inconsistencies exist (`RRDB_2D.py` vs `RRDB_3d.py`).
