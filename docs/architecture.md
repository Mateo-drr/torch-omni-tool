# Architecture

## Package Layout

All public code lives in the subpackages, each of which re-exports its API via `__all__`:

- `torch_omni_tool.blocks` – network blocks (ConvNeXt V2, GRN, RepMoDE 1D/2D/3D, RRDB 2D/3D, SEB, ViT)
- `torch_omni_tool.normalizations` – `LayerNorm` implementations (note filename: `layern_norm_2d.py`)
- `torch_omni_tool.activations` – `Snake`
- `torch_omni_tool.datasets` – dataset/dataloader factory and 2D/3D augmentations
- `torch_omni_tool.utils` – timing and metric formatting helpers

The root `torch_omni_tool` package stays thin: it exposes the subpackages, not the classes.

## Blocks & Variants

Many blocks come in dimension-specific variants. Pick the variant matching your spatial dims:

| Block | Dims | Notes |
|---|---|---|
| RepMoDE | 1D, 2D, 3D | Variants in `torch_omni_tool.blocks.RepMoDE`. Public API re-exports dimensional aliases. |
| RRDB | 2D, 3D | `RRDB_2D.py`, `RRDB_3d.py` (casing differs). |

## Imports & Conventions

- Subpackages re-export public APIs via `__all__`; the root package stays thin.
- `ConvNeXtV2.ConvNeXtV2` imports `LayerNorm` as `from torch_omni_tool.normalizations.layern_norm_2d import LayerNorm`.
- Type hints are present; `mypy` runs in strict mode.
- Casing inconsistencies exist (`RRDB_2D.py` vs `RRDB_3d.py`), and the normalization filename
  has a typo (`layern_norm_2d.py`). Keep import paths consistent if renaming.

## Known Gotchas

- `RepMoDE_1d.one_hot_task_embedding` and `RepMoDE_3d.one_hot_task_embedding` are module-level
  functions taking `self: Any` and cannot be called.
- `MoDEConv(..., causal=True)` in `RepMoDE_1d.py` pads the channel dim instead of the time dim.
- `PositionEmbeddingSine` needs even `(channels + 1) // 2`.
- `augmentations_3d.random_local_rotation_3d` expects specific shapes; 5D volumes may raise.
- In RepMoDE 3D/1D `MoDEConv` `eval()` mode, only the first task's expert weights are used for
  the whole batch.
