"""
Reusable PyTorch blocks, layers and utilities.

The package is importable as one namespace, so the subpackages are reachable straight off the root::

    import torch_omni_tool as omni

    out = omni.activations.Snake(64)(x)
    block = omni.blocks.ConvNeXtV2.ConvNeXtBlock(dim=64)

Public names are exported by the folder that contains their file::

    from torch_omni_tool.blocks.ConvNeXtV2 import ConvNeXtBlock, GRN2d
    from torch_omni_tool.blocks.RepMoDE import MoDEConv1d, MoDEConv3d
    from torch_omni_tool.normalizations import LayerNorm2d
    from torch_omni_tool.activations import Snake
    from torch_omni_tool.datasets import make_dl
    from torch_omni_tool.datasets.CutMix import cutmix
"""

from torch_omni_tool import activations, blocks, datasets, normalizations, utils

__all__ = ["activations", "blocks", "datasets", "normalizations", "utils"]
