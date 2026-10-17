"""Reusable PyTorch blocks, layers and utilities.

The package is importable as one namespace, so the subpackages are reachable
straight off the root::

    import torch_omni_tool as omni

    out = omni.activations.Snake()(x)
    block = omni.blocks.ConvNeXtBlock(dim=64)

Public names can also be imported directly from their subpackages::

    from torch_omni_tool.blocks import ConvNeXtBlock, GRN
    from torch_omni_tool.blocks.RepMoDE import MoDEConv1D, MoDEConv3D
    from torch_omni_tool.normalizations import LayerNorm
    from torch_omni_tool.activations import Snake
    from torch_omni_tool.datasets import make_dl
"""

from torch_omni_tool import activations, blocks, datasets, normalizations, utils

__all__ = ["activations", "blocks", "datasets", "normalizations", "utils"]
