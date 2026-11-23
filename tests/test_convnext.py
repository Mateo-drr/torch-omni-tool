"""
Tests for the ConvNeXt block versions
"""

from typing import Any, Literal

import pytest
import torch

from torch_omni_tool.blocks.ConvNeXtV2 import ConvNeXtBlock, GRN2d


def _make_input(*shape: int) -> torch.Tensor:
    """
    Build a reproducible random tensor
    Args:
        shape (int): Shape of the tensor
    Returns:
        torch.Tensor: Random tensor with the given shape
    """
    gen = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=gen)


@pytest.mark.parametrize(
    ("kwargs", "has_grn", "has_layer_scale"),
    [
        ({}, True, False),
        ({"version": 2, "layer_scale_init_value": 1e-6}, True, False),
        ({"version": 1}, False, True),
        ({"version": 1, "layer_scale_init_value": 0.0}, False, False),
    ],
    ids=["default-v2", "v2-ignores-layer-scale", "v1", "v1-without-layer-scale"],
)
def test_version_selects_grn_or_layer_scale(
    kwargs: dict[str, Any], has_grn: bool, has_layer_scale: bool
) -> None:
    block = ConvNeXtBlock(8, **kwargs)
    assert (block.grn is not None) == has_grn
    assert (block.gamma is not None) == has_layer_scale


def test_default_version_is_2() -> None:
    assert ConvNeXtBlock(8).version == 2


@pytest.mark.parametrize("shape", [(1, 8, 7, 7), (2, 16, 9, 11), (3, 32, 14, 14)])
@pytest.mark.parametrize("version", [1, 2])
def test_output_shape_matches_input(version: Literal[1, 2], shape: tuple[int, ...]) -> None:
    assert ConvNeXtBlock(shape[1], version=version)(_make_input(*shape)).shape == shape


@pytest.mark.parametrize("version", [0, 3])
def test_invalid_version_raises(version: int) -> None:
    with pytest.raises(ValueError):
        ConvNeXtBlock(8, version=version)  # type: ignore[arg-type]


def test_version_2_grn_acts_on_the_expanded_channel_last_features() -> None:
    block = ConvNeXtBlock(8)
    assert isinstance(block.grn, GRN2d)
    assert block.grn.data_format == "chan_last"
    assert block.grn.gamma.shape == (32,)


@pytest.mark.parametrize("version", [1, 2])
def test_dtype_reaches_every_parameter(version: Literal[1, 2]) -> None:
    block = ConvNeXtBlock(8, version=version, dtype=torch.float64)
    assert all(p.dtype == torch.float64 for p in block.parameters())
