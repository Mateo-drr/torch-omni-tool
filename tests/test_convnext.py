"""
Tests for the ConvNeXt block versions
"""

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


def test_default_version_is_2() -> None:
    block = ConvNeXtBlock(8)
    assert block.version == 2
    assert block.grn is not None
    assert block.gamma is None


def test_version_1_has_layer_scale_and_no_grn() -> None:
    block = ConvNeXtBlock(8, layer_scale_init_value=1e-6, version=1)
    assert block.grn is None
    assert block.gamma is not None


def test_version_1_layer_scale_can_be_disabled() -> None:
    assert ConvNeXtBlock(8, layer_scale_init_value=0.0, version=1).gamma is None


@pytest.mark.parametrize("version", [1, 2])
def test_output_shape_matches_input(version: int) -> None:
    block = ConvNeXtBlock(16, version=version)  # type: ignore[arg-type]
    assert block(_make_input(2, 16, 9, 11)).shape == (2, 16, 9, 11)


def test_invalid_version_raises() -> None:
    with pytest.raises(ValueError):
        ConvNeXtBlock(8, version=3)  # type: ignore[arg-type]


def test_version_2_grn_acts_on_expanded_channels() -> None:
    block = ConvNeXtBlock(8)
    assert isinstance(block.grn, GRN2d)
    assert block.grn.data_format == "chan_last"
    assert block.grn.gamma.shape == (32,)
