"""
Tests for the residual dense blocks and RRDB in 2d and 3d
"""

import pytest
import torch
from torch import nn
from torch.testing import assert_close

from torch_omni_tool.blocks.RRDB import (
    RRDB2d,
    RRDB3d,
    ResidualDenseBlock_5C_2d,
    ResidualDenseBlock_5C_3d,
)

Case = tuple[type[nn.Module], tuple[int, ...]]
DENSE: list[Case] = [
    (ResidualDenseBlock_5C_2d, (2, 8, 6, 5)),
    (ResidualDenseBlock_5C_3d, (2, 8, 4, 3, 5)),
]
RRDBS: list[Case] = [(RRDB2d, (2, 8, 6, 5)), (RRDB3d, (2, 8, 4, 3, 5))]


def _randn(shape: tuple[int, ...]) -> torch.Tensor:
    """
    Build a reproducible random tensor
    Args:
        shape (tuple[int, ...]): Shape of the tensor
    Returns:
        torch.Tensor: Random normal tensor
    """
    return torch.randn(shape, generator=torch.Generator().manual_seed(0))


def _silence_last_conv(block: nn.Module) -> None:
    """
    Zero the last convolution of every dense block so each one becomes the identity
    Args:
        block (nn.Module): Dense block or RRDB, modified in place
    """
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, (ResidualDenseBlock_5C_2d, ResidualDenseBlock_5C_3d)):
                module.conv5.weight.zero_()
                assert module.conv5.bias is not None
                module.conv5.bias.zero_()


@pytest.mark.parametrize(("cls", "shape"), DENSE + RRDBS, ids=lambda v: getattr(v, "__name__", ""))
def test_shape_is_preserved(cls: type[nn.Module], shape: tuple[int, ...]) -> None:
    assert cls(shape[1], 4)(_randn(shape)).shape == shape


@pytest.mark.parametrize(("cls", "shape"), DENSE, ids=lambda v: getattr(v, "__name__", ""))
def test_dense_block_is_identity_when_last_conv_is_zero(
    cls: type[nn.Module], shape: tuple[int, ...]
) -> None:
    block = cls(shape[1], 4)
    _silence_last_conv(block)
    x = _randn(shape)
    assert_close(block(x), x)


@pytest.mark.parametrize(("cls", "shape"), RRDBS, ids=lambda v: getattr(v, "__name__", ""))
def test_rrdb_scales_the_stack_by_0_2(cls: type[nn.Module], shape: tuple[int, ...]) -> None:
    block = cls(shape[1], 4)
    _silence_last_conv(block)
    x = _randn(shape)
    # identity dense blocks leave the stack equal to x, so out = 0.2 * x + x
    assert_close(block(x), 1.2 * x)


@pytest.mark.parametrize("growth", [1, 4, 32])
@pytest.mark.parametrize("cls", [ResidualDenseBlock_5C_2d, ResidualDenseBlock_5C_3d])
def test_dense_connections_grow_by_growth_channels(
    cls: type[ResidualDenseBlock_5C_2d] | type[ResidualDenseBlock_5C_3d], growth: int
) -> None:
    block = cls(8, growth)
    convs = [block.conv1, block.conv2, block.conv3, block.conv4, block.conv5]
    assert [c.in_channels for c in convs] == [8 + i * growth for i in range(5)]
    assert [c.out_channels for c in convs] == [growth] * 4 + [8]


@pytest.mark.parametrize("bias", [True, False])
def test_bias_flag(bias: bool) -> None:
    block = ResidualDenseBlock_5C_2d(8, 4, bias=bias)
    assert all((conv.bias is not None) == bias for conv in (block.conv1, block.conv5))
