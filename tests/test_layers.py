"""
Tests for the small layers: Snake, LayerNorm2d, SEBlock and PositionEmbeddingSine
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from torch_omni_tool.activations import Snake
from torch_omni_tool.blocks.SEB import SEBlock
from torch_omni_tool.blocks.ViT import PositionEmbeddingSine
from torch_omni_tool.normalizations import LayerNorm2d


def _randn(*shape: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build a reproducible random tensor
    Args:
        shape (int): Shape of the tensor
        seed (int): Seed of the generator
        dtype (torch.dtype): Data type of the tensor
    Returns:
        torch.Tensor: Random normal tensor
    """
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


# Snake


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (2, 8, 5, 7), (4, 16, 32, 32)])
def test_snake_matches_formula(shape: tuple[int, int, int, int]) -> None:
    snake = Snake(shape[1])
    with torch.no_grad():
        snake.a.copy_(torch.rand(snake.a.shape, generator=torch.Generator().manual_seed(1)) + 0.1)
    x = _randn(*shape)
    a = snake.a
    assert_close(snake(x), x + (1 / (a + snake.eps)) * torch.sin(a * x) ** 2)


def test_snake_initial_frequency_is_one_half() -> None:
    snake = Snake(8)
    assert snake.a.shape == (1, 8, 1, 1)
    assert_close(snake.a, torch.full((1, 8, 1, 1), 0.5))


def test_snake_is_monotonic() -> None:
    x = torch.linspace(-20, 20, 2001).view(1, 1, 1, -1)
    y = Snake(1)(x)
    assert (y.diff(dim=-1) >= -1e-6).all()


def test_snake_learns_its_frequency() -> None:
    snake = Snake(4)
    snake(_randn(2, 4, 3, 3)).sum().backward()
    assert snake.a.grad is not None and snake.a.grad.abs().sum() > 0


# LayerNorm2d


@pytest.mark.parametrize("shape", [(1, 4, 1, 1), (2, 8, 5, 7), (3, 16, 9, 9)])
def test_layer_norm_chan_last_matches_torch(shape: tuple[int, int, int, int]) -> None:
    norm = LayerNorm2d(shape[-1], data_format="chan_last")
    with torch.no_grad():
        norm.weight.copy_(_randn(shape[-1], seed=1))
        norm.bias.copy_(_randn(shape[-1], seed=2))
    x = _randn(*shape)
    expected = F.layer_norm(x, (shape[-1],), norm.weight, norm.bias, norm.eps)
    assert_close(norm(x), expected)


@pytest.mark.parametrize("shape", [(1, 4, 1, 1), (2, 8, 5, 7), (3, 16, 9, 9)])
def test_layer_norm_chan_first_matches_chan_last(shape: tuple[int, int, int, int]) -> None:
    first = LayerNorm2d(shape[1], data_format="chan_first")
    last = LayerNorm2d(shape[1], data_format="chan_last")
    with torch.no_grad():
        first.weight.copy_(_randn(shape[1], seed=1))
        first.bias.copy_(_randn(shape[1], seed=2))
    last.load_state_dict(first.state_dict())
    x = _randn(*shape)
    expected = last(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    assert_close(first(x), expected, rtol=1e-5, atol=1e-5)


def test_layer_norm_normalizes_the_channels_at_init() -> None:
    out = LayerNorm2d(16)(_randn(2, 16, 4, 4) * 5 + 3)
    assert_close(out.mean(dim=1), torch.zeros(2, 4, 4), atol=1e-5, rtol=0)
    assert_close(out.var(dim=1, unbiased=False), torch.ones(2, 4, 4), atol=1e-3, rtol=0)


def test_layer_norm_rejects_unknown_format() -> None:
    with pytest.raises(NotImplementedError):
        LayerNorm2d(4, data_format="nchw")  # type: ignore[arg-type]


# SEBlock


def _se_reference(block: SEBlock, x: torch.Tensor) -> torch.Tensor:
    """
    Squeeze and excite written out with the block's own weights
    Args:
        block (SEBlock): Block whose weights are used
        x (torch.Tensor): Input with shape [B, C, H, W]
    Returns:
        torch.Tensor: Expected output
    """
    first, second = block.excite[0], block.excite[2]
    assert isinstance(first, torch.nn.Linear) and isinstance(second, torch.nn.Linear)
    pooled = x.mean(dim=(2, 3))
    gate = torch.sigmoid(F.linear(torch.relu(F.linear(pooled, first.weight)), second.weight))
    return x * gate[:, :, None, None]


@pytest.mark.parametrize(
    ("num_features", "squeeze_channels"), [(8, 2), (16, 16), (64, 4)], ids=["8-2", "16-16", "64-4"]
)
def test_se_matches_reference(num_features: int, squeeze_channels: int) -> None:
    torch.manual_seed(0)
    block = SEBlock(num_features, squeeze_channels)
    x = _randn(2, num_features, 6, 5)
    assert_close(block(x), _se_reference(block, x))


def test_se_bottleneck_width_is_squeeze_channels() -> None:
    block = SEBlock(32, squeeze_channels=8)
    first, second = block.excite[0], block.excite[2]
    assert isinstance(first, torch.nn.Linear) and isinstance(second, torch.nn.Linear)
    assert first.weight.shape == (8, 32)
    assert second.weight.shape == (32, 8)


def test_se_gate_scales_each_channel_uniformly() -> None:
    x = _randn(2, 8, 6, 6).abs() + 0.1
    ratio = SEBlock(8, 4)(x) / x
    assert_close(ratio, ratio[:, :, :1, :1].expand_as(ratio))
    assert ((ratio > 0) & (ratio < 1)).all()


# PositionEmbeddingSine


def _sine_reference(
    b: int, c: int, h: int, w: int, temperature: float, normalize: bool, scale: float
) -> torch.Tensor:
    """
    DETR style 2D sine embedding written with explicit even / odd channel indexing
    Args:
        b (int): Batch size
        c (int): Number of channels
        h (int): Height
        w (int): Width
        temperature (float): Temperature of the encoding
        normalize (bool): Whether the coordinates are normalized to [0, scale]
        scale (float): Scale of the normalized coordinates
    Returns:
        torch.Tensor: Embedding with shape [b, c, h, w]
    """
    feats = (c + 1) // 2
    ys = torch.arange(1, h + 1, dtype=torch.float32)
    xs = torch.arange(1, w + 1, dtype=torch.float32)
    if normalize:
        ys = ys / (h + 1e-6) * scale
        xs = xs / (w + 1e-6) * scale
    out = torch.empty(2 * feats, h, w)
    for i in range(feats):
        freq = temperature ** (2 * (i // 2) / feats)
        fn = torch.sin if i % 2 == 0 else torch.cos
        out[i] = fn(ys / freq)[:, None].expand(h, w)
        out[feats + i] = fn(xs / freq)[None, :].expand(h, w)
    return out[:c].expand(b, c, h, w)


@pytest.mark.parametrize("channels", [3, 4, 8, 64, 128])
@pytest.mark.parametrize("normalize", [True, False], ids=["normalized", "raw"])
def test_position_embedding_matches_reference(channels: int, normalize: bool) -> None:
    pos = PositionEmbeddingSine(normalize=normalize)(torch.zeros(2, channels, 7, 5))
    expected = _sine_reference(2, channels, 7, 5, 10000, normalize, 2 * math.pi)
    assert_close(pos, expected, rtol=1e-5, atol=1e-5)


def test_position_embedding_only_depends_on_the_shape() -> None:
    module = PositionEmbeddingSine()
    assert_close(module(torch.zeros(1, 8, 4, 4)), module(_randn(1, 8, 4, 4)))


def test_position_embedding_rejects_scale_without_normalize() -> None:
    with pytest.raises(ValueError):
        PositionEmbeddingSine(normalize=False, scale=1.0)


@pytest.mark.xfail(strict=True, reason="known issue: fails when (C + 1) // 2 is odd")
@pytest.mark.parametrize("channels", [6, 10])
def test_position_embedding_odd_feature_count(channels: int) -> None:
    pos = PositionEmbeddingSine()(torch.zeros(1, channels, 4, 4))
    assert pos.shape == (1, channels, 4, 4)
