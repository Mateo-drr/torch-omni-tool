"""
Tests for the GRN (Global Response Normalization) layers.
"""

import pytest
import torch
from torch.testing import assert_close

from torch_omni_tool.blocks.ConvNeXtV2 import GRN1d, GRN2d, GRN3d

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 8, 5, 7
EPS = 1e-6


def _reference_grn(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Straightforward re-implementation of the GRN formula for cross-checking.
    """
    gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
    nx = gx / (gx.mean(dim=1, keepdim=True) + EPS)
    out: torch.Tensor = gamma.view(1, -1, 1, 1) * (x * nx) + beta.view(1, -1, 1, 1) + x
    return out


def _make_input(
    batch: int = BATCH,
    channels: int = CHANNELS,
    height: int = HEIGHT,
    width: int = WIDTH,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, channels, height, width, generator=generator, dtype=dtype)


@pytest.fixture
def grn() -> GRN2d:
    return GRN2d(CHANNELS)


@pytest.mark.parametrize("shape", [(1, 3, 1, 1), (2, 8, 5, 7), (4, 16, 32, 32)])
def test_output_shape_matches_input(shape: tuple[int, int, int, int]) -> None:
    out = GRN2d(shape[1])(_make_input(*shape))
    assert out.shape == shape


def test_gamma_and_beta_shapes_and_initialization(grn: GRN2d) -> None:
    names = dict(grn.named_parameters())
    assert set(names) == {"gamma", "beta"}
    assert names["gamma"].shape == (CHANNELS,)
    assert names["beta"].shape == (CHANNELS,)
    assert_close(names["gamma"], torch.zeros_like(names["gamma"]))
    assert_close(names["beta"], torch.zeros_like(names["beta"]))


def test_identity_at_initialization(grn: GRN2d) -> None:
    x = _make_input()
    assert_close(grn(x), x, rtol=0, atol=0)


def test_matches_reference_formula(grn: GRN2d) -> None:
    torch.manual_seed(1)
    with torch.no_grad():
        grn.gamma.normal_()
        grn.beta.normal_()

    x = _make_input()
    assert_close(grn(x), _reference_grn(x, grn.gamma, grn.beta))


def test_normalization_is_over_spatial_dims_then_channels() -> None:
    x = _make_input()
    gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
    nx = gx / (gx.mean(dim=1, keepdim=True) + EPS)

    # gx holds one L2 norm per (batch, channel) and is broadcast over H, W
    assert gx.shape == (BATCH, CHANNELS, 1, 1)
    assert_close(gx, x.pow(2).sum(dim=(2, 3), keepdim=True).sqrt())
    # the normalized response averages to ~1 across the channel dim
    assert_close(nx.mean(dim=1), torch.ones(BATCH, 1, 1), rtol=1e-4, atol=1e-4)


def test_gamma_scales_and_beta_shifts_per_channel() -> None:
    grn = GRN2d(CHANNELS)
    x = _make_input()
    gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
    nx = gx / (gx.mean(dim=1, keepdim=True) + EPS)

    gamma = torch.arange(1, CHANNELS + 1, dtype=torch.float32).view(1, -1, 1, 1)
    beta = torch.full((1, CHANNELS, 1, 1), 0.5)
    with torch.no_grad():
        grn.gamma.copy_(gamma.view(-1))
        grn.beta.copy_(beta.view(-1))

    assert_close(grn(x), x + gamma * (x * nx) + beta)


def test_beta_is_a_pure_additive_shift() -> None:
    grn = GRN2d(CHANNELS)
    x = _make_input()
    with torch.no_grad():
        grn.gamma.normal_()
    before = grn(x)
    with torch.no_grad():
        grn.beta.fill_(0.25)
    assert_close(grn(x), before + 0.25)


def test_positive_homogeneity_when_beta_is_zero() -> None:
    grn = GRN2d(CHANNELS)
    with torch.no_grad():
        grn.gamma.normal_()
    x = _make_input()
    for scale in (0.5, 2.0, 7.0):
        assert_close(grn(x * scale), grn(x) * scale, rtol=1e-5, atol=1e-5)


def test_train_and_eval_modes_agree(grn: GRN2d) -> None:
    x = _make_input()
    grn.train()
    train_out = grn(x)
    grn.eval()
    assert_close(grn(x), train_out, rtol=0, atol=0)


def test_gradients_flow_to_input_and_parameters(grn: GRN2d) -> None:
    with torch.no_grad():
        grn.gamma.normal_()
        grn.beta.normal_()

    x = _make_input().requires_grad_(True)
    grn(x).pow(2).mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    assert grn.gamma.grad is not None
    assert grn.beta.grad is not None
    assert torch.isfinite(grn.gamma.grad).all()
    assert torch.isfinite(grn.beta.grad).all()
    assert grn.gamma.grad.abs().sum() > 0
    assert grn.beta.grad.abs().sum() > 0


def test_input_gradient_matches_autograd_reference(grn: GRN2d) -> None:
    with torch.no_grad():
        grn.gamma.normal_()
        grn.beta.normal_()

    x = _make_input().requires_grad_(True)
    grad_out = torch.randn(
        BATCH, CHANNELS, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(3)
    )

    assert_close(
        torch.autograd.grad(grn(x), x, grad_out)[0],
        torch.autograd.grad(_reference_grn(x, grn.gamma, grn.beta), x, grad_out)[0],
    )


def test_parameter_gradients_match_autograd_reference(grn: GRN2d) -> None:
    with torch.no_grad():
        grn.gamma.normal_()
        grn.beta.normal_()

    x = _make_input()
    grad_out = torch.randn(
        BATCH, CHANNELS, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(4)
    )

    actual = torch.autograd.grad(grn(x), [grn.gamma, grn.beta], grad_out)
    expected = torch.autograd.grad(
        _reference_grn(x, grn.gamma, grn.beta), [grn.gamma, grn.beta], grad_out
    )
    assert_close(actual[0], expected[0])
    assert_close(actual[1], expected[1])


def test_beta_receives_constant_gradient(grn: GRN2d) -> None:
    x = _make_input()
    grn(x).sum().backward()
    # beta is broadcast over H, W, so summing the output gives a gradient of
    # batch * H * W for every channel
    expected = torch.full_like(grn.beta, float(BATCH * HEIGHT * WIDTH))
    assert_close(grn.beta.grad, expected)
    # gamma starts at zero, so d(out)/d(gamma) = x * nx which is generally nonzero
    assert grn.gamma.grad is not None
    assert grn.gamma.grad.abs().sum() > 0


def test_all_zero_input_does_not_produce_nan(grn: GRN2d) -> None:
    with torch.no_grad():
        grn.gamma.fill_(1.0)
    x = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    assert torch.isfinite(grn(x)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(grn: GRN2d, dtype: torch.dtype) -> None:
    grn = GRN2d(CHANNELS).to(dtype)
    x = _make_input(dtype=dtype)
    assert grn(x).dtype == dtype


def test_non_contiguous_input(grn: GRN2d) -> None:
    x = _make_input(batch=BATCH, channels=CHANNELS, height=HEIGHT, width=WIDTH * 2)
    view = x[..., ::2]
    assert not view.is_contiguous()
    assert_close(grn(view), grn(view.contiguous()))


def test_dim_must_match_channels() -> None:
    grn = GRN2d(CHANNELS)
    with pytest.raises(ValueError, match="channels"):
        grn(_make_input(channels=CHANNELS + 1))


def test_rejects_non_4d_input(grn: GRN2d) -> None:
    with pytest.raises(ValueError, match="4D"):
        grn(torch.randn(BATCH, CHANNELS, HEIGHT))


def test_state_dict_roundtrip() -> None:
    grn = GRN2d(CHANNELS)
    with torch.no_grad():
        grn.gamma.normal_()
        grn.beta.normal_()

    clone = GRN2d(CHANNELS)
    clone.load_state_dict(grn.state_dict())
    x = _make_input()
    assert_close(clone(x), grn(x))


# 1d / 3d variants, channel last layout and checkpoint loading

_VARIANTS = [
    (GRN1d, (BATCH, CHANNELS, 11)),
    (GRN2d, (BATCH, CHANNELS, 5, 7)),
    (GRN3d, (BATCH, CHANNELS, 3, 4, 5)),
]


def _reference_grn_nd(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Reference GRN for channel first inputs with any number of spatial dims
    Args:
        x (torch.Tensor): Input with shape [B, C, *spatial]
        gamma (torch.Tensor): Scale with shape [C]
        beta (torch.Tensor): Shift with shape [C]
    Returns:
        torch.Tensor: Expected output
    """
    shape = (1, -1) + (1,) * (x.ndim - 2)
    gx = torch.norm(x, p=2, dim=tuple(range(2, x.ndim)), keepdim=True)
    nx = gx / (gx.mean(dim=1, keepdim=True) + EPS)
    out: torch.Tensor = gamma.view(shape) * (x * nx) + beta.view(shape) + x
    return out


def _randomize(grn: torch.nn.Module) -> None:
    """
    Fill gamma and beta with reproducible random values
    Args:
        grn (torch.nn.Module): GRN layer to modify in place
    """
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for param in grn.parameters():
            param.copy_(torch.randn(param.shape, generator=gen))


@pytest.mark.parametrize(("cls", "shape"), _VARIANTS)
def test_variants_match_reference(cls: type[GRN2d], shape: tuple[int, ...]) -> None:
    grn = cls(CHANNELS)
    _randomize(grn)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(0))
    assert_close(grn(x), _reference_grn_nd(x, grn.gamma, grn.beta))


@pytest.mark.parametrize(("cls", "shape"), _VARIANTS)
def test_chan_last_matches_chan_first(cls: type[GRN2d], shape: tuple[int, ...]) -> None:
    first, last = cls(CHANNELS), cls(CHANNELS, data_format="chan_last")
    _randomize(first)
    last.load_state_dict(first.state_dict())
    x = torch.randn(shape, generator=torch.Generator().manual_seed(0))
    to_last = (0, *range(2, x.ndim), 1)
    to_first = (0, x.ndim - 1, *range(1, x.ndim - 1))
    assert_close(last(x.permute(to_last)).permute(to_first), first(x))


@pytest.mark.parametrize(("cls", "shape"), _VARIANTS)
def test_variants_reject_other_ranks(cls: type[GRN2d], shape: tuple[int, ...]) -> None:
    grn = cls(CHANNELS)
    with pytest.raises(ValueError, match=f"{len(shape)}D"):
        grn(torch.randn(*shape, 2))


def test_chan_last_checks_the_last_dim() -> None:
    grn = GRN2d(CHANNELS, data_format="chan_last")
    with pytest.raises(ValueError, match="channels"):
        grn(torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH))


def test_invalid_data_format_raises() -> None:
    with pytest.raises(NotImplementedError):
        GRN2d(CHANNELS, data_format="nchw")  # type: ignore[arg-type]


@pytest.mark.parametrize("old_shape", [(1, CHANNELS, 1, 1), (1, 1, 1, CHANNELS)])
def test_loads_pre_shaped_checkpoints(old_shape: tuple[int, ...]) -> None:
    gamma = torch.arange(CHANNELS, dtype=torch.float32)
    beta = torch.ones(CHANNELS)
    grn = GRN2d(CHANNELS)
    grn.load_state_dict({"gamma": gamma.view(old_shape), "beta": beta.view(old_shape)})
    assert_close(grn.gamma.data, gamma)
    assert_close(grn.beta.data, beta)
