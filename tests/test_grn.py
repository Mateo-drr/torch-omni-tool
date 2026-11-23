"""
Tests for the GRN (Global Response Normalization) layers.
"""

import pytest
import torch
from torch.testing import assert_close

from torch_omni_tool.blocks.ConvNeXtV2 import GRN1d, GRN2d, GRN3d

BATCH, CHANNELS = 2, 8
EPS = 1e-6
GRNCls = type[GRN1d] | type[GRN2d] | type[GRN3d]
VARIANTS: list[tuple[GRNCls, tuple[int, ...]]] = [
    (GRN1d, (BATCH, CHANNELS, 11)),
    (GRN2d, (BATCH, CHANNELS, 5, 7)),
    (GRN3d, (BATCH, CHANNELS, 3, 4, 5)),
]
IDS = ["1d", "2d", "3d"]


def _randn(
    shape: tuple[int, ...], seed: int = 0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Build a reproducible random tensor
    Args:
        shape (tuple[int, ...]): Shape of the tensor
        seed (int): Seed of the generator
        dtype (torch.dtype): Data type
    Returns:
        torch.Tensor: Random normal tensor
    """
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _reference(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    GRN written out for channel first inputs with any number of spatial dims
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


def _randomized(cls: GRNCls, data_format: str = "chan_first") -> GRN1d | GRN2d | GRN3d:
    """
    Build a GRN layer with reproducible random gamma and beta
    Args:
        cls (GRNCls): GRN class
        data_format (str): Layout of the input
    Returns:
        GRN1d | GRN2d | GRN3d: Layer with random parameters
    """
    grn = cls(CHANNELS, data_format=data_format)  # type: ignore[arg-type]
    with torch.no_grad():
        grn.gamma.copy_(_randn((CHANNELS,), seed=1))
        grn.beta.copy_(_randn((CHANNELS,), seed=2))
    return grn


# formula


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_matches_reference(cls: GRNCls, shape: tuple[int, ...]) -> None:
    grn = _randomized(cls)
    x = _randn(shape)
    assert_close(grn(x), _reference(x, grn.gamma, grn.beta))


@pytest.mark.parametrize(
    "shape", [(1, 3, 1, 1), (2, 8, 5, 7), (4, 16, 32, 32)], ids=["1x1", "5x7", "32x32"]
)
def test_2d_output_shape_matches_input(shape: tuple[int, int, int, int]) -> None:
    assert GRN2d(shape[1])(_randn(shape)).shape == shape


def test_gamma_scales_and_beta_shifts_per_channel() -> None:
    grn = GRN2d(CHANNELS)
    x = _randn((BATCH, CHANNELS, 5, 7))
    gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
    nx = gx / (gx.mean(dim=1, keepdim=True) + EPS)
    gamma = torch.arange(1, CHANNELS + 1, dtype=torch.float32).view(1, -1, 1, 1)
    beta = torch.full((1, CHANNELS, 1, 1), 0.5)
    with torch.no_grad():
        grn.gamma.copy_(gamma.view(-1))
        grn.beta.copy_(beta.view(-1))
    assert_close(grn(x), x + gamma * (x * nx) + beta)


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_beta_is_a_pure_additive_shift(cls: GRNCls, shape: tuple[int, ...]) -> None:
    grn = _randomized(cls)
    x = _randn(shape)
    with torch.no_grad():
        grn.beta.zero_()
    before = grn(x)
    with torch.no_grad():
        grn.beta.fill_(0.25)
    assert_close(grn(x), before + 0.25)


@pytest.mark.parametrize("scale", [0.5, 2.0, 7.0])
@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_positive_homogeneity_when_beta_is_zero(
    cls: GRNCls, shape: tuple[int, ...], scale: float
) -> None:
    grn = _randomized(cls)
    with torch.no_grad():
        grn.beta.zero_()
    x = _randn(shape)
    assert_close(grn(x * scale), grn(x) * scale, rtol=1e-5, atol=1e-5)


# initialization and modes


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_parameters_are_zero_vectors_at_init(cls: GRNCls, shape: tuple[int, ...]) -> None:
    params = dict(cls(CHANNELS).named_parameters())
    assert set(params) == {"gamma", "beta"}
    for value in params.values():
        assert value.shape == (CHANNELS,)
        assert_close(value.data, torch.zeros(CHANNELS))


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_identity_at_init(cls: GRNCls, shape: tuple[int, ...]) -> None:
    x = _randn(shape)
    assert_close(cls(CHANNELS)(x), x, rtol=0, atol=0)


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_train_and_eval_agree(cls: GRNCls, shape: tuple[int, ...]) -> None:
    grn = _randomized(cls)
    x = _randn(shape)
    train_out = grn.train()(x)
    assert_close(grn.eval()(x), train_out, rtol=0, atol=0)


# autograd


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_gradients_match_reference(cls: GRNCls, shape: tuple[int, ...]) -> None:
    grn = _randomized(cls)
    x = _randn(shape).requires_grad_(True)
    grad_out = _randn(shape, seed=3)
    inputs = [x, grn.gamma, grn.beta]
    actual = torch.autograd.grad(grn(x), inputs, grad_out)
    expected = torch.autograd.grad(_reference(x, grn.gamma, grn.beta), inputs, grad_out)
    for a, e in zip(actual, expected):
        assert torch.isfinite(a).all() and a.abs().sum() > 0
        assert_close(a, e)


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_beta_gradient_counts_the_broadcast_positions(cls: GRNCls, shape: tuple[int, ...]) -> None:
    grn = cls(CHANNELS)
    grn(_randn(shape)).sum().backward()
    positions = shape[0] * int(torch.tensor(shape[2:]).prod())
    assert_close(grn.beta.grad, torch.full((CHANNELS,), float(positions)))
    # gamma starts at zero, so d(out)/d(gamma) = x * nx is generally nonzero
    assert grn.gamma.grad is not None and grn.gamma.grad.abs().sum() > 0


# inputs


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_all_zero_input_stays_finite(cls: GRNCls, shape: tuple[int, ...]) -> None:
    grn = _randomized(cls)
    assert torch.isfinite(grn(torch.zeros(shape))).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_dtype_is_preserved(cls: GRNCls, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    grn = cls(CHANNELS, dtype=dtype)
    assert grn(_randn(shape, dtype=dtype)).dtype == dtype


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_non_contiguous_input(cls: GRNCls, shape: tuple[int, ...]) -> None:
    grn = _randomized(cls)
    view = _randn((*shape[:-1], shape[-1] * 2))[..., ::2]
    assert not view.is_contiguous()
    assert_close(grn(view), grn(view.contiguous()))


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_rejects_a_wrong_channel_count(cls: GRNCls, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="channels"):
        cls(CHANNELS)(_randn((shape[0], CHANNELS + 1, *shape[2:])))


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_rejects_other_ranks(cls: GRNCls, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=f"{len(shape)}D"):
        cls(CHANNELS)(_randn((*shape, 2)))


# layouts and checkpoints


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_chan_last_matches_chan_first(cls: GRNCls, shape: tuple[int, ...]) -> None:
    first = _randomized(cls)
    last = cls(CHANNELS, data_format="chan_last")
    last.load_state_dict(first.state_dict())
    x = _randn(shape)
    to_last = (0, *range(2, x.ndim), 1)
    to_first = (0, x.ndim - 1, *range(1, x.ndim - 1))
    assert_close(last(x.permute(to_last)).permute(to_first), first(x))


def test_chan_last_checks_the_last_dim() -> None:
    with pytest.raises(ValueError, match="channels"):
        GRN2d(CHANNELS, data_format="chan_last")(_randn((BATCH, CHANNELS, 5, 7)))


def test_invalid_data_format_raises() -> None:
    with pytest.raises(NotImplementedError):
        GRN2d(CHANNELS, data_format="nchw")  # type: ignore[arg-type]


@pytest.mark.parametrize(("cls", "shape"), VARIANTS, ids=IDS)
def test_state_dict_round_trip(cls: GRNCls, shape: tuple[int, ...]) -> None:
    grn = _randomized(cls)
    clone = cls(CHANNELS)
    clone.load_state_dict(grn.state_dict())
    x = _randn(shape)
    assert_close(clone(x), grn(x))


@pytest.mark.parametrize(
    "old_shape", [(1, CHANNELS, 1, 1), (1, 1, 1, CHANNELS)], ids=["chan-first", "upstream"]
)
def test_loads_pre_shaped_checkpoints(old_shape: tuple[int, ...]) -> None:
    gamma, beta = torch.arange(CHANNELS, dtype=torch.float32), torch.ones(CHANNELS)
    grn = GRN2d(CHANNELS)
    grn.load_state_dict({"gamma": gamma.view(old_shape), "beta": beta.view(old_shape)})
    assert_close(grn.gamma.data, gamma)
    assert_close(grn.beta.data, beta)
