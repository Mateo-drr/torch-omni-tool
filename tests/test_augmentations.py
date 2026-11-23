"""
Tests for the local rotation augmentations in ``torch_omni_tool.datasets.RandomLocalRotation``.
"""

import random
from collections.abc import Callable
from typing import Any, NamedTuple

import pytest
import torch
from torch.testing import assert_close

from torch_omni_tool.datasets.RandomLocalRotation import (
    random_local_rotation_2d,
    random_local_rotation_3d,
)

BATCH, CHANNELS, SIZE, RADIUS = 3, 2, 32, 8


def _images(
    batch: int = BATCH,
    channels: int = CHANNELS,
    size: int = SIZE,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a reproducible batch of images
    Args:
        batch (int): Batch size
        channels (int): Number of channels
        size (int): Height and width
        seed (int): Seed of the generator
        dtype (torch.dtype): Data type
    Returns:
        torch.Tensor: Uniform random images with shape [batch, channels, size, size]
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, channels, size, size, generator=generator, dtype=dtype)


class _Variant(NamedTuple):
    """
    How to call and feed one of the rotation functions
    """

    rotate: Callable[..., Any]
    make_input: Callable[..., torch.Tensor]
    label_shape: tuple[int, ...]
    plane: Callable[[torch.Tensor], torch.Tensor]


VARIANTS = [
    _Variant(random_local_rotation_2d, _images, (BATCH, SIZE, SIZE), lambda x: x[:, 0]),
    _Variant(
        random_local_rotation_3d,
        lambda seed=0, dtype=torch.float32: _images(1, 1, seed=seed, dtype=dtype),
        (SIZE, SIZE),
        lambda x: x[0, 0],
    ),
]
IDS = ["2d", "3d"]


def _int_label(variant: _Variant) -> torch.Tensor:
    """
    Build a random integer label for a variant
    Args:
        variant (_Variant): Rotation variant
    Returns:
        torch.Tensor: Integer label in [0, 4)
    """
    return torch.randint(0, 4, variant.label_shape, generator=torch.Generator().manual_seed(9))


# shared behaviour of the 2d and 3d versions


@pytest.mark.parametrize("variant", VARIANTS, ids=IDS)
def test_no_label_returns_a_tensor(variant: _Variant) -> None:
    x = variant.make_input()
    out = variant.rotate(x.clone(), p=1.0)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


@pytest.mark.parametrize("variant", VARIANTS, ids=IDS)
def test_with_label_returns_a_tuple(variant: _Variant) -> None:
    x, label = variant.make_input(), _int_label(variant)
    out, lbl = variant.rotate(x, label, p=1.0)
    assert out.shape == x.shape
    assert lbl.shape == label.shape


@pytest.mark.parametrize("variant", VARIANTS, ids=IDS)
def test_default_label_is_none(variant: _Variant) -> None:
    x = variant.make_input()
    random.seed(0)
    default = variant.rotate(x.clone(), p=1.0)
    random.seed(0)
    explicit = variant.rotate(x.clone(), None, p=1.0)
    assert_close(default, explicit)


@pytest.mark.parametrize("variant", VARIANTS, ids=IDS)
def test_label_does_not_change_the_input_rotation(variant: _Variant) -> None:
    # rotating a label consumes no randomness, so the input is augmented identically
    x = variant.make_input()
    random.seed(1)
    without = variant.rotate(x.clone(), p=1.0)
    random.seed(1)
    with_label, _ = variant.rotate(x.clone(), _int_label(variant), p=1.0)
    assert_close(without, with_label)


@pytest.mark.parametrize("variant", VARIANTS, ids=IDS)
def test_modifies_the_input_in_place(variant: _Variant) -> None:
    x = variant.make_input()
    assert variant.rotate(x, p=1.0) is x


@pytest.mark.parametrize("variant", VARIANTS, ids=IDS)
def test_p_zero_is_a_noop(variant: _Variant) -> None:
    x = variant.make_input()
    label = torch.rand(variant.label_shape)
    before_x, before_label = x.clone(), label.clone()
    out, lbl = variant.rotate(x, label, radius=RADIUS, p=0.0)
    assert_close(out, before_x, rtol=0, atol=0)
    assert_close(lbl, before_label, rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS, ids=IDS)
def test_p_one_changes_the_input(variant: _Variant) -> None:
    x = variant.make_input(seed=7)
    assert not torch.equal(variant.rotate(x.clone(), radius=RADIUS, p=1.0), x)


@pytest.mark.parametrize("variant", VARIANTS, ids=IDS)
def test_float_label_rotates_together_with_the_input(variant: _Variant) -> None:
    # a float label equal to one input plane is rotated with the same mask and angle
    x = variant.make_input(seed=3)
    label = variant.plane(x).clone()
    out, lbl = variant.rotate(x, label, radius=RADIUS, p=1.0)
    assert_close(variant.plane(out), lbl)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", VARIANTS, ids=IDS)
def test_input_dtype_is_preserved(variant: _Variant, dtype: torch.dtype) -> None:
    out = variant.rotate(variant.make_input(dtype=dtype), radius=RADIUS, p=1.0)
    assert out.dtype == dtype


# 2d only: batched labels, per-sample draws and validation


def test_2d_integer_label_stays_integer_and_in_range() -> None:
    labels = torch.randint(0, 5, (BATCH, SIZE, SIZE))
    _, lbl = random_local_rotation_2d(_images(), labels.clone(), radius=RADIUS, p=1.0)
    assert lbl.dtype == labels.dtype
    assert lbl.min() >= labels.min()
    assert lbl.max() <= labels.max()


def test_2d_float_label_dtype_is_preserved() -> None:
    labels = torch.rand(BATCH, SIZE, SIZE, dtype=torch.float64)
    _, lbl = random_local_rotation_2d(_images(), labels, radius=RADIUS, p=1.0)
    assert lbl.dtype == torch.float64


def test_2d_augments_every_sample_at_p_one() -> None:
    images = _images(batch=4, seed=5)
    out = random_local_rotation_2d(images.clone(), radius=RADIUS, p=1.0)
    assert all(not torch.equal(out[i], images[i]) for i in range(4))


@pytest.mark.parametrize(
    ("images", "label", "radius"),
    [
        (torch.rand(BATCH, SIZE, SIZE), None, RADIUS),
        (_images(), torch.rand(BATCH, CHANNELS, SIZE, SIZE), RADIUS),
        (_images(size=16), None, 9),
    ],
    ids=["3d-input", "4d-label", "radius-too-large"],
)
def test_2d_rejects_invalid_inputs(
    images: torch.Tensor, label: torch.Tensor | None, radius: int
) -> None:
    with pytest.raises(ValueError):
        random_local_rotation_2d(images, label, radius=radius, p=1.0)


# 3d only: one patch for the whole chunk


def test_3d_rotates_every_sample_with_the_same_patch() -> None:
    chunk = _images(1, 1).expand(3, 1, SIZE, SIZE).clone()
    out = random_local_rotation_3d(chunk, radius=RADIUS, p=1.0)
    assert_close(out[1], out[0], rtol=0, atol=0)
    assert_close(out[2], out[0], rtol=0, atol=0)
