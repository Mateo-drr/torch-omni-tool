"""Tests for the local rotation augmentations in ``torch_omni_tool.datasets``."""

import random

import pytest
import torch
from torch.testing import assert_close

from torch_omni_tool.datasets import (
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
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, channels, size, size, generator=generator, dtype=dtype)


def _chunk(size: int = SIZE, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return _images(batch=1, channels=1, size=size, seed=seed, dtype=dtype)


# --------------------------------------------------------------------------- #
# random_local_rotation_2d
# --------------------------------------------------------------------------- #


def test_2d_no_label_returns_tensor() -> None:
    out = random_local_rotation_2d(_images(), p=1.0)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (BATCH, CHANNELS, SIZE, SIZE)


def test_2d_with_label_returns_tuple() -> None:
    images = _images()
    labels = torch.randint(0, 4, (BATCH, SIZE, SIZE))
    out, lbl = random_local_rotation_2d(images, labels, p=1.0)
    assert out.shape == images.shape
    assert lbl.shape == labels.shape


def test_2d_default_label_is_none() -> None:
    images = _images()
    random.seed(0)
    default = random_local_rotation_2d(images.clone(), p=1.0)
    random.seed(0)
    explicit = random_local_rotation_2d(images.clone(), label=None, p=1.0)
    assert_close(default, explicit)


def test_2d_without_and_with_label_rotate_input_identically() -> None:
    # Rotating a label consumes no randomness, so the input augmentation must be
    # identical whether or not a label is passed.
    images = _images()
    labels = torch.randint(0, 4, (BATCH, SIZE, SIZE))

    random.seed(1)
    without = random_local_rotation_2d(images.clone(), p=1.0)
    random.seed(1)
    with_label, _ = random_local_rotation_2d(images.clone(), labels, p=1.0)

    assert_close(without, with_label)


def test_2d_modifies_input_in_place_and_returns_same_object() -> None:
    images = _images()
    out = random_local_rotation_2d(images, p=1.0)
    assert out is images


def test_2d_p_zero_is_a_noop() -> None:
    images = _images()
    labels = torch.rand(BATCH, SIZE, SIZE)
    before_images = images.clone()
    before_labels = labels.clone()

    out, lbl = random_local_rotation_2d(images, labels, radius=RADIUS, p=0.0)

    assert_close(out, before_images, rtol=0, atol=0)
    assert_close(lbl, before_labels, rtol=0, atol=0)


def test_2d_p_one_changes_the_input() -> None:
    images = _images(seed=7)
    out = random_local_rotation_2d(images.clone(), radius=RADIUS, p=1.0)
    assert not torch.equal(out, images)


def test_2d_float_label_rotates_together_with_input() -> None:
    # When the label is float it is rotated with the same mask, angle and
    # bilinear interpolation as the input, so a shared plane stays equal.
    images = _images(batch=2, channels=1, seed=3)
    labels = images[:, 0].clone()

    out, lbl = random_local_rotation_2d(images, labels, radius=RADIUS, p=1.0)

    assert_close(out[:, 0], lbl)


def test_2d_integer_label_stays_integer_and_in_range() -> None:
    images = _images()
    labels = torch.randint(0, 5, (BATCH, SIZE, SIZE))
    _, lbl = random_local_rotation_2d(images, labels.clone(), radius=RADIUS, p=1.0)

    assert lbl.dtype == labels.dtype
    assert lbl.min() >= labels.min()
    assert lbl.max() <= labels.max()


def test_2d_float_label_dtype_preserved() -> None:
    images = _images()
    labels = torch.rand(BATCH, SIZE, SIZE, dtype=torch.float64)
    _, lbl = random_local_rotation_2d(images, labels, radius=RADIUS, p=1.0)
    assert lbl.dtype == torch.float64


def test_2d_all_batch_samples_augmented_at_p_one() -> None:
    images = _images(batch=4, seed=5)
    out = random_local_rotation_2d(images.clone(), radius=RADIUS, p=1.0)
    for i in range(4):
        assert not torch.equal(out[i], images[i])


def test_2d_rejects_non_4d_input() -> None:
    with pytest.raises(ValueError):
        random_local_rotation_2d(torch.rand(BATCH, SIZE, SIZE), p=1.0)


def test_2d_rejects_non_3d_label() -> None:
    with pytest.raises(ValueError):
        random_local_rotation_2d(_images(), torch.rand(BATCH, CHANNELS, SIZE, SIZE), p=1.0)


def test_2d_rejects_radius_larger_than_spatial_size() -> None:
    with pytest.raises(ValueError):
        random_local_rotation_2d(_images(size=16), radius=9, p=1.0)


# --------------------------------------------------------------------------- #
# random_local_rotation_3d
# --------------------------------------------------------------------------- #


def test_3d_no_label_returns_tensor() -> None:
    out = random_local_rotation_3d(_chunk(), p=1.0)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1, SIZE, SIZE)


def test_3d_with_label_returns_tuple() -> None:
    chunk = _chunk()
    label = torch.randint(0, 4, (SIZE, SIZE))
    out, lbl = random_local_rotation_3d(chunk, label, p=1.0)
    assert out.shape == chunk.shape
    assert lbl.shape == label.shape


def test_3d_default_label_is_none() -> None:
    chunk = _chunk()
    random.seed(0)
    default = random_local_rotation_3d(chunk.clone(), p=1.0)
    random.seed(0)
    explicit = random_local_rotation_3d(chunk.clone(), tlbl=None, p=1.0)
    assert_close(default, explicit)


def test_3d_without_and_with_label_rotate_input_identically() -> None:
    chunk = _chunk()
    label = torch.randint(0, 4, (SIZE, SIZE))

    random.seed(1)
    without = random_local_rotation_3d(chunk.clone(), p=1.0)
    random.seed(1)
    with_label, _ = random_local_rotation_3d(chunk.clone(), label, p=1.0)

    assert_close(without, with_label)


def test_3d_modifies_input_in_place_and_returns_same_object() -> None:
    chunk = _chunk()
    out = random_local_rotation_3d(chunk, p=1.0)
    assert out is chunk


def test_3d_p_zero_is_a_noop() -> None:
    chunk = _chunk()
    label = torch.rand(SIZE, SIZE)
    before_chunk = chunk.clone()
    before_label = label.clone()

    out, lbl = random_local_rotation_3d(chunk, label, radius=RADIUS, p=0.0)

    assert_close(out, before_chunk, rtol=0, atol=0)
    assert_close(lbl, before_label, rtol=0, atol=0)


def test_3d_p_one_changes_the_input() -> None:
    chunk = _chunk(seed=7)
    out = random_local_rotation_3d(chunk.clone(), radius=RADIUS, p=1.0)
    assert not torch.equal(out, chunk)


def test_3d_label_rotates_together_with_input() -> None:
    chunk = _chunk(seed=3)
    label = chunk[0, 0].clone()

    out, lbl = random_local_rotation_3d(chunk, label, radius=RADIUS, p=1.0)

    assert_close(out[0, 0], lbl)


def test_3d_float_dtype_preserved() -> None:
    chunk = _chunk(dtype=torch.float32)
    out = random_local_rotation_3d(chunk, p=1.0)
    assert out.dtype == torch.float32
