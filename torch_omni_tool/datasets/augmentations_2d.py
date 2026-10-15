# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 2026

@author: Mateo-drr
"""

import random
from dataclasses import dataclass
from typing import overload

import torch
import torchvision.transforms.functional as tf
from torchvision.transforms import InterpolationMode


@dataclass
class _Patch:
    """
    Location, angle and masks of a circular region to be rotated
    """

    x: int
    y: int
    diameter: int
    ang: float
    circle: torch.Tensor
    inv_circle: torch.Tensor


def _disk_mask(diameter: int) -> torch.Tensor:
    """
    Build a circular mask inscribed in a square of the given side

    Args
        diameter: Side of the square the circle is inscribed in
    Returns
        torch.Tensor: Float mask with shape [diameter, diameter], 1 inside the
            circle and 0 outside
    """
    coords = torch.arange(diameter, dtype=torch.float32) - (diameter - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    radius = (diameter - 1) / 2
    return ((xx**2 + yy**2) <= radius**2).to(torch.float32)


def _rotate_region(
    data: torch.Tensor,
    patch: _Patch,
    interpolation: InterpolationMode,
) -> None:
    """
    Rotate a circular region of a tensor in place, keeping the surrounding ring

    Args
        data: Tensor whose last two dimensions are spatial, mutated in place
        patch: Location, angle and masks of the circular region to rotate
        interpolation: Interpolation used while rotating
    """
    box = (
        Ellipsis,
        slice(patch.x, patch.x + patch.diameter),
        slice(patch.y, patch.y + patch.diameter),
    )
    region = data[box].to(torch.float32)
    rotated = tf.rotate(
        (region * patch.circle).unsqueeze(0), patch.ang, interpolation=interpolation
    ).squeeze(0)
    combined = rotated + region * patch.inv_circle
    if not torch.is_floating_point(data):
        combined = combined.round()
    data[box] = combined.to(data.dtype)


@overload
def random_local_rotation_2d(
    in_tensor: torch.Tensor,
    label: None = None,
    radius: int = 16,
    p: float = 0.5,
) -> torch.Tensor: ...


@overload
def random_local_rotation_2d(
    in_tensor: torch.Tensor,
    label: torch.Tensor,
    radius: int = 16,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def random_local_rotation_2d(
    in_tensor: torch.Tensor,
    label: torch.Tensor | None = None,
    radius: int = 16,
    p: float = 0.5,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate a random circular patch of a batch of 2D images, optionally together with its labels
    Reference: https://www.mdpi.com/2313-433X/9/2/46
    Args
        in_tensor: Batch of images with shape [b, c, h, w]
        label: Optional batch of label masks with shape [b, h, w]. When None only
            the input tensor is augmented
        radius: Radius of the circular area that gets rotated
        p: Probability of applying the augmentation to each sample
    Returns
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Rotated image batch when
            label is None, otherwise the rotated image batch and label batch
    """
    if in_tensor.ndim != 4:
        raise ValueError(
            f"in_tensor must be 4D with shape [b, c, h, w], got {in_tensor.ndim}D"
        )
    if label is not None and label.ndim != 3:
        raise ValueError(
            f"label must be 3D with shape [b, h, w], got {label.ndim}D"
        )

    b, _, h, w = in_tensor.shape
    diameter = radius * 2
    if diameter > h or diameter > w:
        raise ValueError(
            f"radius {radius} gives a patch of side {diameter} larger than the "
            f"spatial size ({h}, {w})"
        )

    label_interp = (
        InterpolationMode.NEAREST
        if label is None or not torch.is_floating_point(label)
        else InterpolationMode.BILINEAR
    )

    for i in range(b):
        if random.random() >= p:
            continue

        # top corner of the square area that hosts the circle
        x = random.randint(0, h - diameter)
        y = random.randint(0, w - diameter)
        ang = random.uniform(20.0, 340.0)

        circmask = _disk_mask(diameter)
        # surroundings of the circle crop, using the rotated mask
        patch = _Patch(
            x=x,
            y=y,
            diameter=diameter,
            ang=ang,
            circle=circmask,
            inv_circle=1.0 - tf.rotate(circmask.unsqueeze(0), ang).squeeze(0),
        )

        if label is not None:
            _rotate_region(label[i], patch, label_interp)
        _rotate_region(in_tensor[i], patch, InterpolationMode.BILINEAR)

    if label is None:
        return in_tensor
    return in_tensor, label
