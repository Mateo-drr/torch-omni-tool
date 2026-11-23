# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026
@author: Mateo-drr
"""

import random
from typing import overload

import numpy as np
import torch
import torchvision.transforms.functional as tf

from PIL import Image, ImageDraw


@overload
def random_local_rotation_3d(
    in_tensor: torch.Tensor,
    tlbl: None = None,
    radius: int = 16,
    p: float = 0.5,
) -> torch.Tensor: ...


@overload
def random_local_rotation_3d(
    in_tensor: torch.Tensor,
    tlbl: torch.Tensor,
    radius: int = 16,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def random_local_rotation_3d(
    in_tensor: torch.Tensor,
    tlbl: torch.Tensor | None = None,
    radius: int = 16,
    p: float = 0.5,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate a random circular patch of the input, optionally together with its label
    Reference: https://www.mdpi.com/2313-433X/9/2/46
    Args:
        in_tensor (torch.Tensor): Input chunk with shape [b, 1, h, w]
        tlbl (torch.Tensor | None): Optional label mask with shape [h, w]. When None only the input
            chunk is augmented
        radius (int): Radius of the circular area that gets rotated
        p (float): Probability of applying the augmentation
    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Rotated chunk when tlbl is None, otherwise
            the rotated chunk and rotated label mask
    """
    if random.random() < p:
        diameter = radius * 2
        if tlbl is None:
            x, y = in_tensor.shape[-2:]
        else:
            x, y = tlbl.shape
        # get a random point to rotate
        # this point is the top corner of the square area to be rotated
        xcenter = random.randint(
            0, (x - diameter)
        )  # random.choice(idk)#np.random.randint(0,32)#random.randint(0,x-diameter)
        ycenter = random.randint(
            0, (y - diameter)
        )  # random.choice(idk)#np.random.randint(0,32)#random.randint(0,y-diameter)
        ang = random.randint(20, 340)

        # Create the circle mask
        lum_img = Image.new(
            "L", [diameter, diameter], 0
        )  # create a square image size [diameter,diameter]
        draw = ImageDraw.Draw(lum_img)
        draw.pieslice(
            [(0, 0), (diameter - 1, diameter - 1)], 0, 360, fill=255
        )  # draw the circle in the image
        circmaks = torch.tensor(np.array(lum_img) / 255)
        # get surroundings of the circle crop but with the rotated mask
        invcircmask = (tf.rotate(circmaks.unsqueeze(0), ang) - 1) * -1

        if tlbl is not None:
            # ROTATE LABEL
            # place the circle in the image
            circCrop = tlbl[xcenter : xcenter + diameter, ycenter : ycenter + diameter] * circmaks
            # rotate circular crop
            circCrop = tf.rotate(circCrop.unsqueeze(0), ang)
            # get surrounding crop
            invCircCrop = (
                tlbl[xcenter : xcenter + diameter, ycenter : ycenter + diameter] * invcircmask
            )
            # place the rotated circle back
            tlbl[xcenter : xcenter + diameter, ycenter : ycenter + diameter] = (
                circCrop + invCircCrop
            )

        # ROTATE CHUNK
        # place the circle in the image
        circCrop = (
            in_tensor[:, :, xcenter : xcenter + diameter, ycenter : ycenter + diameter] * circmaks
        )
        # rotate circular crop
        circCrop = tf.rotate(circCrop, ang)
        # get surrounding crop
        invCircCrop = (
            in_tensor[:, :, xcenter : xcenter + diameter, ycenter : ycenter + diameter]
            * invcircmask
        )
        # place the rotated circle back
        in_tensor[:, :, xcenter : xcenter + diameter, ycenter : ycenter + diameter] = (
            circCrop + invCircCrop
        )

    if tlbl is None:
        return in_tensor
    return in_tensor, tlbl
