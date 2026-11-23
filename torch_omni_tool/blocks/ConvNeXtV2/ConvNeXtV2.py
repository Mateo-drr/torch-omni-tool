# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026
@author: Mateo-drr
"""

from typing import Literal

import torch
from torch import nn

from torch_omni_tool.blocks.ConvNeXtV2.GRN import GRN2d
from torch_omni_tool.normalizations.layer_norm_2d import LayerNorm2d


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block, there are two equivalent implementations:
    (1) DwConv -> LayerNorm2d (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm2d (channels_last) -> Linear -> GELU -> Linear;
    Permute back
    This block uses (2) as it is slightly faster in PyTorch
    Version 2 (ConvNeXt V2) adds GRN after the GELU and drops LayerScale, version 1 is the
    original ConvNeXt block with LayerScale and no GRN
    """

    def __init__(
        self,
        num_features: int,
        layer_scale_init_value: float = 1e-6,
        version: Literal[1, 2] = 2,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Build a ConvNeXt block
        Args:
            num_features (int): Number of channels, the output keeps the same count
            layer_scale_init_value (float): Initial value of LayerScale, disabled when not
                positive, only used by version 1
            version (Literal[1, 2]): 2 for the ConvNeXt V2 block (GRN, no LayerScale), 1 for the
                original ConvNeXt block (LayerScale, no GRN)
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        if version not in (1, 2):
            raise ValueError(f"version must be 1 or 2, got {version!r}")
        self.version = version
        c = num_features
        # depthwise conv
        self.dw_conv = nn.Conv2d(
            c, c, kernel_size=7, padding=3, groups=c, device=device, dtype=dtype
        )
        self.norm = LayerNorm2d(c, eps=1e-6, data_format="chan_last", device=device, dtype=dtype)
        # pointwise/1x1 convs, implemented with linear layers
        self.pw_conv1 = nn.Linear(c, 4 * c, device=device, dtype=dtype)
        self.act = nn.GELU()
        # V2 normalizes the expanded features, the input is channel last at this point
        self.grn: GRN2d | None = (
            GRN2d(4 * c, data_format="chan_last", device=device, dtype=dtype)
            if version == 2
            else None
        )
        self.pw_conv2 = nn.Linear(4 * c, c, device=device, dtype=dtype)
        # V2 drops LayerScale
        self.gamma: nn.Parameter | None = (
            nn.Parameter(torch.full((c,), layer_scale_init_value, device=device, dtype=dtype))
            if version == 1 and layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the depthwise convolution, inverted bottleneck and residual connection
        Args:
            x (torch.Tensor): Input tensor with shape [N, C, H, W]
        Returns:
            torch.Tensor: Output tensor with shape [N, C, H, W]
        """
        x0 = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        if self.grn is not None:
            x = self.grn(x)
        x = self.pw_conv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = x0 + x
        return x
