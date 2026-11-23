# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026
@author: Mateo-drr
"""

import torch
from torch import nn


class SEBlock(nn.Module):
    """
    Squeeze and Excite block
    """

    def __init__(
        self,
        num_features: int,
        squeeze_channels: int = 16,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Build a squeeze and excite block
        Args:
            num_features (int): Number of channels, the output keeps the same count
            squeeze_channels (int): Number of channels of the bottleneck inside the excitation
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(num_features, squeeze_channels, bias=False, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_channels, num_features, bias=False, device=device, dtype=dtype),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool the spatial dims, compute the channel gate and rescale the input
        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W]
        Returns:
            torch.Tensor: Gated tensor with shape [B, C, H, W]
        """
        b, c, _, _ = x.size()
        y: torch.Tensor = self.squeeze(x).view(b, c)  # Squeeze: [b,c,h,w] → [b,c,1,1] → [b,c]
        y = self.excite(y)  # Excite: [b,c] → [b,r] → [b,c]
        return x * y.view(b, c, 1, 1)  # Scale: [b,c] → [b,c,1,1] then multiply
