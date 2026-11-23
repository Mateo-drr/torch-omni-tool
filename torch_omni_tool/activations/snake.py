# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 22:38:08 2026
@author: Mateo-drr
"""

import torch
from torch import nn


class Snake(nn.Module):
    """
    Sine based periodic activation function
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Build a snake activation
        Args:
            num_features (int): Number of channels of the input, shapes the learnable frequency
            eps (float): Value added to the frequency to avoid dividing by zero
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.a = nn.Parameter(torch.full((1, num_features, 1, 1), 0.5, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add a learnable periodic term to the input
        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W]
        Returns:
            torch.Tensor: Output tensor with shape [B, C, H, W]
        """
        x = x + (1 / (self.a + self.eps)) * torch.sin(self.a * x) ** 2
        return x
