# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:41:06 2024
@author: Mateo-drr
"""

import torch
import torch.nn as nn


class ResidualDenseBlock_5C_2d(nn.Module):
    """
    Residual dense block with 5 convolutions and dense connections
    """

    def __init__(
        self,
        num_features: int,
        growth_channels: int = 32,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Build a 2d residual dense block
        Args:
            num_features (int): Number of channels, the output keeps the same count
            growth_channels (int): Number of channels added by each of the first four convolutions
            bias (bool): Whether the convolutions use a bias term
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        c, g = num_features, growth_channels
        self.conv1 = nn.Conv2d(
            c, g, kernel_size=3, stride=1, padding=1, bias=bias, device=device, dtype=dtype
        )
        self.conv2 = nn.Conv2d(
            c + g, g, kernel_size=3, stride=1, padding=1, bias=bias, device=device, dtype=dtype
        )
        self.conv3 = nn.Conv2d(
            c + 2 * g, g, kernel_size=3, stride=1, padding=1, bias=bias, device=device, dtype=dtype
        )
        self.conv4 = nn.Conv2d(
            c + 3 * g, g, kernel_size=3, stride=1, padding=1, bias=bias, device=device, dtype=dtype
        )
        self.conv5 = nn.Conv2d(
            c + 4 * g, c, kernel_size=3, stride=1, padding=1, bias=bias, device=device, dtype=dtype
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolve the input with every convolution and merge the outputs densely
        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W]
        Returns:
            torch.Tensor: Scaled output tensor with the same shape as the input
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5: torch.Tensor = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB2d(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(
        self,
        num_features: int,
        growth_channels: int = 32,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Build a 2d residual in residual dense block
        Args:
            num_features (int): Number of channels, the output keeps the same count
            growth_channels (int): Growth channels of every residual dense block
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C_2d(
            num_features, growth_channels, device=device, dtype=dtype
        )
        self.RDB2 = ResidualDenseBlock_5C_2d(
            num_features, growth_channels, device=device, dtype=dtype
        )
        self.RDB3 = ResidualDenseBlock_5C_2d(
            num_features, growth_channels, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the three stacked residual dense blocks
        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W]
        Returns:
            torch.Tensor: Scaled output tensor with the same shape as the input
        """
        out: torch.Tensor = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
