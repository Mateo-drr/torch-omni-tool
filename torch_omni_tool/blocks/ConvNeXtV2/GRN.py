# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026
@author: Mateo-drr
"""

from typing import Any, Literal

import torch
from torch import nn


class _GRN(nn.Module):
    """
    Global Response Normalization for any number of spatial dims, base of GRN1d, GRN2d and GRN3d
    Subclasses only set the expected input rank, like torch's BatchNorm1d/2d/3d
    """

    _spatial_dims: int

    def __init__(
        self, dim: int, data_format: Literal["chan_first", "chan_last"] = "chan_first"
    ) -> None:
        """
        Build a GRN layer
        Args:
            dim (int): Number of channels, must match the channel dim of the input
            data_format (Literal['chan_first', 'chan_last']): Layout of the input, chan_first for
                [B, C, *spatial] or chan_last for [B, *spatial, C]
        """
        super().__init__()
        if data_format not in ("chan_first", "chan_last"):
            raise NotImplementedError(f"unknown data_format {data_format!r}")
        self.dim = dim
        self.data_format = data_format
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def _check_input(self, x: torch.Tensor) -> None:
        """
        Validate the rank and channel count of the input
        Args:
            x (torch.Tensor): Input tensor
        """
        expected = self._spatial_dims + 2
        if x.ndim != expected:
            raise ValueError(
                f"{type(self).__name__} expects a {expected}D input, got {x.ndim}D {tuple(x.shape)}"
            )
        chan = 1 if self.data_format == "chan_first" else -1
        if x.shape[chan] != self.dim:
            raise ValueError(
                f"{type(self).__name__} was built for {self.dim} channels, got {x.shape[chan]} "
                f"in input {tuple(x.shape)} ({self.data_format})"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the spatial L2 norm of every channel across channels and rescale the input
        Args:
            x (torch.Tensor): Input tensor with shape [B, C, *spatial] or [B, *spatial, C]
                depending on data_format
        Returns:
            torch.Tensor: Output tensor with the same shape as the input
        """
        self._check_input(x)
        if self.data_format == "chan_first":
            chan = 1
            spatial = tuple(range(2, x.ndim))
        else:
            chan = x.ndim - 1
            spatial = tuple(range(1, x.ndim - 1))
        # broadcast the [C] parameters along the channel axis
        shape = [1] * x.ndim
        shape[chan] = self.dim
        gamma, beta = self.gamma.view(shape), self.beta.view(shape)

        gx = torch.norm(x, p=2, dim=spatial, keepdim=True)  # norm over the spatial dims
        nx = gx / (gx.mean(dim=chan, keepdim=True) + 1e-6)  # normalize over C
        y: torch.Tensor = gamma * (x * nx) + beta + x
        return y

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """
        Accept checkpoints that store gamma and beta pre-shaped, e.g. [1, C, 1, 1] from older
        versions of this layer or [1, 1, 1, C] from the official ConvNeXt V2 code
        Args:
            state_dict (dict[str, Any]): State dict being loaded
            prefix (str): Prefix of this module's keys
            local_metadata (dict[str, Any]): Metadata of this module
            strict (bool): Whether missing or unexpected keys are errors
            missing_keys (list[str]): Collects missing keys
            unexpected_keys (list[str]): Collects unexpected keys
            error_msgs (list[str]): Collects error messages
        """
        for name in ("gamma", "beta"):
            key = prefix + name
            value = state_dict.get(key)
            if isinstance(value, torch.Tensor) and value.ndim > 1 and value.numel() == self.dim:
                state_dict[key] = value.reshape(self.dim)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class GRN1d(_GRN):
    """
    GRN for [B, C, L] (chan_first) or [B, L, C] (chan_last) inputs
    """

    _spatial_dims = 1


class GRN2d(_GRN):
    """
    GRN for [B, C, H, W] (chan_first) or [B, H, W, C] (chan_last) inputs
    """

    _spatial_dims = 2


class GRN3d(_GRN):
    """
    GRN for [B, C, D, H, W] (chan_first) or [B, D, H, W, C] (chan_last) inputs
    """

    _spatial_dims = 3
