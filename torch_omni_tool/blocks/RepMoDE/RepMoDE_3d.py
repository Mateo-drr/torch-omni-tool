# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026
@author: Mateo-drr
"""

from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
import math


def one_hot_task_embedding(self: Any, task_id: torch.Tensor) -> torch.Tensor:
    """
    Turn task ids into one hot encodings
    Args:
        task_id (torch.Tensor): Task id of every sample with shape [B]
    Returns:
        torch.Tensor: One hot task encoding with shape [B, num_tasks]
    """
    n_samples = task_id.shape[0]
    task_embedding = torch.zeros((n_samples, self.num_tasks))
    for i in range(n_samples):
        task_embedding[i, task_id[i]] = 1
    return task_embedding.to(self.device)


class MoDEEncoderBlock3d(torch.nn.Module):
    """
    Encoder block that increases the channels and halves the spatial dims
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_tasks: int,
        num_experts: int = 5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Build a MoDE encoder block
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_tasks (int): Number of tasks the gate can select from
            num_experts (int): Number of experts, only 5 is supported
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_more = MoDESubNet2Conv3d(
            in_channels, out_channels, num_tasks, num_experts, device=device, dtype=dtype
        )
        self.conv_down = torch.nn.Sequential(
            torch.nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            nn.BatchNorm3d(out_channels, affine=True, device=device, dtype=dtype),
            torch.nn.Mish(inplace=True),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convolve the input and downsample it by two
        Args:
            x (torch.Tensor): Input tensor with shape [B, in_channels, D, H, W]
            t (torch.Tensor): One hot task encoding with shape [B, num_tasks]
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Downsampled tensor and skip tensor with out_channels
                channels
        """
        x_skip = self.conv_more(x, t)
        x = self.conv_down(x_skip)
        return x, x_skip


class MoDEDecoderBlock3d(torch.nn.Module):
    """
    Decoder block that halves the channels and doubles the spatial dims
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_tasks: int,
        num_experts: int = 5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Build a MoDE decoder block
        Args:
            in_channels (int): Number of input channels, expected to be 2 * out_channels
            out_channels (int): Number of output channels
            num_tasks (int): Number of tasks the gate can select from
            num_experts (int): Number of experts, only 5 is supported
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convt = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            nn.InstanceNorm3d(out_channels, affine=True, device=device, dtype=dtype),
            torch.nn.Mish(inplace=True),
        )
        self.conv_less = MoDESubNet2Conv3d(
            in_channels, out_channels, num_tasks, num_experts, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Upsample the input, concatenate the skip connection and convolve the result
        Args:
            x (torch.Tensor): Input tensor with shape [B, in_channels, D, H, W]
            x_skip (torch.Tensor): Skip tensor from the matching encoder block
            t (torch.Tensor): One hot task encoding with shape [B, num_tasks]
        Returns:
            torch.Tensor: Output tensor with shape [B, out_channels, 2D, 2H, 2W]
        """
        x = self.convt(x)
        x_cat: torch.Tensor = torch.cat((x_skip, x), 1)  # concatenate
        x_cat = self.conv_less(x_cat, t)
        return x_cat


class MoDESubNet2Conv3d(torch.nn.Module):
    """
    Two stacked MoDE convolutions that keep the spatial dims
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_tasks: int,
        num_experts: int = 5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Build a subnet of two MoDE convolutions
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_tasks (int): Number of tasks the gate can select from
            num_experts (int): Number of experts, only 5 is supported
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.conv1 = MoDEConv3d(
            in_channels,
            out_channels,
            num_tasks,
            num_experts,
            kernel_size=5,
            padding="same",
            device=device,
            dtype=dtype,
        )
        self.conv2 = MoDEConv3d(
            out_channels,
            out_channels,
            num_tasks,
            num_experts,
            kernel_size=5,
            padding="same",
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply both MoDE convolutions in sequence
        Args:
            x (torch.Tensor): Input tensor with shape [B, in_channels, D, H, W]
            t (torch.Tensor): One hot task encoding with shape [B, num_tasks]
        Returns:
            torch.Tensor: Output tensor with shape [B, out_channels, D, H, W]
        """
        x = self.conv1(x, t)  # [b,16,64,64,64] #[...,32,32,32] ... 4,4,4
        # x = self.rrdb(x)
        x = self.conv2(x, t)
        return x


class MoDEConv3d(torch.nn.Module):
    """
    Mixture of Diverse Experts convolution for 3d data
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_tasks: int,
        num_experts: int = 5,
        kernel_size: int = 5,
        stride: int = 1,
        padding: str = "same",
        conv_type: str = "normal",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Build a mixture of diverse experts convolution
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_tasks (int): Number of tasks the gate can select from
            num_experts (int): Number of experts, only 5 is supported
            kernel_size (int): Size of the expert kernels
            stride (int): Stride of the convolution
            padding (str): Padding mode passed to F.conv3d
            conv_type (str): normal applies InstanceNorm and Mish, final skips them
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.stride = stride
        self.padding = padding

        c_out, c_in = out_channels, in_channels
        self.expert_conv5x5_conv = self.gen_conv_kernel(c_out, c_in, 5, device=device, dtype=dtype)
        self.expert_conv3x3_conv = self.gen_conv_kernel(c_out, c_in, 3, device=device, dtype=dtype)
        self.expert_conv1x1_conv = self.gen_conv_kernel(c_out, c_in, 1, device=device, dtype=dtype)
        self.register_buffer(
            "expert_avg3x3_pool", self.gen_avgpool_kernel(3, device=device, dtype=dtype)
        )
        self.expert_avg3x3_conv = self.gen_conv_kernel(c_out, c_in, 1, device=device, dtype=dtype)
        self.register_buffer(
            "expert_avg5x5_pool", self.gen_avgpool_kernel(5, device=device, dtype=dtype)
        )
        self.expert_avg5x5_conv = self.gen_conv_kernel(c_out, c_in, 1, device=device, dtype=dtype)

        assert self.conv_type in ["normal", "final"]
        self.subsequent_layer: nn.Module
        if self.conv_type == "normal":
            self.subsequent_layer = torch.nn.Sequential(
                nn.InstanceNorm3d(out_channels, affine=True, device=device, dtype=dtype),
                torch.nn.Mish(inplace=True),
            )
        else:
            self.subsequent_layer = torch.nn.Identity()

        self.gate = torch.nn.Linear(
            num_tasks, num_experts * out_channels, bias=True, device=device, dtype=dtype
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def gen_conv_kernel(
        self,
        out_channels: int,
        in_channels: int,
        kernel_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Parameter:
        """
        Create an initialized expert kernel
        Args:
            out_channels (int): Number of output channels of the kernel
            in_channels (int): Number of input channels of the kernel
            kernel_size (int): Spatial size of the kernel
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        Returns:
            nn.Parameter: Kernel with shape [out_channels, in_channels, kernel_size, kernel_size,
                kernel_size]
        """
        weight = torch.nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
                kernel_size,
                device=device,
                dtype=dtype,
            )
        )
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode="fan_out")
        return weight

    def gen_avgpool_kernel(
        self,
        kernel_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Create an average pooling kernel
        Args:
            kernel_size (int): Spatial size of the kernel
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        Returns:
            torch.Tensor: Kernel with shape [kernel_size, kernel_size, kernel_size]
        """
        return torch.ones(kernel_size, kernel_size, kernel_size, device=device, dtype=dtype).mul(
            1.0 / kernel_size**3
        )

    def trans_kernel(self, kernel: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Pad an expert kernel symmetrically to the target spatial size
        Args:
            kernel (torch.Tensor): Kernel to pad
            target_size (int): Target spatial size of the kernel
        Returns:
            torch.Tensor: Padded kernel with spatial size target_size
        """
        depth_pad = (target_size - kernel.shape[2]) // 2
        height_pad = (target_size - kernel.shape[3]) // 2
        width_pad = (target_size - kernel.shape[4]) // 2
        return F.pad(kernel, [width_pad, width_pad, height_pad, height_pad, depth_pad, depth_pad])

    def routing(self, g: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Combine the expert kernels weighted by the gate output
        Args:
            g (torch.Tensor): Gate output with shape [batch_size, num_experts, out_channels]
            batch_size (int): Number of samples in the batch
        Returns:
            torch.Tensor: Mixed kernels with shape [batch_size, out_channels, in_channels, K, K, K]
        """
        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(self.expert_conv1x1_conv, self.kernel_size)
        expert_avg3x3 = self.trans_kernel(
            torch.einsum("oidhw,dhw->oidhw", self.expert_avg3x3_conv, self.expert_avg3x3_pool),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum(
            "oidhw,dhw->oidhw", self.expert_avg5x5_conv, self.expert_avg5x5_pool
        )

        weights: list[torch.Tensor] = []
        for n in range(batch_size):
            weight_nth_sample = (
                torch.einsum("oidhw,o->oidhw", expert_conv5x5, g[n, 0, :])
                + torch.einsum("oidhw,o->oidhw", expert_conv3x3, g[n, 1, :])
                + torch.einsum("oidhw,o->oidhw", expert_conv1x1, g[n, 2, :])
                + torch.einsum("oidhw,o->oidhw", expert_avg3x3, g[n, 3, :])
                + torch.einsum("oidhw,o->oidhw", expert_avg5x5, g[n, 4, :])
            )
            weights.append(weight_nth_sample)
        weights_tensor = torch.stack(weights)

        return weights_tensor

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convolve the input with the expert kernels mixed by the task gate
        Args:
            x (torch.Tensor): Input tensor with shape [B, in_channels, D, H, W]
            t (torch.Tensor): One hot task encoding with shape [B, num_tasks]
        Returns:
            torch.Tensor: Output tensor with shape [B, out_channels, D, H, W]
        """
        batch_size = x.shape[0]  # batch size

        g = self.gate(t)  # [b, x out channels * experts]
        g = g.view((batch_size, self.num_experts, self.out_channels))  # [b,experts,x out channels]
        g = self.softmax(g)

        w = self.routing(g, batch_size)  # [b,x out chann, 1, 5,5,5] mix expert kernels

        y: torch.Tensor
        # grouped conv so every sample uses its own mixed kernel
        spatial = x.shape[2:]
        x_grouped = x.reshape(1, batch_size * self.in_channels, *spatial)
        w_grouped = w.reshape(batch_size * self.out_channels, self.in_channels, *w.shape[-3:])
        y = F.conv3d(
            x_grouped, w_grouped, bias=None, stride=1, padding="same", groups=batch_size
        ).view(batch_size, self.out_channels, *spatial)

        y = self.subsequent_layer(y)

        return y
