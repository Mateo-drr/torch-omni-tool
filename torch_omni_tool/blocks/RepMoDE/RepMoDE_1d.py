# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 22:54:53 2025
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


class MoDEEncoderBlock1d(torch.nn.Module):
    """
    Encoder block that increases the channels and halves the sequence length
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
        self.conv_more = MoDESubNet2Conv1d(
            in_channels, out_channels, num_tasks, num_experts, device=device, dtype=dtype
        )
        self.conv_down = torch.nn.Sequential(
            torch.nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            nn.BatchNorm1d(out_channels, affine=True, device=device, dtype=dtype),
            torch.nn.Mish(inplace=True),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convolve the input and downsample it by two
        Args:
            x (torch.Tensor): Input tensor with shape [B, in_channels, L]
            t (torch.Tensor): One hot task encoding with shape [B, num_tasks]
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Downsampled tensor and skip tensor with out_channels
                channels
        """
        x_skip = self.conv_more(x, t)
        x = self.conv_down(x_skip)
        return x, x_skip


class MoDEDecoderBlock1d(torch.nn.Module):
    """
    Decoder block that halves the channels and doubles the sequence length
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
            torch.nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            nn.InstanceNorm1d(out_channels, affine=True, device=device, dtype=dtype),
            torch.nn.Mish(inplace=True),
        )
        self.conv_less = MoDESubNet2Conv1d(
            in_channels, out_channels, num_tasks, num_experts, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Upsample the input, concatenate the skip connection and convolve the result
        Args:
            x (torch.Tensor): Input tensor with shape [B, in_channels, L]
            x_skip (torch.Tensor): Skip tensor from the matching encoder block
            t (torch.Tensor): One hot task encoding with shape [B, num_tasks]
        Returns:
            torch.Tensor: Output tensor with shape [B, out_channels, 2L]
        """
        x = self.convt(x)
        x_cat: torch.Tensor = torch.cat((x_skip, x), 1)  # concatenate
        x_cat = self.conv_less(x_cat, t)
        return x_cat


class MoDESubNet2Conv1d(torch.nn.Module):
    """
    Two stacked MoDE convolutions that keep the sequence length
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
        self.conv1 = MoDEConv1d(
            in_channels,
            out_channels,
            num_tasks,
            num_experts,
            kernel_size=5,
            padding="same",
            device=device,
            dtype=dtype,
        )
        self.conv2 = MoDEConv1d(
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
            x (torch.Tensor): Input tensor with shape [B, in_channels, L]
            t (torch.Tensor): One hot task encoding with shape [B, num_tasks]
        Returns:
            torch.Tensor: Output tensor with shape [B, out_channels, L]
        """
        x = self.conv1(x, t)
        x = self.conv2(x, t)
        return x


class MoDEConv1d(torch.nn.Module):
    """
    Mixture of Diverse Experts convolution for 1d data, optionally causal
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
        causal: bool = False,
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
            padding (str): Padding mode passed to F.conv1d
            conv_type (str): normal applies InstanceNorm and Mish, final skips them
            causal (bool): Whether the convolution only looks at past samples
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
        self.causal = causal

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
                nn.InstanceNorm1d(out_channels, affine=True, device=device, dtype=dtype),
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
            kernel_size (int): Size of the kernel
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        Returns:
            nn.Parameter: Kernel with shape [out_channels, in_channels, kernel_size]
        """
        weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, device=device, dtype=dtype)
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
            kernel_size (int): Size of the kernel
            device (torch.device | str | None): Device of the parameters
            dtype (torch.dtype | None): Data type of the parameters
        Returns:
            torch.Tensor: Kernel with shape [kernel_size]
        """
        return torch.ones(kernel_size, device=device, dtype=dtype).mul(1.0 / kernel_size)

    def trans_kernel(self, kernel: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Pad an expert kernel to the target size, only on the left when causal
        Args:
            kernel (torch.Tensor): Kernel to pad
            target_size (int): Target size of the kernel
        Returns:
            torch.Tensor: Padded kernel with size target_size
        """
        # For 1D convolution, padding is only applied to the last dimension
        if self.causal:
            pad = target_size - kernel.shape[2]
            return F.pad(kernel, [pad, 0])
        else:
            pad = (target_size - kernel.shape[2]) // 2
            return F.pad(kernel, [pad, pad])

    def routing(self, g: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Combine the expert kernels weighted by the gate output
        Args:
            g (torch.Tensor): Gate output with shape [batch_size, num_experts, out_channels]
            batch_size (int): Number of samples in the batch
        Returns:
            torch.Tensor: Mixed kernels with shape [batch_size, out_channels, in_channels, K]
        """
        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(self.expert_conv1x1_conv, self.kernel_size)

        # For 1D convolution, we use einsum with appropriate dimensions
        expert_avg3x3 = self.trans_kernel(
            torch.einsum("oiw,w->oiw", self.expert_avg3x3_conv, self.expert_avg3x3_pool),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum("oiw,w->oiw", self.expert_avg5x5_conv, self.expert_avg5x5_pool)

        weights: list[torch.Tensor] = []
        for n in range(batch_size):
            weight_nth_sample = (
                torch.einsum("oiw,o->oiw", expert_conv5x5, g[n, 0, :])
                + torch.einsum("oiw,o->oiw", expert_conv3x3, g[n, 1, :])
                + torch.einsum("oiw,o->oiw", expert_conv1x1, g[n, 2, :])
                + torch.einsum("oiw,o->oiw", expert_avg3x3, g[n, 3, :])
                + torch.einsum("oiw,o->oiw", expert_avg5x5, g[n, 4, :])
            )
            weights.append(weight_nth_sample)
        weights_tensor = torch.stack(weights)

        return weights_tensor

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convolve the input with the expert kernels mixed by the task gate
        Args:
            x (torch.Tensor): Input tensor with shape [B, in_channels, L]
            t (torch.Tensor): One hot task encoding with shape [B, num_tasks]
        Returns:
            torch.Tensor: Output tensor with shape [B, out_channels, L]
        """
        batch_size = x.shape[0]  # batch size

        g = self.gate(t)  # [b, x out channels * experts]
        g = g.view((batch_size, self.num_experts, self.out_channels))  # [b,experts,x out channels]
        g = self.softmax(g)

        w = self.routing(g, batch_size)  # [b,x out chann, 1, 5] mix expert kernels

        y: torch.Tensor
        if self.causal:  # handle padding to not look into i+1 items
            pad_left = self.kernel_size - 1
            x = x.transpose(1, 2)  # makes [b,seqlen,dmodel] to [b,dmodel,seqlen]
            """
            processing everything as one batch
            """

            # Pad the entire batch at once
            x_padded = F.pad(x, (pad_left, 0), "constant", 0)  # Shape: [48, 768, 387]

            # Create a single large batch for all examples
            x_batched = x_padded.view(1, -1, x_padded.shape[2])  # Shape: [1, 48*768, 387]

            # Reshape and concatenate all weights
            w_batched = w.view(
                batch_size * self.out_channels, self.in_channels, self.kernel_size
            )  # Shape: [48*768, 768, 5]

            # Use grouped convolution
            y_batched = F.conv1d(
                x_batched,  # [1, 48*768, 387]
                w_batched,  # [48*768, 768, 5]
                bias=None,
                stride=1,
                padding=0,
                groups=batch_size,  # 48 groups
            )

            # Reshape result back to original batch format
            y = y_batched.view(batch_size, self.out_channels, -1)  # Shape: [48, 768, 383]

        else:
            # grouped conv so every sample uses its own mixed kernel
            x_grouped = x.reshape(1, batch_size * self.in_channels, x.shape[2])
            w_grouped = w.reshape(
                batch_size * self.out_channels, self.in_channels, self.kernel_size
            )
            y = F.conv1d(
                x_grouped, w_grouped, bias=None, stride=1, padding="same", groups=batch_size
            ).view(batch_size, self.out_channels, x.shape[2])

        y = self.subsequent_layer(y)

        if self.causal:
            y = y.transpose(1, 2)

        return y
