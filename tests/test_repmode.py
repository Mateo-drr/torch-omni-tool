"""
Tests for the MoDE convolutions, the encoder / decoder blocks and the MoDE wrapper
"""

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.testing import assert_close

from torch_omni_tool.blocks.RepMoDE import (
    MoDE,
    MoDEConv1d,
    MoDEConv2d,
    MoDEConv3d,
    MoDEDecoderBlock1d,
    MoDEDecoderBlock3d,
    MoDEEncoderBlock1d,
    MoDEEncoderBlock3d,
    MoDESubNet2Conv1d,
    MoDESubNet2Conv3d,
)

NUM_TASKS = 3
TASKS = torch.eye(NUM_TASKS)[[0, 2, 1]]  # one-hot tasks of a batch of 3, all different
ConvCls = type[MoDEConv1d] | type[MoDEConv2d] | type[MoDEConv3d]
CONVS: list[tuple[ConvCls, tuple[int, ...], Callable[..., torch.Tensor]]] = [
    (MoDEConv1d, (3, 4, 16), F.conv1d),
    (MoDEConv2d, (3, 4, 8, 8), F.conv2d),
    (MoDEConv3d, (3, 4, 6, 6, 6), F.conv3d),
]
CONV_IDS = ["1d", "2d", "3d"]


def _randn(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """
    Build a reproducible random tensor
    Args:
        shape (tuple[int, ...]): Shape of the tensor
        seed (int): Seed of the generator
    Returns:
        torch.Tensor: Random normal tensor
    """
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _mixed_kernels(conv: MoDEConv1d | MoDEConv2d | MoDEConv3d, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the per-sample kernels the gate mixes for a batch of tasks
    Args:
        conv (MoDEConv1d | MoDEConv2d | MoDEConv3d): MoDE convolution
        t (torch.Tensor): One-hot tasks with shape [B, num_tasks]
    Returns:
        torch.Tensor: Kernels with shape [B, out_channels, in_channels, *kernel]
    """
    g = conv.softmax(conv.gate(t).view(t.shape[0], conv.num_experts, conv.out_channels))
    return conv.routing(g, t.shape[0])


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize(("cls", "shape", "conv_fn"), CONVS, ids=CONV_IDS)
def test_each_sample_uses_its_own_task_kernel(
    cls: ConvCls, shape: tuple[int, ...], conv_fn: Callable[..., torch.Tensor], training: bool
) -> None:
    # regression: eval used to convolve the whole batch with the first sample's kernel
    torch.manual_seed(0)
    conv = cls(4, 6, NUM_TASKS, conv_type="final").train(training)
    x = _randn(shape)
    w = _mixed_kernels(conv, TASKS)
    expected = torch.cat([conv_fn(x[i : i + 1], w[i], padding="same") for i in range(3)])
    assert_close(conv(x, TASKS), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(("cls", "shape", "conv_fn"), CONVS, ids=CONV_IDS)
def test_batched_output_matches_single_samples(
    cls: ConvCls, shape: tuple[int, ...], conv_fn: Callable[..., torch.Tensor]
) -> None:
    torch.manual_seed(0)
    conv = cls(4, 6, NUM_TASKS).eval()
    x = _randn(shape)
    single = torch.cat([conv(x[i : i + 1], TASKS[i : i + 1]) for i in range(3)])
    assert_close(conv(x, TASKS), single, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("conv_type", ["normal", "final"])
@pytest.mark.parametrize(("cls", "shape", "conv_fn"), CONVS, ids=CONV_IDS)
def test_output_shape(
    cls: ConvCls, shape: tuple[int, ...], conv_fn: Callable[..., torch.Tensor], conv_type: str
) -> None:
    out = cls(4, 6, NUM_TASKS, conv_type=conv_type)(_randn(shape), TASKS)
    assert out.shape == (shape[0], 6, *shape[2:])


@pytest.mark.parametrize(("cls", "shape", "conv_fn"), CONVS, ids=CONV_IDS)
def test_task_changes_the_output(
    cls: ConvCls, shape: tuple[int, ...], conv_fn: Callable[..., torch.Tensor]
) -> None:
    torch.manual_seed(0)
    conv = cls(4, 6, NUM_TASKS).eval()
    # a broadcast (non-contiguous) batch, regression for MoDEConv2d calling view on it
    x = _randn(shape)[:1].expand(2, *shape[1:])
    out = conv(x, torch.eye(NUM_TASKS)[[0, 1]])
    assert not torch.allclose(out[0], out[1])


@pytest.mark.parametrize(("cls", "shape", "conv_fn"), CONVS, ids=CONV_IDS)
def test_gate_weights_are_a_softmax_over_experts(
    cls: ConvCls, shape: tuple[int, ...], conv_fn: Callable[..., torch.Tensor]
) -> None:
    conv = cls(4, 6, NUM_TASKS)
    g = conv.softmax(conv.gate(TASKS).view(3, conv.num_experts, conv.out_channels))
    assert_close(g.sum(dim=1), torch.ones(3, 6))


def test_mode_conv_2d_grouped_and_loop_agree() -> None:
    torch.manual_seed(0)
    conv = MoDEConv2d(4, 6, NUM_TASKS)
    x = _randn((3, 4, 8, 8))
    assert_close(conv(x, TASKS, grouped=True), conv(x, TASKS, grouped=False), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kernel_size", [3, 7])
@pytest.mark.xfail(strict=True, reason="known issue: MoDEConv2d only supports kernel_size=5")
def test_mode_conv_2d_other_kernel_sizes(kernel_size: int) -> None:
    conv = MoDEConv2d(4, 6, NUM_TASKS, kernel_size=kernel_size)
    assert conv(_randn((3, 4, 8, 8)), TASKS).shape == (3, 6, 8, 8)


# causal 1d


def test_causal_mode_uses_batch_length_channels_layout() -> None:
    conv = MoDEConv1d(4, 6, NUM_TASKS, causal=True, conv_type="final")
    assert conv(_randn((3, 20, 4)), TASKS).shape == (3, 20, 6)


def _future_leak(conv_type: str) -> bool:
    """
    Tell whether changing the second half of a sequence changes the first half of the output
    Args:
        conv_type (str): conv_type of the causal convolution
    Returns:
        bool: True when future steps leak into past outputs
    """
    torch.manual_seed(0)
    conv = MoDEConv1d(4, 6, NUM_TASKS, causal=True, conv_type=conv_type).eval()
    a = _randn((1, 20, 4))
    b = a.clone()
    b[:, 10:] += 5
    t = TASKS[:1]
    return not torch.allclose(conv(a, t)[:, :10], conv(b, t)[:, :10])


def test_causal_final_conv_does_not_see_the_future() -> None:
    assert not _future_leak("final")


@pytest.mark.xfail(strict=True, reason="known issue: InstanceNorm1d leaks future steps")
def test_causal_normal_conv_does_not_see_the_future() -> None:
    assert not _future_leak("normal")


# encoder / decoder / subnet


@pytest.mark.parametrize(
    ("encoder", "decoder", "subnet", "spatial"),
    [
        (MoDEEncoderBlock1d, MoDEDecoderBlock1d, MoDESubNet2Conv1d, (16,)),
        (MoDEEncoderBlock3d, MoDEDecoderBlock3d, MoDESubNet2Conv3d, (8, 8, 8)),
    ],
    ids=["1d", "3d"],
)
def test_encoder_decoder_round_trip_shapes(
    encoder: type[nn.Module],
    decoder: type[nn.Module],
    subnet: type[nn.Module],
    spatial: tuple[int, ...],
) -> None:
    x = _randn((3, 4, *spatial))
    down, skip = encoder(4, 8, NUM_TASKS)(x, TASKS)
    assert skip.shape == (3, 8, *spatial)
    assert down.shape == (3, 8, *(s // 2 for s in spatial))
    bottle = subnet(8, 16, NUM_TASKS)(down, TASKS)
    assert bottle.shape == (3, 16, *(s // 2 for s in spatial))
    assert decoder(16, 8, NUM_TASKS)(bottle, skip, TASKS).shape == (3, 8, *spatial)


# MoDE wrapper


def test_mode_returns_tasks_unchanged_in_eval() -> None:
    block = MoDE(4, 6, num_tasks=2).eval()
    task_id = torch.tensor([1, 2, 0])
    out, used = block(_randn((3, 4, 8, 8)), task_id)
    assert out.shape == (3, 6, 8, 8)
    assert torch.equal(used, task_id)


@pytest.mark.parametrize(("prob", "expected"), [(0.0, [1, 2, 1]), (1.0, [0, 0, 0])])
def test_mode_global_task_swap_in_training(prob: float, expected: list[int]) -> None:
    block = MoDE(4, 6, num_tasks=2, global_task_train_prob=prob).train()
    task_id = torch.tensor([1, 2, 1])
    _, used = block(_randn((3, 4, 8, 8)), task_id)
    assert used.tolist() == expected
    assert task_id.tolist() == [1, 2, 1]  # the caller's tensor is not modified


def test_mode_rejects_out_of_range_tasks_in_training() -> None:
    with pytest.raises(AssertionError):
        MoDE(4, 6, num_tasks=2).train()(_randn((1, 4, 8, 8)), torch.tensor([3]))


@pytest.mark.parametrize("task_id", [None, 1])
def test_mode_task_weights(task_id: int | None) -> None:
    weights = MoDE(4, 6, num_tasks=2).get_task_weights(task_id)
    assert sorted(weights) == ([0, 1, 2] if task_id is None else [1])
    for w in weights.values():
        assert w.shape == (5, 6)
        assert_close(torch.from_numpy(w).sum(dim=0), torch.ones(6))


@pytest.mark.parametrize(
    "build",
    [
        lambda: MoDEConv1d(4, 6, NUM_TASKS, dtype=torch.float64),
        lambda: MoDEConv3d(4, 6, NUM_TASKS, dtype=torch.float64),
        lambda: MoDEEncoderBlock3d(4, 8, NUM_TASKS, dtype=torch.float64),
        lambda: MoDE(4, 6, 2, dtype=torch.float64),
    ],
    ids=["conv1d", "conv3d", "encoder3d", "mode"],
)
def test_dtype_reaches_parameters_and_buffers(build: Callable[[], nn.Module]) -> None:
    module = build()
    tensors = [*module.parameters(), *module.buffers()]
    assert all(t.dtype == torch.float64 for t in tensors if t.is_floating_point())
