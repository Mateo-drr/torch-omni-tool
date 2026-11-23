# RepMoDE

Mixture-of-Diverse-Experts (MoDE) convolutions whose expert kernels are re-parameterized per
task by a gate, so the network keeps the topology of a plain convolution while behaving like a
task-specific expert.

| Class | File | Input |
| --- | --- | --- |
| `MoDE` (wrapper), `MoDEConv2d` | [`RepMoDE_2d.py`](RepMoDE_2d.py) | `[B, C, H, W]` |
| `MoDEConv1d`, `MoDEEncoderBlock1d`, `MoDESubNet2Conv1d`, `MoDEDecoderBlock1d` | [`RepMoDE_1d.py`](RepMoDE_1d.py) | `[B, C, L]` (`[B, L, C]` with `causal=True`) |
| `MoDEConv3d`, `MoDEEncoderBlock3d`, `MoDESubNet2Conv3d`, `MoDEDecoderBlock3d` | [`RepMoDE_3d.py`](RepMoDE_3d.py) | `[B, C, D, H, W]` |

All classes are exported by `torch_omni_tool.blocks.RepMoDE`.

## Source

- **Paper:** Donghao Zhou, Chunbin Gu, Junde Xu, Furui Liu, Qiong Wang, Guangyong Chen,
  Pheng-Ann Heng, *RepMode: Learning to Re-parameterize Diverse Experts for Subcellular
  Structure Prediction*, CVPR 2023 — [arXiv:2212.10066](https://arxiv.org/abs/2212.10066)
- **Reference code:** the official 3D implementation,
  [`correr-zhou/RepMode` `fnet/nn_modules/RepMode.py`](https://github.com/correr-zhou/RepMode/blob/main/fnet/nn_modules/RepMode.py)
  (project page: [correr-zhou.github.io/RepMode](https://correr-zhou.github.io/RepMode)). The 3D
  module keeps its classes (upstream `MoDEConv`, `MoDEEncoderBlock`, `MoDEDecoderBlock`,
  `MoDESubNet2Conv`, here with a `3d` suffix) and methods (`gen_conv_kernel`, `trans_kernel`,
  `routing`) with these differences:
  - Mish instead of ReLU, and InstanceNorm instead of BatchNorm after the MoDE and transposed
    convolutions (the encoder downsampling keeps BatchNorm).
  - Upstream convolves the whole batch with `w[0]` in eval mode; here every sample always uses
    its own kernel through a grouped convolution.
  - The 1D (with the causal option) and 2D modules are ports of the same design, and the `MoDE`
    wrapper with the global task is an addition of this package.

## How a MoDE convolution works

Instead of routing samples to separate expert sub-networks, every layer stores **five
shared expert kernels** and mixes them per sample and per output channel:

| Expert | Kernel | Diversity |
| --- | --- | --- |
| 0 | 5x5 conv | shape (receptive field) |
| 1 | 3x3 conv, zero-padded to `kernel_size` | shape |
| 2 | 1x1 conv, zero-padded to `kernel_size` | shape |
| 3 | 1x1 conv composed with a 3x3 average-pool kernel | kernel |
| 4 | 1x1 conv composed with a 5x5 average-pool kernel | kernel |

That covers the paper's two diversity axes: *shape* diversity (different receptive fields)
and *kernel* diversity (plain convolutions vs. smoothed ones).

The gate is a single `nn.Linear(num_tasks, num_experts * out_channels)` applied to the one-hot
task encoding, reshaped to `[B, num_experts, out_channels]` and softmaxed over the expert axis.
`routing()` then builds one mixed kernel per sample and per output channel:

```
w[b, o] = sum_e g[b, e, o] * expert_e[o]
```

giving a tensor of shape `[B, out_channels, in_channels, K, K]`, which is applied with a grouped
convolution (`groups=B`) so every sample uses its own kernel, in train and eval mode alike. Everything is differentiable: the
gradient flows through the gate into the shared expert kernels.

Shared building blocks across the three modules:

- `gen_conv_kernel(chans_out, chans_in, k_size)` — Kaiming-initialized expert kernel.
- `gen_avg_pool_kernel(kernel_size)` / `gen_avgpool_kernel` — uniform averaging kernel kept
  in a buffer.
- `trans_kernel(kernel, target_size)` — pads a small expert kernel up to `kernel_size`
  (left only when `causal=True`).
- `routing(g, batch_size)` — the weighted expert mixture.
- `conv_type="normal"` appends `InstanceNorm + Mish`; `"final"` (or any other value)
  appends `nn.Identity`, which is what the output convolution of a network wants.

## Task encoding

Every block takes a one-hot task tensor `t` of shape `[B, num_tasks]` as its second
argument. In the paper the task id is a prior supplied per sample; here it is up to the
caller to build it, e.g. `F.one_hot(task_id, num_classes=num_tasks).float()`.

`MoDE` wraps `MoDEConv2d` (fixed at 5 experts, `kernel_size=5`,
`conv_type="final"`) and adds the **global task**: during training, with probability
`global_task_train_prob` (0.2 by default) a sample's task id is replaced by
`global_task_id` (0 by default), which keeps the shared experts useful for samples whose
task is unknown at inference time. `forward` returns the tensor *and* the task ids it
actually used, and `get_task_weights(task_id=None)` returns the gate weights as numpy
arrays of shape `[num_experts, out_channels]`, either for one task or for all of them.

```python
import torch
from torch_omni_tool.blocks.RepMoDE import MoDE

block = MoDE(in_channels=3, out_channels=64, num_tasks=2)
y, task_id = block(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]))
weights = block.get_task_weights()  # {task_id: [5, 64]}
```

## Encoder / decoder topology (1d and 3d)

The 1d and 3d modules ship the autoencoder topology from the paper's code:

Each block comes as `...1d` and `...3d`:

- `MoDEEncoderBlock{1d,3d}(in_channels, out_channels, num_tasks)` — `MoDESubNet2Conv` to widen the channels, then a
  strided `Conv{1,3}d(kernel=2, stride=2)` + norm + Mish to halve the resolution. Returns
  the downsampled tensor and a skip tensor with `out_channels` channels.
- `MoDESubNet2Conv{1d,3d}(in_channels, out_channels, num_tasks)` — two `MoDEConv`s (`kernel_size=5`, `padding="same"`) at
  constant resolution.
- `MoDEDecoderBlock{1d,3d}(in_channels, out_channels, num_tasks)` — `ConvTranspose{1,3}d(kernel=2, stride=2)` +
  norm + Mish to double the resolution, concatenate the matching skip, then
  `MoDESubNet2Conv`. `in_channels` is expected to be `2 * out_channels`.

```python
import torch
from torch_omni_tool.blocks.RepMoDE import (
    MoDEDecoderBlock3d,
    MoDEEncoderBlock3d,
    MoDESubNet2Conv3d,
)

t = torch.eye(3)[:2]  # one-hot task ids [B, num_tasks]
x = torch.randn(2, 4, 16, 16, 16)
y, skip = MoDEEncoderBlock3d(in_channels=4, out_channels=8, num_tasks=3)(x, t)
b = MoDESubNet2Conv3d(in_channels=8, out_channels=16, num_tasks=3)(y, t)
d = MoDEDecoderBlock3d(in_channels=16, out_channels=8, num_tasks=3)(b, skip, t)
```

A full U-Net style stack (encoder → bottle → decoder → output conv) chains the blocks with
the channels doubling at every level:

```python
from torch import nn
import torch.nn.functional as F
from torch_omni_tool.blocks.RepMoDE import (
    MoDEConv3d,
    MoDEDecoderBlock3d,
    MoDEEncoderBlock3d,
    MoDESubNet2Conv3d,
)


class MoDEUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_tasks: int, base: int = 16) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        chans = [in_channels, base, base * 2, base * 4, base * 8]
        self.encoders = nn.ModuleList(
            MoDEEncoderBlock3d(c_in, c_out, num_tasks) for c_in, c_out in zip(chans, chans[1:])
        )
        self.bottle = MoDESubNet2Conv3d(base * 8, base * 16, num_tasks)
        self.decoders = nn.ModuleList(
            MoDEDecoderBlock3d(c * 2, c, num_tasks) for c in reversed(chans[1:])
        )
        self.conv_out = MoDEConv3d(base, out_channels, num_tasks, conv_type="final")

    def forward(self, x, task_id):
        t = F.one_hot(task_id, num_classes=self.num_tasks).float()
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x, t)
            skips.append(skip)
        x = self.bottle(x, t)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip, t)
        return self.conv_out(x, t)
```

The same stack works with the `1d` blocks on `[B, C, L]` input.

## Caveats

- `num_experts` must stay at its default of 5 (`routing()` indexes experts `0..4`) and
  `kernel_size` must be 5
  (the 5x5 expert is never resized).
- `stride` is ignored by `MoDEConv1d` / `MoDEConv3d`, and `MoDEConv2d` fails for `stride != 1`.
- With `causal=True` `MoDEConv1d` takes and returns `[B, L, C]`, unlike the `[B, C, L]`
  layout of the non-causal path. Use `conv_type="final"` for strict causality, the
  `InstanceNorm1d` of `conv_type="normal"` normalizes over the whole sequence, future steps
  included.
- `one_hot_task_embedding` in the 1d and 3d modules takes `self` and cannot be called as a
  plain function. Build the one-hot tensor with `F.one_hot` instead.
