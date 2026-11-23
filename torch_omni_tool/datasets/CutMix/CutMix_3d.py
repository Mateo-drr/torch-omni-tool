# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026
@author: Mateo-drr
"""

import copy
import random

import torch


def cutmix(
    chunks: torch.Tensor, lbls: torch.Tensor, mask_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Swap a random cubic patch between two adjacent samples of the batch
    Reference: https://arxiv.org/pdf/1905.04899.pdf
    Args:
        chunks (torch.Tensor): Batch of chunks with shape [b, 1, 64, 64, 64]
        lbls (torch.Tensor): Batch of label masks with shape [b, 64, 64]
        mask_size (int): Size of the cubic patch that gets swapped
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Chunks and label masks with the patch swapped
    """
    idx1 = random.randint(0, chunks.shape[0] - 2)
    idx2 = idx1 + 1

    mask_dims = (mask_size, mask_size, mask_size)
    # Randomly choose a position for the mask
    depth, height, width = chunks[idx1].shape[1:]
    d_start = random.randint(0, depth - mask_dims[0])
    h_start = random.randint(0, height - mask_dims[1])
    w_start = random.randint(0, width - mask_dims[2])

    tchunk = copy.deepcopy(chunks[idx1])
    # put crop in chunk1
    chunks[idx1][
        :,
        d_start : d_start + mask_dims[0],
        h_start : h_start + mask_dims[1],
        w_start : w_start + mask_dims[2],
    ] = chunks[idx2][
        :,
        d_start : d_start + mask_dims[0],
        h_start : h_start + mask_dims[1],
        w_start : w_start + mask_dims[2],
    ]
    # put crop in chunk2
    chunks[idx2][
        :,
        d_start : d_start + mask_dims[0],
        h_start : h_start + mask_dims[1],
        w_start : w_start + mask_dims[2],
    ] = tchunk[
        :,
        d_start : d_start + mask_dims[0],
        h_start : h_start + mask_dims[1],
        w_start : w_start + mask_dims[2],
    ]

    tlbl = copy.deepcopy(lbls[idx1])
    # put crop in label1
    lbls[idx1][h_start : h_start + mask_dims[1], w_start : w_start + mask_dims[2]] = lbls[idx2][
        h_start : h_start + mask_dims[1], w_start : w_start + mask_dims[2]
    ]
    # put crop in label2
    lbls[idx2][h_start : h_start + mask_dims[1], w_start : w_start + mask_dims[2]] = tlbl[
        h_start : h_start + mask_dims[1], w_start : w_start + mask_dims[2]
    ]

    return chunks, lbls
