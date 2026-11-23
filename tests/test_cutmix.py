"""
Tests for the CutMix augmentation on volumes with dense labels
"""

import random

import pytest
import torch

from torch_omni_tool.datasets.CutMix import cutmix


def _batch(batch: int, size: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a reproducible batch of chunks and labels where every value is unique
    Args:
        batch (int): Batch size
        size (int): Side of the cubic chunks and of the square labels
        seed (int): Seed of the generator
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Chunks [b, 1, s, s, s] and labels [b, s, s]
    """
    gen = torch.Generator().manual_seed(seed)
    chunks = torch.rand(batch, 1, size, size, size, generator=gen)
    labels = torch.rand(batch, size, size, generator=gen)
    return chunks, labels


@pytest.mark.parametrize("batch", [2, 3, 8])
@pytest.mark.parametrize("mask_size", [4, 16, 32])
def test_swaps_one_cube_between_two_adjacent_samples(batch: int, mask_size: int) -> None:
    random.seed(batch * 100 + mask_size)
    chunks, labels = _batch(batch)
    old_chunks, old_labels = chunks.clone(), labels.clone()
    new_chunks, new_labels = cutmix(chunks, labels, mask_size=mask_size)

    changed = [i for i in range(batch) if not torch.equal(new_chunks[i], old_chunks[i])]
    assert len(changed) == 2 and changed[1] == changed[0] + 1
    i, j = changed
    diff = new_chunks[i] != old_chunks[i]
    # the changed voxels form one mask_size cube, swapped in both directions
    idx = diff[0].nonzero()
    assert (idx.max(dim=0).values - idx.min(dim=0).values + 1).tolist() == [mask_size] * 3
    assert torch.equal(new_chunks[i][diff], old_chunks[j][diff])
    assert torch.equal(new_chunks[j][diff], old_chunks[i][diff])
    # the labels swap the matching [h, w] window
    window = diff[0].any(dim=0)
    assert torch.equal(new_labels[i][window], old_labels[j][window])
    assert torch.equal(new_labels[j][window], old_labels[i][window])
    assert torch.equal(new_labels[i][~window], old_labels[i][~window])


def test_modifies_the_inputs_in_place() -> None:
    random.seed(0)
    chunks, labels = _batch(2)
    new_chunks, new_labels = cutmix(chunks, labels, mask_size=8)
    assert new_chunks is chunks and new_labels is labels


def test_preserves_the_batch_content() -> None:
    random.seed(0)
    chunks, labels = _batch(4)
    total_chunks, total_labels = chunks.sum(dim=0).clone(), labels.sum(dim=0).clone()
    new_chunks, new_labels = cutmix(chunks, labels, mask_size=8)
    torch.testing.assert_close(new_chunks.sum(dim=0), total_chunks)
    torch.testing.assert_close(new_labels.sum(dim=0), total_labels)


def test_needs_a_batch_of_at_least_two() -> None:
    chunks, labels = _batch(1)
    with pytest.raises(ValueError):
        cutmix(chunks, labels, mask_size=8)
