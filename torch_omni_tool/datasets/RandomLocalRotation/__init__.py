"""
Random local rotation augmentation, in 2d and 3d
"""

from torch_omni_tool.datasets.RandomLocalRotation.RandomLocalRotation_2d import (
    random_local_rotation_2d,
)
from torch_omni_tool.datasets.RandomLocalRotation.RandomLocalRotation_3d import (
    random_local_rotation_3d,
)

__all__ = ["random_local_rotation_2d", "random_local_rotation_3d"]
