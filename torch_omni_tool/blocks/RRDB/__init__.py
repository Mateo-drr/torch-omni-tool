"""
Residual-in-Residual Dense Blocks from ESRGAN, in 2d and 3d
"""

from torch_omni_tool.blocks.RRDB.RRDB_2d import RRDB2d, ResidualDenseBlock_5C_2d
from torch_omni_tool.blocks.RRDB.RRDB_3d import RRDB3d, ResidualDenseBlock_5C_3d

__all__ = ["RRDB2d", "RRDB3d", "ResidualDenseBlock_5C_2d", "ResidualDenseBlock_5C_3d"]
