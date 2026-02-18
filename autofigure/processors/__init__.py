"""图像处理器模块"""

from .sam3 import segment_with_sam3
from .rmbg import BriaRMBG2Remover, crop_and_remove_background

__all__ = [
    'segment_with_sam3',
    'BriaRMBG2Remover',
    'crop_and_remove_background',
]

