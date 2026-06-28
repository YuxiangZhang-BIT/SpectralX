
from .WHU_OHS import WHUOHSDataModule
from .WHU_OHS_ST import WHUOHSSTDataModule
from .DFC2020 import DFCDataModule
from .DFC2020_ST import DFCSTDataModule
from .MTS12 import MTS12DataModule
from .MTS12_ST import MTS12STDataModule

__all__ = [
    "WHUOHSDataModule",
    'WHUOHSSTDataModule',
    "DFCDataModule",
    "DFCSTDataModule",
    'MTS12DataModule',
    'MTS12STDataModule',
]
