"""
Data loading and preprocessing module for the Virtual Rodent dataset.

This module provides PyTorch and PyTorch Lightning data utilities for working with
paired neural signal and pose data from the Virtual Rodent dataset.

Main classes:
- VirtualRodentDataset: PyTorch Dataset for neural-pose pairs
- SingleSessionDataModule: DataModule for single-session training
- PerSessionDataModule: DataModule for per-session evaluation (all sessions)
- PerAnimalDataModule: DataModule for per-animal chronological splits
"""

from virtual_rodent.data.dataset import VirtualRodentDataset
from virtual_rodent.data.datamodule import VirtualRodentDataModule
from virtual_rodent.data.loader import SessionLoader, DatasetIndex, discover_sessions
from virtual_rodent.data.preprocessing import (
    normalize_neural,
    normalize_pose,
    compute_pose_velocity,
    smooth_neural,
)
from virtual_rodent.data.single_session_datamodule import (
    SingleSessionDataModule,
    get_all_session_ids,
)
from virtual_rodent.data.per_session_datamodule import PerSessionDataModule
from virtual_rodent.data.per_animal_datamodule import PerAnimalDataModule

__all__ = [
    "VirtualRodentDataset",
    "VirtualRodentDataModule",
    "SingleSessionDataModule",
    "PerSessionDataModule",
    "PerAnimalDataModule",
    "SessionLoader",
    "DatasetIndex",
    "discover_sessions",
    "get_all_session_ids",
    "normalize_neural",
    "normalize_pose",
    "compute_pose_velocity",
    "smooth_neural",
]
