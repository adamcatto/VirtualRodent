"""
Per-session DataModule with temporal splits.

This module provides a PyTorch Lightning DataModule that splits each session
temporally into train/val/test sets with gap regions to prevent temporal leakage.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from virtual_rodent.data.dataset import VirtualRodentDataset
from virtual_rodent.data.loader import SessionLoader, discover_sessions
from virtual_rodent.environment.session_env import SessionEnvironment


def _should_pin_memory() -> bool:
    """Determine if pin_memory should be used (not supported on MPS)."""
    if torch.backends.mps.is_available():
        return False
    return torch.cuda.is_available()


class PerSessionDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for per-session evaluation.

    Each session is split temporally:
    - 60% train / 1% gap / 19% val / 1% gap / 19% test

    The gap regions prevent temporal leakage between splits. This strategy
    tests model performance on later time points within the same recording session.

    Example:
        >>> dm = PerSessionDataModule(
        ...     data_dir="data/Virtual_Rodent",
        ...     neural_history=750,
        ...     pose_horizon=250,
        ...     batch_size=32,
        ... )
        >>> dm.setup("fit")
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        brain_regions: Optional[List[str]] = None,
        animals: Optional[List[str]] = None,
        # Temporal configuration
        neural_history: int = 750,  # 15 seconds at 50Hz
        pose_history: int = 750,
        pose_horizon: int = 250,  # 5 seconds at 50Hz
        # Split configuration
        split_ratios: tuple = (0.6, 0.01, 0.19, 0.01, 0.19),
        # Dataset configuration
        pose_type: str = "keypoints",
        flatten_pose: bool = True,
        normalize_neural_method: Optional[str] = "sqrt",
        normalize_pose_method: Optional[str] = "center",
        # DataLoader configuration
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: Optional[bool] = None,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        # Random seed
        seed: int = 42,
    ):
        """
        Initialize the Per-Session DataModule.

        Args:
            data_dir: Root directory containing Virtual_Rodent data
            brain_regions: Brain regions to include (None = all)
            animals: Animals to include (None = all)
            neural_history: Frames of neural history
            pose_history: Frames of pose history
            pose_horizon: Frames to predict into future
            split_ratios: Tuple of (train, gap1, val, gap2, test) ratios
            pose_type: "keypoints" or "qpos"
            flatten_pose: Flatten pose to 1D
            normalize_neural_method: Neural normalization method
            normalize_pose_method: Pose normalization method
            batch_size: Batch size for DataLoaders
            num_workers: Number of data loading workers
            pin_memory: Pin tensors to CUDA memory
            persistent_workers: Keep workers alive between epochs
            prefetch_factor: Number of batches to prefetch per worker
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.brain_regions = brain_regions
        self.animals = animals
        self.neural_history = neural_history
        self.pose_history = pose_history
        self.pose_horizon = pose_horizon
        self.split_ratios = split_ratios
        self.pose_type = pose_type
        self.flatten_pose = flatten_pose
        self.normalize_neural_method = normalize_neural_method
        self.normalize_pose_method = normalize_pose_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory if pin_memory is not None else _should_pin_memory()
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.seed = seed

        # Will be set in setup()
        self.train_dataset: Optional[VirtualRodentDataset] = None
        self.val_dataset: Optional[VirtualRodentDataset] = None
        self.test_dataset: Optional[VirtualRodentDataset] = None
        self.session_envs: List[SessionEnvironment] = []

        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['data_dir'])

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage in ["fit", "validate", None]:
            self._setup_splits()

        if stage == "test" or stage is None:
            if self.test_dataset is None:
                self._setup_splits()

    def _setup_splits(self):
        """Set up train/val/test splits with temporal boundaries."""
        # Discover all sessions
        index = discover_sessions(
            self.data_dir,
            brain_regions=self.brain_regions,
            animals=self.animals,
        )

        if len(index.sessions) == 0:
            raise ValueError(f"No sessions found in {self.data_dir}")

        # Create SessionEnvironments for each session
        self.session_envs = [
            SessionEnvironment(
                SessionLoader(session.file_path),
                split_ratios=self.split_ratios,
            )
            for session in index.sessions
        ]

        print(f"Created {len(self.session_envs)} session environments")

        # Create base dataset (full data)
        full_dataset = VirtualRodentDataset(
            data_dir=self.data_dir,
            brain_regions=self.brain_regions,
            animals=self.animals,
            neural_history=self.neural_history,
            pose_history=self.pose_history,
            pose_horizon=self.pose_horizon,
            pose_type=self.pose_type,
            flatten_pose=self.flatten_pose,
            normalize_neural_method=self.normalize_neural_method,
            normalize_pose_method=self.normalize_pose_method,
            preload=False,
            cache_sessions=4,
        )

        print(f"Base dataset created with {len(full_dataset)} total samples")

        # Get the valid indices from the dataset
        valid_indices = full_dataset._valid_indices
        
        # For each valid index, determine which split it belongs to
        train_mask = np.zeros(len(valid_indices), dtype=bool)
        val_mask = np.zeros(len(valid_indices), dtype=bool)
        test_mask = np.zeros(len(valid_indices), dtype=bool)
        
        for i, global_idx in enumerate(valid_indices):
            # Find which session this index belongs to
            session_idx, frame_idx = full_dataset.index.get_session_and_frame(global_idx)
            
            # Get the session environment
            env = self.session_envs[session_idx]
            
            # Check which split this frame belongs to
            for split_name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
                start, end = env.splits[split_name]
                # Also check temporal margins
                margin_before = max(self.neural_history, self.pose_history) - 1
                margin_after = self.pose_horizon - 1
                if frame_idx >= start + margin_before and frame_idx < end - margin_after:
                    mask[i] = True
                    break

        # Get indices into the dataset (not global frame indices)
        train_indices = np.where(train_mask)[0].tolist()
        val_indices = np.where(val_mask)[0].tolist()
        test_indices = np.where(test_mask)[0].tolist()

        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples: {len(val_indices)}")
        print(f"Test samples: {len(test_indices)}")

        # Create subsets
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader (uses test set)."""
        return self.test_dataloader()

    @property
    def input_dim(self) -> int:
        """Dimension of neural input features."""
        if self.train_dataset is not None:
            ds = self.train_dataset
            if isinstance(ds, Subset):
                ds = ds.dataset
            return ds.input_dim
        return 0

    @property
    def output_dim(self) -> int:
        """Dimension of pose output features."""
        if self.train_dataset is not None:
            ds = self.train_dataset
            if isinstance(ds, Subset):
                ds = ds.dataset
            return ds.output_dim
        return 0

    def __repr__(self) -> str:
        train_len = len(self.train_dataset) if self.train_dataset else 0
        val_len = len(self.val_dataset) if self.val_dataset else 0
        test_len = len(self.test_dataset) if self.test_dataset else 0

        return (
            f"PerSessionDataModule(\n"
            f"  strategy='per_session',\n"
            f"  train_samples={train_len},\n"
            f"  val_samples={val_len},\n"
            f"  test_samples={test_len},\n"
            f"  batch_size={self.batch_size},\n"
            f"  sessions={len(self.session_envs)}\n"
            f")"
        )
