"""
Per-animal DataModule with chronological session splits.

This module provides a PyTorch Lightning DataModule that splits sessions
chronologically per animal to test generalization to later recording sessions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from virtual_rodent.data.dataset import VirtualRodentDataset
from virtual_rodent.data.loader import BRAIN_REGIONS, discover_sessions


def _should_pin_memory() -> bool:
    """Determine if pin_memory should be used (not supported on MPS)."""
    if torch.backends.mps.is_available():
        return False
    return torch.cuda.is_available()


class PerAnimalDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for per-animal evaluation.

    Sessions are split chronologically per animal:
    - First sessions → train
    - Middle sessions → val
    - Last sessions → test

    This tests whether the model can generalize to later recording sessions,
    which is critical for neuroscience applications where neural distributions
    may drift over time.

    Example:
        >>> dm = PerAnimalDataModule(
        ...     data_dir="data/Virtual_Rodent",
        ...     train_ratio=0.6,
        ...     val_ratio=0.2,
        ...     test_ratio=0.2,
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
        # Split configuration
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        # Temporal configuration
        neural_history: int = 750,  # 15 seconds at 50Hz
        pose_history: int = 750,
        pose_horizon: int = 250,  # 5 seconds at 50Hz
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
        # Random seed (for shuffling animals if needed)
        seed: int = 42,
    ):
        """
        Initialize the Per-Animal DataModule.

        Args:
            data_dir: Root directory containing Virtual_Rodent data
            brain_regions: Brain regions to include (None = all)
            animals: Animals to include (None = all)
            train_ratio: Fraction of sessions for training
            val_ratio: Fraction of sessions for validation
            test_ratio: Fraction of sessions for testing
            neural_history: Frames of neural history
            pose_history: Frames of pose history
            pose_horizon: Frames to predict into future
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

        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )

        self.data_dir = Path(data_dir)
        self.brain_regions = brain_regions
        self.animals = animals
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.neural_history = neural_history
        self.pose_history = pose_history
        self.pose_horizon = pose_horizon
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
        self.animal_session_splits: Dict[str, Dict[str, List[str]]] = {}

        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['data_dir'])

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage in ["fit", "validate", None]:
            self._setup_chronological_splits()

        if stage == "test" or stage is None:
            if self.test_dataset is None:
                self._setup_chronological_splits()

    def _setup_chronological_splits(self):
        """Split sessions chronologically per animal."""
        # Discover all sessions
        index = discover_sessions(
            self.data_dir,
            brain_regions=self.brain_regions,
            animals=self.animals,
        )

        if len(index.sessions) == 0:
            raise ValueError(f"No sessions found in {self.data_dir}")

        # Group sessions by animal
        animal_sessions: Dict[str, List] = {}
        for session in index.sessions:
            animal_sessions.setdefault(session.animal, []).append(session)

        print(f"Found sessions for {len(animal_sessions)} animals")

        # Split sessions chronologically for each animal
        train_sessions = []
        val_sessions = []
        test_sessions = []

        self.animal_session_splits = {}

        for animal, sessions in animal_sessions.items():
            # Sort sessions chronologically by session_id
            sessions_sorted = sorted(sessions, key=lambda s: s.session_id)

            n_sessions = len(sessions_sorted)
            n_train = int(n_sessions * self.train_ratio)
            n_val = int(n_sessions * self.val_ratio)

            # Split chronologically
            animal_train = sessions_sorted[:n_train]
            animal_val = sessions_sorted[n_train:n_train + n_val]
            animal_test = sessions_sorted[n_train + n_val:]

            # Store splits for this animal
            self.animal_session_splits[animal] = {
                'train': [s.session_id for s in animal_train],
                'val': [s.session_id for s in animal_val],
                'test': [s.session_id for s in animal_test],
            }

            # Aggregate across animals
            train_sessions.extend([s.session_id for s in animal_train])
            val_sessions.extend([s.session_id for s in animal_val])
            test_sessions.extend([s.session_id for s in animal_test])

            print(
                f"Animal {animal}: {len(animal_train)} train, "
                f"{len(animal_val)} val, {len(animal_test)} test sessions"
            )

        print(f"Total: {len(train_sessions)} train, {len(val_sessions)} val, {len(test_sessions)} test sessions")

        # Create datasets filtered by session IDs
        self.train_dataset = VirtualRodentDataset(
            data_dir=self.data_dir,
            brain_regions=self.brain_regions,
            animals=self.animals,
            sessions=train_sessions,
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

        self.val_dataset = VirtualRodentDataset(
            data_dir=self.data_dir,
            brain_regions=self.brain_regions,
            animals=self.animals,
            sessions=val_sessions,
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

        self.test_dataset = VirtualRodentDataset(
            data_dir=self.data_dir,
            brain_regions=self.brain_regions,
            animals=self.animals,
            sessions=test_sessions,
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

        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Val dataset: {len(self.val_dataset)} samples")
        print(f"Test dataset: {len(self.test_dataset)} samples")

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
            return self.train_dataset.input_dim
        return 0

    @property
    def output_dim(self) -> int:
        """Dimension of pose output features."""
        if self.train_dataset is not None:
            return self.train_dataset.output_dim
        return 0

    def get_animal_splits(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get session splits per animal.

        Returns:
            Dict mapping animal names to dicts of split → session_id lists
        """
        return self.animal_session_splits.copy()

    def __repr__(self) -> str:
        train_len = len(self.train_dataset) if self.train_dataset else 0
        val_len = len(self.val_dataset) if self.val_dataset else 0
        test_len = len(self.test_dataset) if self.test_dataset else 0

        return (
            f"PerAnimalDataModule(\n"
            f"  strategy='per_animal',\n"
            f"  train_samples={train_len},\n"
            f"  val_samples={val_len},\n"
            f"  test_samples={test_len},\n"
            f"  batch_size={self.batch_size},\n"
            f"  animals={len(self.animal_session_splits)}\n"
            f")"
        )
