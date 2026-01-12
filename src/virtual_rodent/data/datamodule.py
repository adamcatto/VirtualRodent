"""
PyTorch Lightning DataModule for Virtual Rodent dataset.

Provides a complete data loading pipeline with train/val/test splits,
automatic batching, and multi-worker data loading.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from virtual_rodent.data.dataset import (
    VirtualRodentDataset,
    VirtualRodentSequenceDataset,
)
from virtual_rodent.data.loader import BRAIN_REGIONS


def _should_pin_memory() -> bool:
    """Determine if pin_memory should be used (not supported on MPS)."""
    if torch.backends.mps.is_available():
        return False
    return torch.cuda.is_available()


class VirtualRodentDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Virtual Rodent data.
    
    Handles train/val/test splitting, data loading, and batching.
    Supports multiple split strategies:
    - random: Random split across all samples
    - session: Split by session (no data leakage between sessions)
    - animal: Split by animal (test generalization to new animals)
    
    Example:
        >>> dm = VirtualRodentDataModule(
        ...     data_dir="data/Virtual_Rodent",
        ...     batch_size=64,
        ...     split_strategy="session",
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
        split_strategy: str = "random",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        # Dataset configuration
        neural_history: int = 10,
        pose_horizon: int = 5,
        pose_type: str = "keypoints",
        flatten_pose: bool = True,
        normalize_neural_method: str = "sqrt",
        normalize_pose_method: str = "center",
        sequence_mode: bool = False,
        sequence_length: int = 50,
        # DataLoader configuration
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: Optional[bool] = None,  # Auto-detect based on backend
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        # Random seed
        seed: int = 42,
    ):
        """
        Initialize the DataModule.
        
        Args:
            data_dir: Root directory containing Virtual_Rodent data
            brain_regions: Brain regions to include (None = all)
            animals: Animals to include (None = all)
            split_strategy: How to split data ("random", "session", "animal")
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            neural_history: Frames of neural history
            pose_horizon: Frames to predict into future
            pose_type: "keypoints" or "qpos"
            flatten_pose: Flatten pose to 1D
            normalize_neural_method: Neural normalization method
            normalize_pose_method: Pose normalization method
            sequence_mode: Use sequence dataset variant
            sequence_length: Length of sequences (if sequence_mode)
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
        self.split_strategy = split_strategy
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.neural_history = neural_history
        self.pose_horizon = pose_horizon
        self.pose_type = pose_type
        self.flatten_pose = flatten_pose
        self.normalize_neural_method = normalize_neural_method
        self.normalize_pose_method = normalize_pose_method
        self.sequence_mode = sequence_mode
        self.sequence_length = sequence_length
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
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['data_dir'])
    
    def _create_dataset(
        self,
        animals: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
    ) -> VirtualRodentDataset:
        """Create a dataset with the configured parameters."""
        DatasetClass = (
            VirtualRodentSequenceDataset if self.sequence_mode 
            else VirtualRodentDataset
        )
        
        kwargs = dict(
            data_dir=self.data_dir,
            brain_regions=self.brain_regions,
            animals=animals or self.animals,
            sessions=sessions,
            neural_history=self.neural_history,
            pose_horizon=self.pose_horizon,
            pose_type=self.pose_type,
            flatten_pose=self.flatten_pose,
            normalize_neural_method=self.normalize_neural_method,
            normalize_pose_method=self.normalize_pose_method,
        )
        
        if self.sequence_mode:
            kwargs['sequence_length'] = self.sequence_length
        
        return DatasetClass(**kwargs)
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            if self.split_strategy == "random":
                self._setup_random_split()
            elif self.split_strategy == "session":
                self._setup_session_split()
            elif self.split_strategy == "animal":
                self._setup_animal_split()
            else:
                raise ValueError(f"Unknown split strategy: {self.split_strategy}")
        
        if stage == "test" or stage is None:
            if self.test_dataset is None:
                # Create test dataset if not already done
                self._setup_random_split()
    
    def _setup_random_split(self):
        """Split data randomly across all samples."""
        full_dataset = self._create_dataset()
        
        n_total = len(full_dataset)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        n_test = n_total - n_train - n_val
        
        generator = torch.Generator().manual_seed(self.seed)
        train_idx, val_idx, test_idx = random_split(
            range(n_total),
            [n_train, n_val, n_test],
            generator=generator,
        )
        
        self.train_dataset = Subset(full_dataset, train_idx.indices)
        self.val_dataset = Subset(full_dataset, val_idx.indices)
        self.test_dataset = Subset(full_dataset, test_idx.indices)
    
    def _setup_session_split(self):
        """Split data by session (no session appears in multiple splits)."""
        full_dataset = self._create_dataset()
        sessions = [s.session_id for s in full_dataset.index.sessions]
        
        np.random.seed(self.seed)
        np.random.shuffle(sessions)
        
        n_sessions = len(sessions)
        n_train = int(n_sessions * self.train_ratio)
        n_val = int(n_sessions * self.val_ratio)
        
        train_sessions = sessions[:n_train]
        val_sessions = sessions[n_train:n_train + n_val]
        test_sessions = sessions[n_train + n_val:]
        
        self.train_dataset = self._create_dataset(sessions=train_sessions)
        self.val_dataset = self._create_dataset(sessions=val_sessions)
        self.test_dataset = self._create_dataset(sessions=test_sessions)
    
    def _setup_animal_split(self):
        """Split data by animal (test generalization to new animals)."""
        # Get all available animals
        all_animals = []
        for region in (self.brain_regions or BRAIN_REGIONS.keys()):
            all_animals.extend(BRAIN_REGIONS.get(region, []))
        
        if self.animals is not None:
            all_animals = [a for a in all_animals if a in self.animals]
        # HARD CODE JUST ONE ANIMAL FOR TESTING
        all_animals = all_animals[0]
        
        np.random.seed(self.seed)
        np.random.shuffle(all_animals)
        
        n_animals = len(all_animals)
        n_train = max(1, int(n_animals * self.train_ratio))
        n_val = max(1, int(n_animals * self.val_ratio))
        
        train_animals = all_animals[:n_train]
        val_animals = all_animals[n_train:n_train + n_val]
        test_animals = all_animals[n_train + n_val:]
        
        if not test_animals:
            test_animals = val_animals  # Use val animals if not enough
        
        self.train_dataset = self._create_dataset(animals=train_animals)
        self.val_dataset = self._create_dataset(animals=val_animals)
        self.test_dataset = self._create_dataset(animals=test_animals)
    
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
    
    def get_sample_batch(self, n: int = 4) -> Dict[str, torch.Tensor]:
        """Get a sample batch for debugging or visualization."""
        if self.train_dataset is None:
            self.setup("fit")
        
        samples = [self.train_dataset[i] for i in range(min(n, len(self.train_dataset)))]
        neural = torch.stack([s[0] for s in samples])
        pose = torch.stack([s[1] for s in samples])
        
        return {'neural': neural, 'pose': pose}
    
    def __repr__(self) -> str:
        train_len = len(self.train_dataset) if self.train_dataset else 0
        val_len = len(self.val_dataset) if self.val_dataset else 0
        test_len = len(self.test_dataset) if self.test_dataset else 0
        
        return (
            f"VirtualRodentDataModule(\n"
            f"  train_samples={train_len},\n"
            f"  val_samples={val_len},\n"
            f"  test_samples={test_len},\n"
            f"  batch_size={self.batch_size},\n"
            f"  split_strategy='{self.split_strategy}'\n"
            f")"
        )
