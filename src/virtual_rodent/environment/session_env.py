"""
Session environment class for managing full session data with temporal splits.

This module provides the SessionEnvironment class which wraps SessionLoader
and adds temporal split logic for per-session evaluation strategies.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from virtual_rodent.data.loader import SessionLoader, SessionMetadata


class SessionEnvironment:
    """
    Manages full session data with temporal split support.

    This class wraps SessionLoader and provides:
    - Temporal split computation (60% train / 1% gap / 19% val / 1% gap / 19% test)
    - Split-aware data access
    - Metadata tracking with split boundaries
    - Validation of sample indices against split boundaries

    The split strategy prevents temporal leakage by adding gap regions between
    train/val and val/test splits.

    Example:
        >>> from virtual_rodent.data.loader import SessionLoader
        >>> loader = SessionLoader("data/Virtual_Rodent/motor_cortex/duke/session_001.h5")
        >>> env = SessionEnvironment(loader)
        >>> train_indices = env.get_split_indices('train')
        >>> print(f"Training frames: {len(train_indices)}")
    """

    def __init__(
        self,
        session_loader: SessionLoader,
        split_ratios: Tuple[float, float, float, float, float] = (0.6, 0.01, 0.19, 0.01, 0.19),
    ):
        """
        Initialize the session environment.

        Args:
            session_loader: SessionLoader instance for data access
            split_ratios: Tuple of (train, gap1, val, gap2, test) ratios.
                         Default is (0.6, 0.01, 0.19, 0.01, 0.19) which sums to 1.0.
        """
        self.loader = session_loader
        self.split_ratios = split_ratios

        # Validate split ratios
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

        # Compute split boundaries
        self._compute_splits()

    def _compute_splits(self):
        """
        Compute frame indices for train/gap1/val/gap2/test splits.

        Creates self.splits dict with (start_idx, end_idx) for each split.
        """
        total_frames = self.loader.num_frames

        # Compute cumulative split points
        cumulative = np.cumsum([0] + list(self.split_ratios))
        split_points = (cumulative * total_frames).astype(int)

        self.splits = {
            'train': (split_points[0], split_points[1]),
            'gap1': (split_points[1], split_points[2]),
            'val': (split_points[2], split_points[3]),
            'gap2': (split_points[3], split_points[4]),
            'test': (split_points[4], split_points[5]),
        }

        # Store usable splits (exclude gaps)
        self.usable_splits = ['train', 'val', 'test']

    def get_split_indices(self, split: str) -> np.ndarray:
        """
        Get frame indices for a specific split.

        Args:
            split: Split name ('train', 'val', 'test', 'gap1', 'gap2')

        Returns:
            Array of frame indices for the split

        Raises:
            ValueError: If split name is invalid
        """
        if split not in self.splits:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {list(self.splits.keys())}"
            )

        start, end = self.splits[split]
        return np.arange(start, end)

    def get_split_boundaries(self) -> Dict[str, Tuple[int, int]]:
        """
        Get (start, end) frame indices for all splits.

        Returns:
            Dict mapping split names to (start_idx, end_idx) tuples
        """
        return self.splits.copy()

    def is_valid_sample(
        self,
        center_frame: int,
        split: str,
        neural_history: int = 1,
        pose_history: int = 0,
        pose_horizon: int = 1,
    ) -> bool:
        """
        Check if a sample is valid for a given split.

        A sample is valid if:
        1. It falls within the split boundaries
        2. All frames needed (history + future) are within the split
        3. No temporal leakage into adjacent splits

        Args:
            center_frame: Center frame index for the sample
            split: Split name to check against
            neural_history: Number of past frames needed for neural input
            pose_history: Number of past frames needed for pose input
            pose_horizon: Number of future frames needed for prediction

        Returns:
            True if sample is valid for the split, False otherwise
        """
        if split not in self.splits:
            return False

        start, end = self.splits[split]

        # Compute required frame range
        margin_before = max(neural_history, pose_history) - 1
        margin_after = pose_horizon - 1

        frame_start = center_frame - margin_before
        frame_end = center_frame + margin_after + 1

        # Check if all required frames are within split boundaries
        return frame_start >= start and frame_end <= end

    def get_valid_sample_indices(
        self,
        split: str,
        neural_history: int = 1,
        pose_history: int = 0,
        pose_horizon: int = 1,
    ) -> np.ndarray:
        """
        Get all valid center frame indices for a split respecting temporal margins.

        This is a convenience method that filters split indices to only include
        frames that can serve as valid sample centers given the history/horizon requirements.

        Args:
            split: Split name ('train', 'val', 'test')
            neural_history: Number of past frames needed for neural input
            pose_history: Number of past frames needed for pose input
            pose_horizon: Number of future frames needed for prediction

        Returns:
            Array of valid center frame indices
        """
        split_indices = self.get_split_indices(split)
        start, end = self.splits[split]

        # Compute margins
        margin_before = max(neural_history, pose_history) - 1
        margin_after = pose_horizon - 1

        # Filter indices
        valid_indices = split_indices[
            (split_indices >= start + margin_before) &
            (split_indices < end - margin_after)
        ]

        return valid_indices

    def get_metadata(self) -> Dict:
        """
        Get session metadata including split information.

        Returns:
            Dict with session metadata and split boundaries
        """
        session_meta = self.loader.get_metadata()

        return {
            'file_path': str(session_meta.file_path),
            'brain_region': session_meta.brain_region,
            'animal': session_meta.animal,
            'session_id': session_meta.session_id,
            'num_frames': session_meta.num_frames,
            'num_neurons': session_meta.num_neurons,
            'duration_sec': session_meta.duration_sec,
            'duration_min': session_meta.duration_min,
            'splits': self.splits,
            'split_ratios': self.split_ratios,
        }

    # Delegate data access methods to SessionLoader

    def get_frame(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single frame of data.

        Delegates to SessionLoader.get_frame().

        Args:
            idx: Frame index

        Returns:
            Dict with keys: 'spike_counts', 'keypoints', 'qpos', 'behavior'
        """
        return self.loader.get_frame(idx)

    def get_sequence(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """
        Get a sequence of frames.

        Delegates to SessionLoader.get_sequence().

        Args:
            start_idx: Start frame index (inclusive)
            end_idx: End frame index (exclusive)

        Returns:
            Dict with keys: 'spike_counts', 'keypoints', 'qpos', 'behavior'
        """
        return self.loader.get_sequence(start_idx, end_idx)

    def __len__(self) -> int:
        """Total number of frames in the session."""
        return self.loader.num_frames

    def __repr__(self) -> str:
        return (
            f"SessionEnvironment("
            f"session_id='{self.loader.session_id}', "
            f"animal='{self.loader.animal}', "
            f"num_frames={self.loader.num_frames}, "
            f"splits={len(self.usable_splits)})"
        )
