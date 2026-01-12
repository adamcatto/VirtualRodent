"""
PyTorch Dataset for Virtual Rodent neural-pose pairs.

Provides flexible access to neural signals and pose data for training
neural decoding models.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from virtual_rodent.data.loader import (
    SessionLoader,
    DatasetIndex,
    discover_sessions,
    SAMPLING_RATE,
    NUM_KEYPOINTS,
)
from virtual_rodent.data.preprocessing import (
    normalize_neural,
    normalize_pose,
    smooth_neural,
    flatten_keypoints,
)


class VirtualRodentDataset(Dataset):
    """
    PyTorch Dataset for Virtual Rodent neural-pose pairs.
    
    Provides samples of (neural_input, pose_target) pairs for training
    neural decoding models. Supports:
    - Flexible input/output configurations
    - Temporal context windows
    - Future prediction horizons
    - Multiple normalization strategies
    - Lazy or preloaded data access
    
    Example:
        >>> dataset = VirtualRodentDataset(
        ...     data_dir="data/Virtual_Rodent",
        ...     brain_regions=["motor_cortex"],
        ...     neural_history=10,  # 200ms of neural history
        ...     pose_horizon=5,     # Predict 100ms into future
        ... )
        >>> neural, pose = dataset[0]
        >>> print(neural.shape, pose.shape)
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        brain_regions: Optional[List[str]] = None,
        animals: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        # Temporal configuration
        neural_history: int = 1,
        pose_history: int = 0,
        pose_horizon: int = 1,
        # Data selection
        pose_type: str = "keypoints",  # "keypoints" or "qpos"
        flatten_pose: bool = True,
        include_behavior: bool = False,
        # Preprocessing
        normalize_neural_method: Optional[str] = "sqrt",
        normalize_pose_method: Optional[str] = "center",
        smooth_neural_sigma: float = 0.0,
        # Loading options
        preload: bool = False,
        cache_sessions: int = 4,
        # Transforms
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing Virtual_Rodent data
            brain_regions: List of brain regions to include (None = all)
            animals: List of animals to include (None = all)
            sessions: List of session IDs to include (None = all)
            neural_history: Number of past frames to include in neural input
            pose_history: Number of past frames to include in pose input
            pose_horizon: Number of future frames to predict
            pose_type: Type of pose representation ("keypoints" or "qpos")
            flatten_pose: Whether to flatten pose to 1D
            include_behavior: Whether to include behavior labels
            normalize_neural_method: Normalization for neural data
            normalize_pose_method: Normalization for pose data
            smooth_neural_sigma: Gaussian smoothing sigma for neural data
            preload: Whether to preload all data into memory
            cache_sessions: Number of sessions to keep in cache
            transform: Optional transform for inputs
            target_transform: Optional transform for targets
        """
        self.data_dir = Path(data_dir)
        self.neural_history = neural_history
        self.pose_history = pose_history
        self.pose_horizon = pose_horizon
        self.pose_type = pose_type
        self.flatten_pose = flatten_pose
        self.include_behavior = include_behavior
        self.normalize_neural_method = normalize_neural_method
        self.normalize_pose_method = normalize_pose_method
        self.smooth_neural_sigma = smooth_neural_sigma
        self.preload = preload
        self.cache_sessions = cache_sessions
        self.transform = transform
        self.target_transform = target_transform
        
        # Discover and index sessions
        self.index = discover_sessions(
            data_dir=data_dir,
            brain_regions=brain_regions,
            animals=animals,
        )
        
        # Filter sessions if specified
        if sessions is not None:
            self.index.sessions = [
                s for s in self.index.sessions if s.session_id in sessions
            ]
            self.index._rebuild_index()
        
        # Session loaders cache
        self._session_cache: Dict[int, SessionLoader] = {}
        self._cache_order: List[int] = []
        
        # Preload all sessions if requested
        if preload:
            for i, session in enumerate(self.index.sessions):
                self._session_cache[i] = SessionLoader(
                    session.file_path, preload=True
                )
        
        # Compute valid indices (accounting for history and horizon)
        self._compute_valid_indices()
        
        # Compute max neurons for padding (to handle variable neuron counts)
        self._max_neurons = max(s.num_neurons for s in self.index.sessions)
        
        # Compute normalization statistics
        self._neural_stats: Optional[Tuple] = None
        self._pose_stats: Optional[Tuple] = None
    
    def _compute_valid_indices(self):
        """Compute valid sample indices accounting for temporal context."""
        margin_before = max(self.neural_history, self.pose_history) - 1
        margin_after = self.pose_horizon - 1
        
        valid_indices = []
        for session_idx, session in enumerate(self.index.sessions):
            num_valid = session.num_frames - margin_before - margin_after
            if num_valid > 0:
                session_start = self.index.cumulative_frames[session_idx]
                for i in range(num_valid):
                    global_idx = session_start + margin_before + i
                    valid_indices.append(global_idx)
        
        self._valid_indices = np.array(valid_indices)
    
    def _get_session_loader(self, session_idx: int) -> SessionLoader:
        """Get or create a session loader with LRU caching."""
        if session_idx in self._session_cache:
            return self._session_cache[session_idx]
        
        # Load new session
        session = self.index.sessions[session_idx]
        loader = SessionLoader(session.file_path, preload=self.preload)
        
        # Manage cache
        if len(self._session_cache) >= self.cache_sessions:
            # Remove oldest
            oldest_idx = self._cache_order.pop(0)
            if oldest_idx in self._session_cache:
                self._session_cache[oldest_idx].close()
                del self._session_cache[oldest_idx]
        
        self._session_cache[session_idx] = loader
        self._cache_order.append(session_idx)
        
        return loader
    
    def __len__(self) -> int:
        return len(self._valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample (neural_input, pose_history, pose_target).

        Returns:
            neural_input: Tensor of shape (neural_history, num_neurons)
            pose_history: Tensor of shape (pose_history, pose_dim)
            pose_target: Tensor of shape (pose_horizon, pose_dim)
        """
        global_idx = self._valid_indices[idx]
        session_idx, frame_idx = self.index.get_session_and_frame(global_idx)

        loader = self._get_session_loader(session_idx)

        # Get neural history
        neural_start = frame_idx - self.neural_history + 1
        neural_end = frame_idx + 1
        data = loader.get_sequence(neural_start, neural_end)
        neural = data['spike_counts'].astype(np.float32)

        # Get pose history (same time range as neural)
        if self.pose_type == "keypoints":
            pose_hist = data['keypoints'].astype(np.float32)
        else:
            pose_hist = data['qpos'].astype(np.float32)

        # Get pose target (future frames)
        pose_start = frame_idx + 1
        pose_end = frame_idx + 1 + self.pose_horizon
        future_data = loader.get_sequence(pose_start, pose_end)

        if self.pose_type == "keypoints":
            pose_target = future_data['keypoints'].astype(np.float32)
        else:
            pose_target = future_data['qpos'].astype(np.float32)

        # Apply preprocessing
        if self.smooth_neural_sigma > 0 and neural.shape[0] > 1:
            neural = smooth_neural(neural, sigma=self.smooth_neural_sigma)

        if self.normalize_neural_method is not None:
            neural = normalize_neural(neural, method=self.normalize_neural_method)
            if isinstance(neural, tuple):
                neural = neural[0]

        if self.normalize_pose_method is not None:
            pose_hist = normalize_pose(pose_hist, method=self.normalize_pose_method)
            if isinstance(pose_hist, tuple):
                pose_hist = pose_hist[0]
            pose_target = normalize_pose(pose_target, method=self.normalize_pose_method)
            if isinstance(pose_target, tuple):
                pose_target = pose_target[0]

        # Flatten pose if requested
        if self.flatten_pose and self.pose_type == "keypoints":
            if pose_hist.ndim == 3:
                pose_hist = flatten_keypoints(pose_hist)
            if pose_target.ndim == 3:
                pose_target = flatten_keypoints(pose_target)

        # Squeeze if single time step
        if self.neural_history == 1:
            neural = neural.squeeze(0)
        if self.pose_horizon == 1:
            pose_target = pose_target.squeeze(0)

        # Convert to tensors
        neural = torch.from_numpy(neural)
        pose_hist = torch.from_numpy(pose_hist)
        pose_target = torch.from_numpy(pose_target)

        # Pad neural to max_neurons for consistent tensor sizes across sessions
        if neural.ndim == 1:
            # Shape: (num_neurons,) -> (max_neurons,)
            if neural.shape[0] < self._max_neurons:
                padding = torch.zeros(self._max_neurons - neural.shape[0], dtype=neural.dtype)
                neural = torch.cat([neural, padding], dim=0)
        else:
            # Shape: (neural_history, num_neurons) -> (neural_history, max_neurons)
            if neural.shape[-1] < self._max_neurons:
                padding = torch.zeros(neural.shape[0], self._max_neurons - neural.shape[-1], dtype=neural.dtype)
                neural = torch.cat([neural, padding], dim=1)

        # Apply transforms
        if self.transform is not None:
            neural = self.transform(neural)
        if self.target_transform is not None:
            pose_target = self.target_transform(pose_target)

        return neural, pose_hist, pose_target
    
    def get_sample_with_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample with full metadata.
        
        Returns:
            Dict with 'neural', 'pose', 'behavior', 'session', 'frame_idx', etc.
        """
        global_idx = self._valid_indices[idx]
        session_idx, frame_idx = self.index.get_session_and_frame(global_idx)
        
        loader = self._get_session_loader(session_idx)
        session = self.index.sessions[session_idx]
        
        neural, pose = self[idx]
        
        # Get behavior label
        data = loader.get_frame(frame_idx)
        behavior = data['behavior']
        
        return {
            'neural': neural,
            'pose': pose,
            'behavior': int(behavior),
            'session_id': session.session_id,
            'animal': session.animal,
            'brain_region': session.brain_region,
            'frame_idx': frame_idx,
            'global_idx': global_idx,
        }
    
    @property
    def input_dim(self) -> int:
        """Dimension of neural input (max across all sessions)."""
        return self._max_neurons
    
    @property
    def max_neurons(self) -> int:
        """Maximum number of neurons across all sessions."""
        return self._max_neurons
    
    @property
    def output_dim(self) -> int:
        """Dimension of pose output."""
        if self.pose_type == "keypoints":
            return NUM_KEYPOINTS * 3 if self.flatten_pose else (3, NUM_KEYPOINTS)
        else:
            return 74  # qpos dimensions
    
    def get_all_neurons_count(self) -> Dict[str, int]:
        """Get neuron count per animal."""
        return {
            session.animal: session.num_neurons
            for session in self.index.sessions
        }
    
    def __repr__(self) -> str:
        return (
            f"VirtualRodentDataset("
            f"samples={len(self)}, "
            f"sessions={len(self.index.sessions)}, "
            f"neural_history={self.neural_history}, "
            f"pose_horizon={self.pose_horizon})"
        )


class VirtualRodentSequenceDataset(VirtualRodentDataset):
    """
    Dataset variant that returns full sequences for sequence-to-sequence models.
    
    Useful for RNN, Transformer, or other sequence models that need aligned
    input-output sequences.
    """
    
    def __init__(
        self,
        *args,
        sequence_length: int = 50,
        overlap: int = 0,
        **kwargs,
    ):
        """
        Initialize the sequence dataset.
        
        Args:
            sequence_length: Length of each sequence
            overlap: Number of overlapping frames between sequences
            *args, **kwargs: Passed to VirtualRodentDataset
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        
        super().__init__(*args, **kwargs)
    
    def _compute_valid_indices(self):
        """Compute valid sequence start indices."""
        stride = self.sequence_length - self.overlap
        
        valid_indices = []
        for session_idx, session in enumerate(self.index.sessions):
            num_sequences = (session.num_frames - self.sequence_length) // stride + 1
            session_start = self.index.cumulative_frames[session_idx]
            for i in range(num_sequences):
                global_idx = session_start + i * stride
                valid_indices.append(global_idx)
        
        self._valid_indices = np.array(valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a full sequence."""
        global_idx = self._valid_indices[idx]
        session_idx, frame_idx = self.index.get_session_and_frame(global_idx)
        
        loader = self._get_session_loader(session_idx)
        
        # Get full sequence
        data = loader.get_sequence(frame_idx, frame_idx + self.sequence_length)
        
        neural = data['spike_counts'].astype(np.float32)
        
        if self.pose_type == "keypoints":
            pose = data['keypoints'].astype(np.float32)
        else:
            pose = data['qpos'].astype(np.float32)
        
        # Apply preprocessing
        if self.normalize_neural_method is not None:
            neural = normalize_neural(neural, method=self.normalize_neural_method)
            if isinstance(neural, tuple):
                neural = neural[0]
        
        if self.normalize_pose_method is not None:
            pose = normalize_pose(pose, method=self.normalize_pose_method)
            if isinstance(pose, tuple):
                pose = pose[0]
        
        if self.flatten_pose and self.pose_type == "keypoints":
            pose = flatten_keypoints(pose)
        
        neural = torch.from_numpy(neural)
        pose = torch.from_numpy(pose)
        
        if self.transform is not None:
            neural = self.transform(neural)
        if self.target_transform is not None:
            pose = self.target_transform(pose)
        
        return neural, pose
