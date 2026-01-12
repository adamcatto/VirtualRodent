"""
Low-level data loading utilities for Virtual Rodent HDF5 files.

This module provides efficient loading and indexing of the dataset without
loading all data into memory at once.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import h5py
import numpy as np


# Constants based on dataset analysis
SAMPLING_RATE = 50  # Hz (20ms bins)
NUM_KEYPOINTS = 23
KEYPOINT_DIMS = 3  # x, y, z
QPOS_DIMS = 74  # Full skeletal model DoF

# Brain regions and their animals
BRAIN_REGIONS = {
    "DLS": ["art", "bud", "coltrane"],
    "motor_cortex": ["duke", "freddie", "gerry"],
}

# Keypoint names (based on typical rat skeleton tracking)
KEYPOINT_NAMES = [
    "nose", "left_ear", "right_ear", "neck",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
    "spine_upper", "spine_mid", "spine_lower",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
]


@dataclass
class SessionMetadata:
    """Metadata for a single recording session."""
    
    file_path: Path
    brain_region: str
    animal: str
    session_id: str
    num_frames: int
    num_neurons: int
    duration_sec: float
    
    @property
    def duration_min(self) -> float:
        return self.duration_sec / 60


@dataclass
class DatasetIndex:
    """
    Index structure for efficient access to samples across multiple sessions.
    
    This allows O(1) lookup of which session and frame index corresponds to
    a global sample index.
    """
    
    sessions: List[SessionMetadata] = field(default_factory=list)
    cumulative_frames: np.ndarray = field(default_factory=lambda: np.array([0]))
    
    def __post_init__(self):
        if len(self.sessions) > 0:
            self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the cumulative frame index."""
        frames = [0] + [s.num_frames for s in self.sessions]
        self.cumulative_frames = np.cumsum(frames)
    
    def add_session(self, metadata: SessionMetadata):
        """Add a session to the index."""
        self.sessions.append(metadata)
        self._rebuild_index()
    
    def __len__(self) -> int:
        """Total number of frames across all sessions."""
        return int(self.cumulative_frames[-1])
    
    def get_session_and_frame(self, global_idx: int) -> Tuple[int, int]:
        """
        Convert a global index to (session_idx, frame_idx).
        
        Args:
            global_idx: Global frame index across all sessions
            
        Returns:
            Tuple of (session_index, local_frame_index)
        """
        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(f"Global index {global_idx} out of range [0, {len(self)})")
        
        session_idx = np.searchsorted(self.cumulative_frames[1:], global_idx, side='right')
        frame_idx = global_idx - self.cumulative_frames[session_idx]
        return int(session_idx), int(frame_idx)
    
    def get_global_index(self, session_idx: int, frame_idx: int) -> int:
        """Convert (session_idx, frame_idx) to global index."""
        return int(self.cumulative_frames[session_idx] + frame_idx)
    
    def get_sessions_by_animal(self, animal: str) -> List[SessionMetadata]:
        """Get all sessions for a specific animal."""
        return [s for s in self.sessions if s.animal == animal]
    
    def get_sessions_by_region(self, region: str) -> List[SessionMetadata]:
        """Get all sessions for a specific brain region."""
        return [s for s in self.sessions if s.brain_region == region]
    
    @property
    def total_duration_hours(self) -> float:
        """Total recording duration in hours."""
        return sum(s.duration_sec for s in self.sessions) / 3600
    
    @property
    def total_neurons(self) -> int:
        """Total number of unique neurons (max per animal for now)."""
        return max(s.num_neurons for s in self.sessions) if self.sessions else 0


class SessionLoader:
    """
    Efficient loader for a single HDF5 session file.
    
    Supports lazy loading and memory-mapped access for large files.
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        preload: bool = False,
        cache_size: int = 10000,
    ):
        """
        Initialize the session loader.
        
        Args:
            file_path: Path to the HDF5 file
            preload: If True, load all data into memory
            cache_size: Number of frames to cache (if not preloading)
        """
        self.file_path = Path(file_path)
        self.preload = preload
        self.cache_size = cache_size
        
        # Parse metadata from filename
        self.session_id = self.file_path.stem
        self.animal = self.file_path.parent.name
        self.brain_region = self.file_path.parent.parent.name
        
        # Load metadata
        self._load_metadata()
        
        # Preload data if requested
        self._data_cache: Optional[Dict[str, np.ndarray]] = None
        if preload:
            self._preload_data()
    
    def _load_metadata(self):
        """Load metadata from the HDF5 file."""
        with h5py.File(self.file_path, 'r') as f:
            self.num_frames = f['ephys/spike_counts'].shape[0]
            self.num_neurons = f['ephys/spike_counts'].shape[1]
            self.duration_sec = self.num_frames / SAMPLING_RATE
            
            # Verify expected structure
            assert 'pose/keypoints' in f, "Missing pose/keypoints"
            assert 'pose/qpos' in f, "Missing pose/qpos"
            assert 'behavior/motion_mapper' in f, "Missing behavior/motion_mapper"
    
    def _preload_data(self):
        """Preload all data into memory."""
        with h5py.File(self.file_path, 'r') as f:
            self._data_cache = {
                'spike_counts': f['ephys/spike_counts'][:],
                'keypoints': f['pose/keypoints'][:],
                'qpos': f['pose/qpos'][:],
                'behavior': f['behavior/motion_mapper'][:],
            }
    
    def get_metadata(self) -> SessionMetadata:
        """Get session metadata."""
        return SessionMetadata(
            file_path=self.file_path,
            brain_region=self.brain_region,
            animal=self.animal,
            session_id=self.session_id,
            num_frames=self.num_frames,
            num_neurons=self.num_neurons,
            duration_sec=self.duration_sec,
        )
    
    def get_frame(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single frame of data.
        
        Args:
            idx: Frame index
            
        Returns:
            Dict with keys: 'spike_counts', 'keypoints', 'qpos', 'behavior'
        """
        if self._data_cache is not None:
            return {
                'spike_counts': self._data_cache['spike_counts'][idx],
                'keypoints': self._data_cache['keypoints'][idx],
                'qpos': self._data_cache['qpos'][idx],
                'behavior': self._data_cache['behavior'][idx],
            }
        else:
            with h5py.File(self.file_path, 'r') as f:
                return {
                    'spike_counts': f['ephys/spike_counts'][idx],
                    'keypoints': f['pose/keypoints'][idx],
                    'qpos': f['pose/qpos'][idx],
                    'behavior': f['behavior/motion_mapper'][idx],
                }
    
    def get_sequence(
        self,
        start_idx: int,
        end_idx: int,
    ) -> Dict[str, np.ndarray]:
        """
        Get a sequence of frames.
        
        Args:
            start_idx: Start frame index (inclusive)
            end_idx: End frame index (exclusive)
            
        Returns:
            Dict with keys: 'spike_counts', 'keypoints', 'qpos', 'behavior'
        """
        if self._data_cache is not None:
            return {
                'spike_counts': self._data_cache['spike_counts'][start_idx:end_idx],
                'keypoints': self._data_cache['keypoints'][start_idx:end_idx],
                'qpos': self._data_cache['qpos'][start_idx:end_idx],
                'behavior': self._data_cache['behavior'][start_idx:end_idx],
            }
        else:
            with h5py.File(self.file_path, 'r') as f:
                return {
                    'spike_counts': f['ephys/spike_counts'][start_idx:end_idx],
                    'keypoints': f['pose/keypoints'][start_idx:end_idx],
                    'qpos': f['pose/qpos'][start_idx:end_idx],
                    'behavior': f['behavior/motion_mapper'][start_idx:end_idx],
                }
    
    def __len__(self) -> int:
        return self.num_frames
    
    def close(self):
        """Release cached data."""
        self._data_cache = None


def discover_sessions(
    data_dir: Union[str, Path],
    brain_regions: Optional[List[str]] = None,
    animals: Optional[List[str]] = None,
) -> DatasetIndex:
    """
    Discover all sessions in the data directory.
    
    Args:
        data_dir: Root data directory (e.g., data/Virtual_Rodent)
        brain_regions: Filter by brain regions (e.g., ['DLS', 'motor_cortex'])
        animals: Filter by animal names
        
    Returns:
        DatasetIndex with all discovered sessions
    """
    data_dir = Path(data_dir)
    index = DatasetIndex()
    
    # Default to all brain regions
    if brain_regions is None:
        brain_regions = list(BRAIN_REGIONS.keys())
    
    for region in brain_regions:
        region_dir = data_dir / region
        if not region_dir.exists():
            continue
        
        # Get animals for this region
        region_animals = BRAIN_REGIONS.get(region, [])
        if animals is not None:
            region_animals = [a for a in region_animals if a in animals]
        
        for animal in region_animals:
            animal_dir = region_dir / animal
            if not animal_dir.exists():
                continue
            
            # Find all HDF5 files
            for h5_file in sorted(animal_dir.glob("*.h5")):
                if h5_file.name.startswith("._"):
                    continue
                
                try:
                    loader = SessionLoader(h5_file)
                    index.add_session(loader.get_metadata())
                except Exception as e:
                    print(f"Warning: Could not load {h5_file}: {e}")
    
    return index
