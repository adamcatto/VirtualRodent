"""
Tests for the Virtual Rodent data pipeline.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

# Skip tests if data is not available
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "Virtual_Rodent"
SKIP_NO_DATA = not DATA_DIR.exists()


class TestLoader:
    """Tests for data loader utilities."""
    
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
    def test_discover_sessions(self):
        from virtual_rodent.data.loader import discover_sessions
        
        index = discover_sessions(DATA_DIR)
        assert len(index.sessions) > 0
        assert len(index) > 0
    
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
    def test_session_loader(self):
        from virtual_rodent.data.loader import discover_sessions, SessionLoader
        
        index = discover_sessions(DATA_DIR)
        session = index.sessions[0]
        
        loader = SessionLoader(session.file_path)
        assert loader.num_frames > 0
        assert loader.num_neurons > 0
        
        frame = loader.get_frame(0)
        assert 'spike_counts' in frame
        assert 'keypoints' in frame
        assert 'qpos' in frame
        assert 'behavior' in frame
    
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
    def test_dataset_index(self):
        from virtual_rodent.data.loader import discover_sessions
        
        index = discover_sessions(DATA_DIR)
        
        # Test global to local index conversion
        session_idx, frame_idx = index.get_session_and_frame(0)
        assert session_idx == 0
        assert frame_idx == 0
        
        # Test reverse conversion
        global_idx = index.get_global_index(session_idx, frame_idx)
        assert global_idx == 0


class TestPreprocessing:
    """Tests for preprocessing functions."""
    
    def test_normalize_neural_zscore(self):
        from virtual_rodent.data.preprocessing import normalize_neural
        
        data = np.random.randn(100, 50).astype(np.float32)
        normalized, stats = normalize_neural(data, method="zscore")
        
        # Check output shape
        assert normalized.shape == data.shape
        
        # Check normalization (mean ~0, std ~1 per neuron)
        assert np.abs(normalized.mean(axis=0)).max() < 0.1
        assert np.abs(normalized.std(axis=0) - 1).max() < 0.1
    
    def test_normalize_neural_sqrt(self):
        from virtual_rodent.data.preprocessing import normalize_neural
        
        data = np.random.poisson(5, size=(100, 50)).astype(np.float32)
        normalized = normalize_neural(data, method="sqrt")
        
        assert normalized.shape == data.shape
        assert np.all(normalized >= 0)
    
    def test_normalize_pose_center(self):
        from virtual_rodent.data.preprocessing import normalize_pose
        
        # Create fake pose data (T, 3, 23)
        pose = np.random.randn(100, 3, 23).astype(np.float32)
        normalized = normalize_pose(pose, method="center")
        
        # Reference keypoint should be at origin
        assert np.allclose(normalized[:, :, 14], 0, atol=1e-6)
    
    def test_flatten_unflatten_keypoints(self):
        from virtual_rodent.data.preprocessing import flatten_keypoints, unflatten_keypoints
        
        pose = np.random.randn(100, 3, 23).astype(np.float32)
        flat = flatten_keypoints(pose)
        assert flat.shape == (100, 69)
        
        unflat = unflatten_keypoints(flat)
        assert unflat.shape == (100, 3, 23)
        assert np.allclose(pose, unflat)
    
    def test_smooth_neural(self):
        from virtual_rodent.data.preprocessing import smooth_neural
        
        data = np.random.randn(100, 50).astype(np.float32)
        
        smoothed_gaussian = smooth_neural(data, method="gaussian", sigma=2.0)
        assert smoothed_gaussian.shape == data.shape
        
        smoothed_boxcar = smooth_neural(data, method="boxcar", window_size=5)
        assert smoothed_boxcar.shape == data.shape
    
    def test_create_temporal_windows(self):
        from virtual_rodent.data.preprocessing import create_temporal_windows
        
        data = np.random.randn(100, 10).astype(np.float32)
        windows = create_temporal_windows(data, window_size=5, stride=1)
        
        assert windows.shape[1] == 5  # Window size
        assert windows.shape[2] == 10  # Feature dim


class TestDataset:
    """Tests for PyTorch Dataset."""
    
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
    def test_dataset_creation(self):
        from virtual_rodent.data.dataset import VirtualRodentDataset
        
        dataset = VirtualRodentDataset(
            data_dir=DATA_DIR,
            neural_history=5,
            pose_horizon=1,
        )
        
        assert len(dataset) > 0
        assert dataset.input_dim > 0
        assert dataset.output_dim > 0
    
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
    def test_dataset_getitem(self):
        from virtual_rodent.data.dataset import VirtualRodentDataset
        
        dataset = VirtualRodentDataset(
            data_dir=DATA_DIR,
            neural_history=5,
            pose_horizon=1,
            flatten_pose=True,
        )
        
        neural, pose = dataset[0]
        
        assert isinstance(neural, torch.Tensor)
        assert isinstance(pose, torch.Tensor)
        assert neural.dtype == torch.float32
        assert pose.dtype == torch.float32
    
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
    def test_dataset_shapes(self):
        from virtual_rodent.data.dataset import VirtualRodentDataset
        
        neural_history = 10
        pose_horizon = 5
        
        dataset = VirtualRodentDataset(
            data_dir=DATA_DIR,
            neural_history=neural_history,
            pose_horizon=pose_horizon,
            flatten_pose=True,
        )
        
        neural, pose = dataset[0]
        
        # Neural: (history, neurons) if history > 1
        assert neural.shape[0] == neural_history
        
        # Pose: (horizon, 69) if horizon > 1
        assert pose.shape[0] == pose_horizon
        assert pose.shape[1] == 69  # 23 keypoints * 3 coords
    
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
    def test_dataset_metadata(self):
        from virtual_rodent.data.dataset import VirtualRodentDataset
        
        dataset = VirtualRodentDataset(data_dir=DATA_DIR)
        
        sample = dataset.get_sample_with_metadata(0)
        
        assert 'neural' in sample
        assert 'pose' in sample
        assert 'behavior' in sample
        assert 'animal' in sample
        assert 'brain_region' in sample


class TestDataModule:
    """Tests for PyTorch Lightning DataModule."""
    
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
    def test_datamodule_setup(self):
        from virtual_rodent.data.datamodule import VirtualRodentDataModule
        
        dm = VirtualRodentDataModule(
            data_dir=DATA_DIR,
            batch_size=32,
            num_workers=0,  # Use main process for testing
        )
        
        dm.setup("fit")
        
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
    
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
    def test_datamodule_dataloaders(self):
        from virtual_rodent.data.datamodule import VirtualRodentDataModule
        
        dm = VirtualRodentDataModule(
            data_dir=DATA_DIR,
            batch_size=32,
            num_workers=0,
        )
        
        dm.setup("fit")
        
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        neural, pose = batch
        assert neural.shape[0] == 32  # Batch size
    
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
    def test_datamodule_split_strategies(self):
        from virtual_rodent.data.datamodule import VirtualRodentDataModule
        
        for strategy in ["random", "session"]:
            dm = VirtualRodentDataModule(
                data_dir=DATA_DIR,
                split_strategy=strategy,
                num_workers=0,
            )
            dm.setup("fit")
            
            assert len(dm.train_dataset) > 0
            assert len(dm.val_dataset) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
