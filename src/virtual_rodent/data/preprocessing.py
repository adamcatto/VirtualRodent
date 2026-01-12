"""
Preprocessing utilities for neural and pose data.

Provides normalization, smoothing, and feature extraction functions.
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


def normalize_neural(
    spike_counts: np.ndarray,
    method: str = "zscore",
    axis: int = 0,
    eps: float = 1e-8,
    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Normalize neural spike counts.
    
    Args:
        spike_counts: Array of shape (T, N) or (N,)
        method: Normalization method ('zscore', 'minmax', 'sqrt', 'log')
        axis: Axis along which to compute statistics (0 for time)
        eps: Small constant to avoid division by zero
        stats: Pre-computed (mean, std) or (min, max) for inference
        
    Returns:
        Normalized spike counts. If stats is None, also returns computed stats.
    """
    if method == "zscore":
        if stats is not None:
            mean, std = stats
        else:
            mean = np.mean(spike_counts, axis=axis, keepdims=True)
            std = np.std(spike_counts, axis=axis, keepdims=True) + eps
        
        normalized = (spike_counts - mean) / std
        
        if stats is None:
            return normalized, (mean.squeeze(axis), std.squeeze(axis))
        return normalized
    
    elif method == "minmax":
        if stats is not None:
            min_val, max_val = stats
        else:
            min_val = np.min(spike_counts, axis=axis, keepdims=True)
            max_val = np.max(spike_counts, axis=axis, keepdims=True)
        
        normalized = (spike_counts - min_val) / (max_val - min_val + eps)
        
        if stats is None:
            return normalized, (min_val.squeeze(axis), max_val.squeeze(axis))
        return normalized
    
    elif method == "sqrt":
        # Square root transformation (variance-stabilizing for Poisson)
        return np.sqrt(spike_counts + 0.5)
    
    elif method == "log":
        # Log transformation
        return np.log1p(spike_counts)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_pose(
    pose: np.ndarray,
    method: str = "center",
    reference_keypoint: int = 14,  # spine_lower (approximate body center)
    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Normalize 3D pose keypoints.
    
    Args:
        pose: Array of shape (T, 3, 23) or (3, 23)
        method: Normalization method ('center', 'zscore', 'none')
        reference_keypoint: Keypoint index to use as reference for centering
        stats: Pre-computed (mean, std) for zscore normalization
        
    Returns:
        Normalized pose. If stats is None and method='zscore', also returns stats.
    """
    if method == "center":
        # Center pose relative to reference keypoint
        if pose.ndim == 3:  # (T, 3, 23)
            reference = pose[:, :, reference_keypoint:reference_keypoint+1]
        else:  # (3, 23)
            reference = pose[:, reference_keypoint:reference_keypoint+1]
        
        return pose - reference
    
    elif method == "zscore":
        if stats is not None:
            mean, std = stats
            return (pose - mean) / (std + 1e-8)
        else:
            mean = np.mean(pose, axis=0, keepdims=True)
            std = np.std(pose, axis=0, keepdims=True) + 1e-8
            return (pose - mean) / std, (mean.squeeze(0), std.squeeze(0))
    
    elif method == "none":
        return pose
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_pose_velocity(
    pose: np.ndarray,
    dt: float = 1.0 / 50.0,  # 50 Hz sampling
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """
    Compute velocity of pose keypoints.
    
    Args:
        pose: Array of shape (T, 3, 23) or (T, D)
        dt: Time step in seconds
        smooth_sigma: Gaussian smoothing sigma (0 = no smoothing)
        
    Returns:
        Velocity array of same shape as input
    """
    if smooth_sigma > 0:
        pose = gaussian_filter1d(pose, sigma=smooth_sigma, axis=0)
    
    velocity = np.gradient(pose, dt, axis=0)
    return velocity


def smooth_neural(
    spike_counts: np.ndarray,
    method: str = "gaussian",
    window_size: int = 5,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Smooth neural spike counts.
    
    Args:
        spike_counts: Array of shape (T, N)
        method: Smoothing method ('gaussian', 'boxcar', 'exponential')
        window_size: Size of smoothing window (for boxcar)
        sigma: Standard deviation (for Gaussian)
        
    Returns:
        Smoothed spike counts
    """
    if method == "gaussian":
        return gaussian_filter1d(spike_counts, sigma=sigma, axis=0)
    
    elif method == "boxcar":
        kernel = np.ones(window_size) / window_size
        smoothed = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'),
            axis=0,
            arr=spike_counts,
        )
        return smoothed
    
    elif method == "exponential":
        # Exponential moving average
        alpha = 2 / (window_size + 1)
        smoothed = np.zeros_like(spike_counts, dtype=np.float32)
        smoothed[0] = spike_counts[0]
        for t in range(1, len(spike_counts)):
            smoothed[t] = alpha * spike_counts[t] + (1 - alpha) * smoothed[t - 1]
        return smoothed
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def flatten_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Flatten keypoints from (T, 3, 23) to (T, 69).
    
    Args:
        keypoints: Array of shape (T, 3, 23) or (3, 23)
        
    Returns:
        Flattened array of shape (T, 69) or (69,)
    """
    if keypoints.ndim == 3:
        return keypoints.reshape(keypoints.shape[0], -1)
    else:
        return keypoints.reshape(-1)


def unflatten_keypoints(flat_keypoints: np.ndarray) -> np.ndarray:
    """
    Unflatten keypoints from (T, 69) to (T, 3, 23).
    
    Args:
        flat_keypoints: Array of shape (T, 69) or (69,)
        
    Returns:
        Unflattened array of shape (T, 3, 23) or (3, 23)
    """
    if flat_keypoints.ndim == 2:
        return flat_keypoints.reshape(flat_keypoints.shape[0], 3, 23)
    else:
        return flat_keypoints.reshape(3, 23)


def create_temporal_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
    pad_mode: str = "edge",
) -> np.ndarray:
    """
    Create sliding windows from temporal data.
    
    Args:
        data: Array of shape (T, D)
        window_size: Size of each window
        stride: Stride between windows
        pad_mode: Padding mode for edges ('edge', 'zero', 'none')
        
    Returns:
        Array of shape (num_windows, window_size, D)
    """
    T, D = data.shape
    
    # Handle padding
    if pad_mode == "edge":
        pad_before = window_size // 2
        pad_after = window_size - pad_before - 1
        data = np.pad(data, ((pad_before, pad_after), (0, 0)), mode='edge')
    elif pad_mode == "zero":
        pad_before = window_size // 2
        pad_after = window_size - pad_before - 1
        data = np.pad(data, ((pad_before, pad_after), (0, 0)), mode='constant')
    elif pad_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown pad mode: {pad_mode}")
    
    # Create windows using stride tricks for efficiency
    T_padded = len(data)
    num_windows = (T_padded - window_size) // stride + 1
    
    shape = (num_windows, window_size, D)
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
    
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return windows.copy()  # Copy to avoid memory issues


def compute_statistics(
    data_iterator,
    keys: Optional[list] = None,
) -> dict:
    """
    Compute running statistics (mean, std, min, max) over a data iterator.
    
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        data_iterator: Iterator yielding dicts with data arrays
        keys: Keys to compute statistics for (default: all keys)
        
    Returns:
        Dict with statistics for each key
    """
    stats = {}
    counts = {}
    
    for batch in data_iterator:
        if keys is None:
            keys = list(batch.keys())
        
        for key in keys:
            data = batch[key]
            
            if key not in stats:
                stats[key] = {
                    'mean': np.zeros_like(data[0], dtype=np.float64),
                    'M2': np.zeros_like(data[0], dtype=np.float64),
                    'min': np.full_like(data[0], np.inf, dtype=np.float64),
                    'max': np.full_like(data[0], -np.inf, dtype=np.float64),
                }
                counts[key] = 0
            
            for sample in data:
                counts[key] += 1
                delta = sample - stats[key]['mean']
                stats[key]['mean'] += delta / counts[key]
                delta2 = sample - stats[key]['mean']
                stats[key]['M2'] += delta * delta2
                stats[key]['min'] = np.minimum(stats[key]['min'], sample)
                stats[key]['max'] = np.maximum(stats[key]['max'], sample)
    
    # Finalize statistics
    for key in stats:
        variance = stats[key]['M2'] / max(counts[key] - 1, 1)
        stats[key]['std'] = np.sqrt(variance)
        del stats[key]['M2']
    
    return stats
