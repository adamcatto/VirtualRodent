"""
Metrics computation for neural decoding evaluation.

Provides comprehensive metrics for per-session and per-animal analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from virtual_rodent.environment.session_env import SessionEnvironment
from virtual_rodent.models.rodent_agent import RodentAgent


def compute_session_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    session_id: str,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Compute comprehensive metrics for a single session.

    Args:
        predictions: (num_samples, horizon, pose_dim) predictions
        ground_truth: (num_samples, horizon, pose_dim) ground truth
        session_id: Session identifier
        output_dir: Directory to save metrics (optional)

    Returns:
        Dict with metrics:
            - mse: Mean squared error
            - mae: Mean absolute error
            - rmse: Root mean squared error
            - r2_per_coordinate: R² for each coordinate
            - r2_mean: Mean R² across coordinates
            - velocity_mse: MSE on velocities
            - temporal_coherence: Smoothness metric
    """
    # Flatten batch and time dimensions for overall metrics
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    gt_flat = ground_truth.reshape(-1, ground_truth.shape[-1])

    # Basic metrics
    mse = mean_squared_error(gt_flat, pred_flat)
    mae = mean_absolute_error(gt_flat, pred_flat)
    rmse = np.sqrt(mse)

    # R² per coordinate
    r2_scores = []
    for i in range(pred_flat.shape[1]):
        r2 = r2_score(gt_flat[:, i], pred_flat[:, i])
        r2_scores.append(r2)

    r2_mean = np.mean(r2_scores)

    # Velocity MSE (temporal smoothness)
    if predictions.shape[1] > 1:
        pred_vel = predictions[:, 1:] - predictions[:, :-1]
        gt_vel = ground_truth[:, 1:] - ground_truth[:, :-1]
        velocity_mse = mean_squared_error(
            gt_vel.reshape(-1, gt_vel.shape[-1]),
            pred_vel.reshape(-1, pred_vel.shape[-1])
        )
    else:
        velocity_mse = 0.0

    # Temporal coherence (lower is better, measures jitter)
    if predictions.shape[1] > 2:
        pred_accel = predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2]
        temporal_coherence = np.mean(np.abs(pred_accel))
    else:
        temporal_coherence = 0.0

    metrics = {
        'session_id': session_id,
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_per_coordinate': [float(r) for r in r2_scores],
        'r2_mean': float(r2_mean),
        'velocity_mse': float(velocity_mse),
        'temporal_coherence': float(temporal_coherence),
        'num_samples': int(predictions.shape[0]),
    }

    # Save to file if output_dir provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{session_id}_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    return metrics


def aggregate_per_animal_metrics(
    session_metrics: List[Dict],
    animal: str,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Aggregate metrics across sessions for one animal.

    Args:
        session_metrics: List of per-session metric dicts
        animal: Animal name
        output_dir: Directory to save aggregated metrics

    Returns:
        Dict with aggregated metrics
    """
    if not session_metrics:
        return {}

    # Aggregate metrics
    agg_metrics = {
        'animal': animal,
        'num_sessions': len(session_metrics),
        'mse_mean': np.mean([m['mse'] for m in session_metrics]),
        'mse_std': np.std([m['mse'] for m in session_metrics]),
        'mae_mean': np.mean([m['mae'] for m in session_metrics]),
        'mae_std': np.std([m['mae'] for m in session_metrics]),
        'rmse_mean': np.mean([m['rmse'] for m in session_metrics]),
        'rmse_std': np.std([m['rmse'] for m in session_metrics]),
        'r2_mean': np.mean([m['r2_mean'] for m in session_metrics]),
        'r2_std': np.std([m['r2_mean'] for m in session_metrics]),
        'velocity_mse_mean': np.mean([m['velocity_mse'] for m in session_metrics]),
        'temporal_coherence_mean': np.mean([m['temporal_coherence'] for m in session_metrics]),
        'session_ids': [m['session_id'] for m in session_metrics],
    }

    # Save to file
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{animal}_summary.json"
        with open(output_file, 'w') as f:
            json.dump(agg_metrics, f, indent=2)

    return agg_metrics


def save_metrics_plots(
    all_session_metrics: List[Dict],
    animal_metrics: Dict[str, Dict],
    output_dir: Path,
):
    """
    Generate and save static plots for file browser viewing.

    Creates:
    - Per-session comparison bar plots
    - Per-animal comparison bar plots
    - R² per coordinate distributions
    - MSE distributions

    Args:
        all_session_metrics: List of all session metrics
        animal_metrics: Dict mapping animal names to aggregated metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    # 1. Per-session MSE comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    session_ids = [m['session_id'] for m in all_session_metrics]
    mses = [m['mse'] for m in all_session_metrics]

    ax.bar(range(len(session_ids)), mses)
    ax.set_xlabel('Session Index')
    ax.set_ylabel('MSE')
    ax.set_title('Per-Session Mean Squared Error')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_session_mse.png', dpi=150)
    plt.close()

    # 2. Per-session R² comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    r2s = [m['r2_mean'] for m in all_session_metrics]

    ax.bar(range(len(session_ids)), r2s)
    ax.set_xlabel('Session Index')
    ax.set_ylabel('Mean R²')
    ax.set_title('Per-Session Mean R² Score')
    ax.axhline(y=0.7, color='r', linestyle='--', label='Good threshold (0.7)')
    ax.axhline(y=0.85, color='g', linestyle='--', label='Excellent threshold (0.85)')
    ax.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_session_r2.png', dpi=150)
    plt.close()

    # 3. Per-animal comparison
    if animal_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        animals = list(animal_metrics.keys())

        # MSE
        mse_means = [animal_metrics[a]['mse_mean'] for a in animals]
        mse_stds = [animal_metrics[a]['mse_std'] for a in animals]
        axes[0, 0].bar(animals, mse_means, yerr=mse_stds, capsize=5)
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('Per-Animal MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # R²
        r2_means = [animal_metrics[a]['r2_mean'] for a in animals]
        r2_stds = [animal_metrics[a]['r2_std'] for a in animals]
        axes[0, 1].bar(animals, r2_means, yerr=r2_stds, capsize=5)
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_title('Per-Animal R²')
        axes[0, 1].axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # MAE
        mae_means = [animal_metrics[a]['mae_mean'] for a in animals]
        mae_stds = [animal_metrics[a]['mae_std'] for a in animals]
        axes[1, 0].bar(animals, mae_means, yerr=mae_stds, capsize=5)
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Per-Animal MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Number of sessions
        num_sessions = [animal_metrics[a]['num_sessions'] for a in animals]
        axes[1, 1].bar(animals, num_sessions)
        axes[1, 1].set_ylabel('Number of Sessions')
        axes[1, 1].set_title('Sessions per Animal')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'per_animal_comparison.png', dpi=150)
        plt.close()

    # 4. R² distribution across all coordinates
    all_r2_coords = []
    for m in all_session_metrics:
        all_r2_coords.extend(m['r2_per_coordinate'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_r2_coords, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('R² Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of R² Scores Across All Coordinates')
    ax.axvline(x=np.mean(all_r2_coords), color='r', linestyle='--',
               label=f'Mean: {np.mean(all_r2_coords):.3f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_distribution.png', dpi=150)
    plt.close()

    print(f"\nMetrics plots saved to {output_dir}/")


def compute_keypoint_wise_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_keypoints: int = 23,
) -> Dict:
    """
    Compute per-keypoint metrics (assumes flattened pose with 69 dims = 23 keypoints * 3).

    Args:
        predictions: (num_samples, horizon, 69) predictions
        ground_truth: (num_samples, horizon, 69) ground truth
        num_keypoints: Number of keypoints (default 23)

    Returns:
        Dict with per-keypoint MSE, MAE, R²
    """
    # Reshape to separate keypoints
    pred = predictions.reshape(-1, num_keypoints, 3)  # (N, 23, 3)
    gt = ground_truth.reshape(-1, num_keypoints, 3)

    keypoint_metrics = {}

    for kp_idx in range(num_keypoints):
        pred_kp = pred[:, kp_idx, :]
        gt_kp = gt[:, kp_idx, :]

        mse = mean_squared_error(gt_kp, pred_kp)
        mae = mean_absolute_error(gt_kp, pred_kp)

        # R² for this keypoint (averaged across x, y, z)
        r2_scores = []
        for dim in range(3):
            r2 = r2_score(gt_kp[:, dim], pred_kp[:, dim])
            r2_scores.append(r2)

        keypoint_metrics[f'keypoint_{kp_idx}'] = {
            'mse': float(mse),
            'mae': float(mae),
            'r2_mean': float(np.mean(r2_scores)),
        }

    return keypoint_metrics
