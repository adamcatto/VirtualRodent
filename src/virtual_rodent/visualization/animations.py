"""
Animation generation for pose predictions.

Creates side-by-side comparison videos of ground truth vs predicted poses.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# Skeleton connections for rat (23 keypoints)
SKELETON_CONNECTIONS = [
    (0, 3), (3, 4), (3, 5),  # Head-neck-shoulders
    (4, 6), (6, 8), (8, 10),  # Left arm
    (5, 7), (7, 9), (9, 11),  # Right arm
    (3, 12), (12, 13), (13, 14),  # Spine
    (14, 15), (14, 16),  # Hips
    (15, 17), (17, 19), (19, 21),  # Left leg
    (16, 18), (18, 20), (20, 22),  # Right leg
]


def unflatten_keypoints(flat_keypoints: np.ndarray) -> np.ndarray:
    """
    Unflatten keypoints from (T, 69) to (T, 23, 3).

    Args:
        flat_keypoints: Array of shape (T, 69) or (69,)

    Returns:
        Unflattened array of shape (T, 23, 3) or (23, 3)
    """
    if flat_keypoints.ndim == 2:
        return flat_keypoints.reshape(flat_keypoints.shape[0], 23, 3)
    else:
        return flat_keypoints.reshape(23, 3)


def plot_skeleton_3d(ax, keypoints: np.ndarray, color='blue', label=''):
    """
    Plot a 3D skeleton.

    Args:
        ax: Matplotlib 3D axis
        keypoints: (23, 3) array of keypoint positions
        color: Color for the skeleton
        label: Label for legend
    """
    # Plot keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
               c=color, s=30, alpha=0.8)

    # Plot bones
    for i, j in SKELETON_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            xs = [keypoints[i, 0], keypoints[j, 0]]
            ys = [keypoints[i, 1], keypoints[j, 1]]
            zs = [keypoints[i, 2], keypoints[j, 2]]
            ax.plot(xs, ys, zs, c=color, linewidth=2, alpha=0.6)


def generate_prediction_video(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
    fps: int = 50,
    duration: Optional[int] = None,
):
    """
    Generate side-by-side comparison video of predictions vs ground truth.

    Args:
        predictions: (T, 69) or (T, 250, 69) predicted poses
        ground_truth: (T, 69) or (T, 250, 69) ground truth poses
        output_path: Path to save video
        fps: Frames per second
        duration: Maximum duration in frames (None = all frames)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Unflatten if needed
    if predictions.ndim == 2:
        pred_kp = unflatten_keypoints(predictions)
        gt_kp = unflatten_keypoints(ground_truth)
    else:
        # Take first sample if batched
        pred_kp = unflatten_keypoints(predictions[0])
        gt_kp = unflatten_keypoints(ground_truth[0])

    num_frames = min(len(pred_kp), len(gt_kp))
    if duration is not None:
        num_frames = min(num_frames, duration)

    # Create figure with 3D subplots
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Set limits based on data
    all_data = np.concatenate([pred_kp[:num_frames].reshape(-1, 3),
                                gt_kp[:num_frames].reshape(-1, 3)])
    x_range = [all_data[:, 0].min(), all_data[:, 0].max()]
    y_range = [all_data[:, 1].min(), all_data[:, 1].max()]
    z_range = [all_data[:, 2].min(), all_data[:, 2].max()]

    def update(frame):
        ax1.cla()
        ax2.cla()

        # Plot ground truth
        plot_skeleton_3d(ax1, gt_kp[frame], color='green', label='Ground Truth')
        ax1.set_title(f'Ground Truth - Frame {frame}')
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.set_zlim(z_range)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Plot prediction
        plot_skeleton_3d(ax2, pred_kp[frame], color='red', label='Prediction')
        ax2.set_title(f'Prediction - Frame {frame}')
        ax2.set_xlim(x_range)
        ax2.set_ylim(y_range)
        ax2.set_zlim(z_range)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)

    # Save animation
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()

    print(f"Animation saved successfully!")
