"""
Neural population visualization utilities.

Creates heatmaps, PCA plots, and correlation matrices for neural activity.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_neural_heatmap(
    neural_data: np.ndarray,
    output_path: Path,
    title: str = "Neural Activity Heatmap",
):
    """
    Plot neural activity heatmap (time × neurons).

    Args:
        neural_data: (T, N) array of neural activity
        output_path: Path to save plot
        title: Plot title
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(neural_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Neuron Index')
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label='Activity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Neural heatmap saved to {output_path}")


def plot_hidden_state_pca(
    hidden_states: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "LSTM Hidden State PCA",
):
    """
    Plot PCA of LSTM hidden states.

    Args:
        hidden_states: (T, hidden_dim) hidden state activations
        labels: Optional (T,) array of labels for coloring
        output_path: Path to save plot
        title: Plot title
    """
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(hidden_states)

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels,
                           cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"PCA plot saved to {output_path}")
    else:
        plt.show()


def visualize_neural_encoding(
    neural_data: np.ndarray,
    pose_data: np.ndarray,
    output_dir: Path,
    session_id: str = "session",
):
    """
    Generate comprehensive neural encoding visualizations.

    Args:
        neural_data: (T, N) neural activity
        pose_data: (T, pose_dim) pose data
        output_dir: Directory to save visualizations
        session_id: Session identifier for file naming
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Neural heatmap
    plot_neural_heatmap(
        neural_data,
        output_dir / f"{session_id}_heatmap.png",
        title=f"Neural Activity - {session_id}",
    )

    # 2. Neuron-pose correlation matrix
    if pose_data.shape[0] == neural_data.shape[0]:
        corr_matrix = np.corrcoef(neural_data.T, pose_data.T)
        n_neurons = neural_data.shape[1]

        # Extract neuron-pose block
        neuron_pose_corr = corr_matrix[:n_neurons, n_neurons:]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(neuron_pose_corr, cmap='coolwarm', center=0,
                   ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_xlabel('Pose Dimension')
        ax.set_ylabel('Neuron Index')
        ax.set_title(f'Neuron-Pose Correlation - {session_id}')

        plt.tight_layout()
        plt.savefig(output_dir / f"{session_id}_correlation.png", dpi=150)
        plt.close()

        print(f"Correlation matrix saved")

    print(f"All neural visualizations saved to {output_dir}")
