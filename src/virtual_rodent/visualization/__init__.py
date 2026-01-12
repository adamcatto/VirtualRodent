"""
Visualization utilities for neural decoding results.
"""

from virtual_rodent.visualization.animations import generate_prediction_video
from virtual_rodent.visualization.neural_viz import (
    visualize_neural_encoding,
    plot_neural_heatmap,
    plot_hidden_state_pca,
)

__all__ = [
    "generate_prediction_video",
    "visualize_neural_encoding",
    "plot_neural_heatmap",
    "plot_hidden_state_pca",
]
