"""
Analysis and metrics computation for neural decoding.
"""

from virtual_rodent.analysis.metrics import (
    compute_session_metrics,
    aggregate_per_animal_metrics,
    save_metrics_plots,
)

__all__ = [
    "compute_session_metrics",
    "aggregate_per_animal_metrics",
    "save_metrics_plots",
]
