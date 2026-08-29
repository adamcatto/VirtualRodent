"""
Provenance layer for VirtualRodent prediction / forecasting experiments.

This package turns each experiment run into a small, self-contained, git-tracked
record that answers: *what was tried, on what code, in what environment, with
what choices, and what came out.*

Records live under the repository's top-level ``experiments/`` directory, one
folder per run, holding a machine-readable ``manifest.json``, a human-readable
``README.md``, an optional ``config.yaml`` snapshot, and copied ``figures/``.
A generated ``experiments/README.md`` indexes them all.

Typical usage from training / evaluation code::

    from virtual_rodent.provenance import log_experiment

    log_experiment(
        name="LSTM multimodal baseline",
        description="Per-session decoder, 15s neural history, 5s horizon.",
        config=resolved_config_dict,
        metrics={"r2_mean": 0.71, "best_val_loss": 0.0123},
        structure="Per-session split (70/15/15). Bidirectional LSTM, 2 layers...",
        tags=["baseline", "per_session"],
        artifacts={"checkpoint": "outs/checkpoints/exp/best.ckpt"},
        figures=["outs/results/exp/per_session_r2.png"],
    )

See :mod:`virtual_rodent.provenance.cli` for the command-line interface
(``python -m virtual_rodent.provenance``).
"""

from virtual_rodent.provenance.environment import EnvironmentInfo, capture_environment
from virtual_rodent.provenance.git_info import GitInfo, capture_git_info
from virtual_rodent.provenance.record import ExperimentRecord, slugify
from virtual_rodent.provenance.store import ExperimentStore, log_experiment

__all__ = [
    "ExperimentRecord",
    "ExperimentStore",
    "log_experiment",
    "GitInfo",
    "capture_git_info",
    "EnvironmentInfo",
    "capture_environment",
    "slugify",
]
