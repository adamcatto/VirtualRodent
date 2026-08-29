"""
:class:`ExperimentStore` — the on-disk home of the provenance layer.

The store owns the top-level ``experiments/`` directory. Each experiment gets
its own folder::

    experiments/
      README.md                      # generated leaderboard / index
      20260829-142530_lstm-baseline/
        manifest.json                # the ExperimentRecord (source of truth)
        config.yaml                  # resolved config, if provided
        README.md                    # rendered human-readable page
        figures/
          per_session_r2.png
          ...

``manifest.json`` is the machine-readable source of truth; ``README.md`` is
rendered from it. Everything here is small and text-first so it can be committed
to git. Large binary outputs (checkpoints, prediction arrays) stay in ``outs/``
and are referenced by path via the record's ``artifacts`` field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import shutil

from virtual_rodent.provenance.record import ExperimentRecord
from virtual_rodent.provenance.report import render_experiment_readme, render_index

MANIFEST_NAME = "manifest.json"
README_NAME = "README.md"
CONFIG_NAME = "config.yaml"
FIGURES_DIR = "figures"


def _default_experiments_dir() -> Path:
    """Locate the repo-level ``experiments/`` directory."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "experiments"


class ExperimentStore:
    """Read/write access to the ``experiments/`` provenance directory.

    Args:
        root: Path to the experiments directory. Defaults to ``experiments/``
            at the repository root.
    """

    def __init__(self, root: Optional[Union[str, Path]] = None) -> None:
        self.root = Path(root) if root is not None else _default_experiments_dir()

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def experiment_dir(self, experiment_id: str) -> Path:
        return self.root / experiment_id

    def save(
        self,
        record: ExperimentRecord,
        *,
        figures: Optional[List[Union[str, Path]]] = None,
        config_yaml: Optional[str] = None,
        update_index: bool = True,
        overwrite: bool = False,
    ) -> Path:
        """Persist a record to its own folder and refresh the index.

        Args:
            record: The experiment record to write. Its ``figures`` list is
                extended with the basenames of any copied ``figures``.
            figures: Image files to copy into the experiment's ``figures/``
                subdirectory. Paths that do not exist are skipped with a
                warning rather than raising.
            config_yaml: Optional pre-serialized YAML config to write alongside
                ``manifest.json`` (e.g. ``OmegaConf.to_yaml(cfg)``).
            update_index: Rebuild the top-level ``README.md`` after saving.
            overwrite: Allow writing into an existing, non-empty experiment
                folder. When ``False`` (default) an existing folder raises.

        Returns:
            The path to the experiment directory.
        """
        exp_dir = self.experiment_dir(record.experiment_id)
        if exp_dir.exists() and any(exp_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Experiment folder already exists and is not empty: {exp_dir}. "
                "Pass overwrite=True to replace it, or use a different id."
            )
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Copy figures in and register their relative paths on the record.
        copied: List[str] = []
        if figures:
            fig_dir = exp_dir / FIGURES_DIR
            fig_dir.mkdir(exist_ok=True)
            for fig in figures:
                src = Path(fig)
                if not src.exists():
                    print(f"[provenance] warning: figure not found, skipping: {src}")
                    continue
                dst = fig_dir / src.name
                shutil.copy2(src, dst)
                copied.append(f"{FIGURES_DIR}/{src.name}")
        # Merge without duplicates, preserving order.
        for rel in copied:
            if rel not in record.figures:
                record.figures.append(rel)

        # Write manifest (source of truth).
        manifest_path = exp_dir / MANIFEST_NAME
        with open(manifest_path, "w") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)

        # Write optional YAML config snapshot.
        if config_yaml is not None:
            with open(exp_dir / CONFIG_NAME, "w") as f:
                f.write(config_yaml)

        # Render human-readable page.
        with open(exp_dir / README_NAME, "w") as f:
            f.write(render_experiment_readme(record))

        if update_index:
            self.rebuild_index()

        return exp_dir

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def load(self, experiment_id: str) -> ExperimentRecord:
        """Load a single record by id."""
        manifest = self.experiment_dir(experiment_id) / MANIFEST_NAME
        if not manifest.exists():
            raise FileNotFoundError(f"No manifest found for experiment: {experiment_id}")
        with open(manifest) as f:
            data = json.load(f)
        return ExperimentRecord.from_dict(data)

    def list_records(self) -> List[ExperimentRecord]:
        """Load every record under the store, skipping malformed folders."""
        records: List[ExperimentRecord] = []
        if not self.root.exists():
            return records
        for child in sorted(self.root.iterdir()):
            manifest = child / MANIFEST_NAME
            if child.is_dir() and manifest.exists():
                try:
                    with open(manifest) as f:
                        records.append(ExperimentRecord.from_dict(json.load(f)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    print(f"[provenance] warning: could not read {manifest}: {exc}")
        return records

    def list_ids(self) -> List[str]:
        """Return the ids of all recorded experiments."""
        return [r.experiment_id for r in self.list_records()]

    # ------------------------------------------------------------------ #
    # Index
    # ------------------------------------------------------------------ #
    def rebuild_index(self) -> Path:
        """Regenerate the top-level ``README.md`` leaderboard."""
        self.root.mkdir(parents=True, exist_ok=True)
        index_path = self.root / README_NAME
        with open(index_path, "w") as f:
            f.write(render_index(self.list_records()))
        return index_path


def log_experiment(
    name: str,
    description: str = "",
    *,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    structure: Any = None,
    tags: Optional[List[str]] = None,
    artifacts: Optional[Dict[str, str]] = None,
    figures: Optional[List[Union[str, Path]]] = None,
    config_yaml: Optional[str] = None,
    experiment_id: Optional[str] = None,
    status: str = "completed",
    store: Optional[ExperimentStore] = None,
    capture_git: bool = True,
    capture_env: bool = True,
    overwrite: bool = True,
) -> ExperimentRecord:
    """Capture and persist an experiment in one call.

    This is the primary entry point for instrumentation code. It builds an
    :class:`ExperimentRecord` (capturing git and environment state), copies any
    figures in, writes the folder, and refreshes the index.

    Args:
        name: Human-readable experiment title.
        description: Prose description / hypothesis / notes (Markdown).
        config: Parameter / training / eval configuration.
        metrics: Result metrics (may be nested).
        structure: How the experiment was structured (prose or dict).
        tags: Labels for grouping.
        artifacts: Named references to out-of-git outputs (e.g. checkpoints).
        figures: Image files to copy into the experiment folder.
        config_yaml: Optional pre-serialized YAML config snapshot.
        experiment_id: Explicit id; auto-generated from ``name`` when omitted.
        status: Lifecycle marker.
        store: Target store. Defaults to the repo-level ``experiments/``.
        capture_git: Capture git state.
        capture_env: Capture environment info.
        overwrite: Replace an existing folder with the same id (default
            ``True`` so re-running an experiment refreshes its record).

    Returns:
        The saved :class:`ExperimentRecord` (with ``figures`` populated).
    """
    record = ExperimentRecord.create(
        name=name,
        description=description,
        experiment_id=experiment_id,
        structure=structure,
        config=config,
        metrics=metrics,
        tags=tags,
        artifacts=artifacts,
        status=status,
        capture_git=capture_git,
        capture_env=capture_env,
    )
    store = store or ExperimentStore()
    store.save(record, figures=figures, config_yaml=config_yaml, overwrite=overwrite)
    return record
