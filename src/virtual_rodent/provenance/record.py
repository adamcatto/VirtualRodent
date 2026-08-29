"""
The :class:`ExperimentRecord` schema — the machine-readable heart of the
provenance layer.

A record answers, for a single experiment run: *what was tried, on what code,
in what environment, with what choices, and what came out.* It is designed to
round-trip losslessly through JSON so it can be committed to git and re-read by
tooling (the index builder, notebooks, the website) later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import re

from virtual_rodent.provenance.environment import EnvironmentInfo, capture_environment
from virtual_rodent.provenance.git_info import GitInfo, capture_git_info

SCHEMA_VERSION = 1


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str) -> str:
    """Turn a free-form name into a filesystem- and URL-safe slug."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "experiment"


@dataclass
class ExperimentRecord:
    """A single provenance record for one experiment run.

    The fields map directly onto the questions a provenance layer must answer:

    - **What / why**: ``name``, ``description``, ``tags``.
    - **How it was structured**: ``structure``, ``config``.
    - **On what code / environment**: ``git``, ``environment``.
    - **What came out**: ``metrics``, ``figures``, ``artifacts``.

    Attributes:
        experiment_id: Unique, slug-like identifier (also the folder name).
        name: Human-readable title.
        description: Free-form prose (Markdown) describing the experiment,
            its hypothesis, and its design.
        structure: Notes on how the experiment was structured — data splits,
            model architecture summary, training loop, evaluation protocol.
            May be prose or a nested dict.
        config: Full parameter / training / eval configuration (e.g. the
            resolved Hydra config), captured verbatim for reproducibility.
        metrics: Results — arbitrary (possibly nested) numeric summaries.
        tags: Short labels for filtering and grouping.
        figures: Relative paths (within the experiment folder) to saved
            visualizations.
        artifacts: Named references to larger outputs kept outside git
            (e.g. checkpoint paths under ``outs/``).
        git: Repository state at run time.
        environment: Interpreter and dependency versions at run time.
        created_at: ISO-8601 UTC timestamp when the record was created.
        status: Lifecycle marker, e.g. ``"completed"``, ``"running"``,
            ``"failed"``.
        schema_version: Version of this record schema.
    """

    experiment_id: str
    name: str = ""
    description: str = ""
    structure: Any = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    git: GitInfo = field(default_factory=GitInfo)
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)
    created_at: str = field(default_factory=_utcnow_iso)
    status: str = "completed"
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        *,
        experiment_id: Optional[str] = None,
        structure: Any = None,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        status: str = "completed",
        capture_git: bool = True,
        capture_env: bool = True,
    ) -> "ExperimentRecord":
        """Build a record, capturing git and environment state automatically.

        Args:
            name: Human-readable title.
            description: Prose description of the experiment.
            experiment_id: Explicit id. When omitted, a timestamped slug of
                ``name`` is generated so ids sort chronologically and stay
                unique across runs.
            structure: How the experiment was structured (prose or dict).
            config: Parameter / training / eval configuration.
            metrics: Result metrics.
            tags: Labels for grouping.
            artifacts: Named references to out-of-git outputs.
            status: Lifecycle marker.
            capture_git: Capture git state (set ``False`` in tests / non-repos).
            capture_env: Capture environment info.
        """
        if experiment_id is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            experiment_id = f"{stamp}_{slugify(name)}"

        return cls(
            experiment_id=experiment_id,
            name=name,
            description=description,
            structure=structure,
            config=config or {},
            metrics=metrics or {},
            tags=tags or [],
            artifacts=artifacts or {},
            git=capture_git_info() if capture_git else GitInfo(),
            environment=capture_environment() if capture_env else EnvironmentInfo(),
            status=status,
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        data = asdict(self)
        # asdict already recurses into GitInfo / EnvironmentInfo dataclasses.
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        """Reconstruct a record from :meth:`to_dict` output."""
        data = dict(data)
        git = data.pop("git", {}) or {}
        env = data.pop("environment", {}) or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["git"] = GitInfo.from_dict(git) if isinstance(git, dict) else git
        kwargs["environment"] = EnvironmentInfo.from_dict(env) if isinstance(env, dict) else env
        return cls(**kwargs)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def primary_metric(self) -> Optional[tuple]:
        """Return a ``(name, value)`` pair for the headline scalar metric.

        Picks the first scalar (int/float) found by scanning common result
        keys, then any remaining scalar. Used for index / leaderboard columns.
        Returns ``None`` when no scalar metric is present.
        """
        preferred = ["r2_mean", "r2", "best_val_loss", "val_loss", "mse", "rmse", "mae"]
        flat = _flatten_scalars(self.metrics)
        for key in preferred:
            for full_key, value in flat.items():
                if full_key == key or full_key.endswith("." + key):
                    return (full_key, value)
        for full_key, value in flat.items():
            return (full_key, value)
        return None


def _flatten_scalars(d: Any, prefix: str = "") -> Dict[str, float]:
    """Flatten nested dict scalars into ``dotted.key -> value`` pairs."""
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for key, value in d.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_scalars(value, full))
    elif isinstance(d, bool):
        # Skip booleans; they are rarely the headline metric.
        pass
    elif isinstance(d, (int, float)):
        out[prefix] = float(d)
    return out
