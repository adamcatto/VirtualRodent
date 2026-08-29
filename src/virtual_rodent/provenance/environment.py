"""
Capture the software environment an experiment ran in.

Two experiments with the same git commit can still diverge if their dependency
versions differ. Recording the interpreter and key package versions makes that
drift visible when you revisit a result months later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import platform
import sys

# Packages whose versions materially affect results in this project. Extend as
# needed; missing packages are simply omitted rather than erroring.
DEFAULT_TRACKED_PACKAGES: List[str] = [
    "numpy",
    "scipy",
    "pandas",
    "torch",
    "pytorch_lightning",
    "scikit-learn",
    "hydra-core",
    "matplotlib",
    "h5py",
]


@dataclass
class EnvironmentInfo:
    """Snapshot of the interpreter and dependency versions.

    Attributes:
        python_version: ``sys.version`` string of the running interpreter.
        platform: Human-readable OS/architecture description.
        hostname: Machine the experiment ran on.
        packages: Mapping of package name to installed version.
    """

    python_version: str = ""
    platform: str = ""
    hostname: str = ""
    packages: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EnvironmentInfo":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive
        return None


def capture_environment(
    tracked_packages: Optional[List[str]] = None,
) -> EnvironmentInfo:
    """Capture interpreter, platform, and dependency versions.

    Args:
        tracked_packages: Package names to record. Defaults to
            :data:`DEFAULT_TRACKED_PACKAGES`. Packages that are not installed
            are omitted from the result.

    Returns:
        An :class:`EnvironmentInfo`.
    """
    names = tracked_packages if tracked_packages is not None else DEFAULT_TRACKED_PACKAGES
    packages: Dict[str, str] = {}
    for name in names:
        ver = _package_version(name)
        if ver is not None:
            packages[name] = ver

    return EnvironmentInfo(
        python_version=sys.version.replace("\n", " "),
        platform=platform.platform(),
        hostname=platform.node(),
        packages=packages,
    )
