"""
Capture the git state of the repository for experiment reproducibility.

The commit hash recorded alongside an experiment lets you check out the exact
code that produced a result. A ``dirty`` flag records whether there were
uncommitted changes at run time, so you know whether the commit hash tells the
full story.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional
import subprocess


@dataclass
class GitInfo:
    """Snapshot of the repository's git state at experiment time.

    Attributes:
        commit: Full 40-character commit SHA (``None`` if not a git repo).
        short_commit: Abbreviated commit SHA.
        branch: Current branch name (``HEAD`` if detached).
        dirty: Whether the working tree had uncommitted changes.
        dirty_files: Paths (relative to repo root) with uncommitted changes.
        remote_url: URL of the ``origin`` remote, if configured.
        describe: Output of ``git describe --tags --always --dirty``.
        commit_subject: Subject line of the recorded commit.
        committed_at: ISO-8601 author date of the recorded commit.
        available: ``True`` if git metadata was successfully captured.
    """

    commit: Optional[str] = None
    short_commit: Optional[str] = None
    branch: Optional[str] = None
    dirty: bool = False
    dirty_files: List[str] = field(default_factory=list)
    remote_url: Optional[str] = None
    describe: Optional[str] = None
    commit_subject: Optional[str] = None
    committed_at: Optional[str] = None
    available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GitInfo":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _run_git(args: List[str], repo_root: Path) -> Optional[str]:
    """Run a git command, returning stripped stdout or ``None`` on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def capture_git_info(repo_root: Optional[Path] = None) -> GitInfo:
    """Capture the current git state of ``repo_root``.

    Args:
        repo_root: Repository directory to inspect. Defaults to walking up from
            this file to find the enclosing repository.

    Returns:
        A :class:`GitInfo`. When the directory is not a git repository (or git
        is unavailable), ``available`` is ``False`` and the SHA fields are
        ``None`` rather than raising.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    repo_root = Path(repo_root)

    toplevel = _run_git(["rev-parse", "--show-toplevel"], repo_root)
    if toplevel is None:
        return GitInfo(available=False)

    root = Path(toplevel)
    commit = _run_git(["rev-parse", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    dirty_files: List[str] = []
    if status:
        for line in status.splitlines():
            # Porcelain format: two status chars, a space, then the path.
            dirty_files.append(line[3:].strip())

    return GitInfo(
        commit=commit,
        short_commit=_run_git(["rev-parse", "--short", "HEAD"], root),
        branch=_run_git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        dirty=bool(status),
        dirty_files=dirty_files,
        remote_url=_run_git(["config", "--get", "remote.origin.url"], root),
        describe=_run_git(["describe", "--tags", "--always", "--dirty"], root),
        commit_subject=_run_git(["log", "-1", "--pretty=%s"], root),
        committed_at=_run_git(["log", "-1", "--pretty=%cI"], root),
        available=commit is not None,
    )
