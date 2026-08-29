"""
Render human-readable Markdown from :class:`ExperimentRecord` objects.

Two views are produced:

- :func:`render_experiment_readme` — the per-experiment page that lives inside
  each ``experiments/<id>/`` folder and renders nicely on GitHub, with the
  description, config, results table, and inline figures.
- :func:`render_index` — the top-level ``experiments/README.md`` leaderboard
  linking to every recorded experiment.
"""

from __future__ import annotations

from typing import Any, Dict, List
import json

from virtual_rodent.provenance.record import ExperimentRecord, _flatten_scalars


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        # Compact but informative formatting for metric values.
        if value != value:  # NaN
            return "nan"
        if abs(value) >= 1e4 or (0 < abs(value) < 1e-4):
            return f"{value:.3e}"
        return f"{value:.4g}"
    return str(value)


def _metrics_table(metrics: Dict[str, Any]) -> str:
    flat = _flatten_scalars(metrics)
    if not flat:
        return "_No scalar metrics recorded._\n"
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in flat.items():
        lines.append(f"| `{key}` | {_fmt_value(value)} |")
    return "\n".join(lines) + "\n"


def _code_block(data: Any, lang: str = "json") -> str:
    if lang == "json":
        body = json.dumps(data, indent=2, default=str)
    else:
        body = str(data)
    return f"```{lang}\n{body}\n```\n"


def _git_section(record: ExperimentRecord) -> str:
    git = record.git
    if not git.available:
        return "_Git state was not captured (not a repository, or git unavailable)._\n"
    lines = [
        f"- **Commit**: `{git.commit}`" + (f" ({git.short_commit})" if git.short_commit else ""),
        f"- **Branch**: `{git.branch}`" if git.branch else "",
        f"- **Subject**: {git.commit_subject}" if git.commit_subject else "",
        f"- **Committed at**: {git.committed_at}" if git.committed_at else "",
        f"- **Describe**: `{git.describe}`" if git.describe else "",
        f"- **Remote**: {git.remote_url}" if git.remote_url else "",
        f"- **Working tree**: {'dirty (uncommitted changes present)' if git.dirty else 'clean'}",
    ]
    out = "\n".join(line for line in lines if line) + "\n"
    if git.dirty and git.dirty_files:
        shown = git.dirty_files[:20]
        out += "\n<details><summary>Uncommitted files</summary>\n\n"
        out += "\n".join(f"- `{f}`" for f in shown)
        if len(git.dirty_files) > len(shown):
            out += f"\n- … and {len(git.dirty_files) - len(shown)} more"
        out += "\n\n</details>\n"
    return out


def _structure_section(structure: Any) -> str:
    if structure is None or structure == "":
        return "_Not specified._\n"
    if isinstance(structure, str):
        return structure.rstrip() + "\n"
    return _code_block(structure, "json")


def render_experiment_readme(record: ExperimentRecord) -> str:
    """Render the per-experiment ``README.md`` body."""
    parts: List[str] = []
    parts.append(f"# {record.name or record.experiment_id}\n")

    meta = [
        f"**ID**: `{record.experiment_id}`",
        f"**Created**: {record.created_at}",
        f"**Status**: {record.status}",
    ]
    if record.tags:
        meta.append("**Tags**: " + ", ".join(f"`{t}`" for t in record.tags))
    parts.append("  \n".join(meta) + "\n")

    if record.description:
        parts.append("## Description\n")
        parts.append(record.description.rstrip() + "\n")

    parts.append("## How it was structured\n")
    parts.append(_structure_section(record.structure))

    parts.append("## Reproducibility\n")
    parts.append(_git_section(record))

    if record.environment and (record.environment.packages or record.environment.python_version):
        env = record.environment
        parts.append("\n<details><summary>Environment</summary>\n")
        env_lines = []
        if env.python_version:
            env_lines.append(f"- **Python**: {env.python_version}")
        if env.platform:
            env_lines.append(f"- **Platform**: {env.platform}")
        if env.hostname:
            env_lines.append(f"- **Host**: {env.hostname}")
        for name, ver in env.packages.items():
            env_lines.append(f"- `{name}`: {ver}")
        parts.append("\n" + "\n".join(env_lines) + "\n\n</details>\n")

    parts.append("## Results\n")
    parts.append(_metrics_table(record.metrics))

    if record.figures:
        parts.append("\n## Visualizations\n")
        for fig in record.figures:
            title = fig.rsplit("/", 1)[-1]
            if fig.startswith("figures/"):
                # Copied into the folder — embed so it renders on GitHub.
                parts.append(f"### {title}\n")
                parts.append(f"![{title}]({fig})\n")
            else:
                # Referenced by path (not committed, keeps the repo lean).
                parts.append(f"- `{fig}` — referenced by path (not committed)")
        parts.append("")

    if record.config:
        parts.append("\n## Configuration\n")
        parts.append("Full parameter / training / evaluation configuration used for this run:\n")
        parts.append(_code_block(record.config, "json"))

    if record.artifacts:
        parts.append("\n## Artifacts\n")
        parts.append(
            "Large outputs kept outside git (paths are relative to the "
            "repository root at run time):\n"
        )
        for name, path in record.artifacts.items():
            parts.append(f"- **{name}**: `{path}`")
        parts.append("")

    parts.append(
        "\n---\n_Generated by the VirtualRodent provenance layer "
        "(`virtual_rodent.provenance`)._\n"
    )
    return "\n".join(parts)


def render_index(records: List[ExperimentRecord]) -> str:
    """Render the top-level ``experiments/README.md`` leaderboard."""
    parts: List[str] = []
    parts.append("# Experiments\n")
    parts.append(
        "Provenance records for prediction / forecasting experiments. Each row "
        "links to a self-contained folder holding the run's description, exact "
        "git commit, configuration, results, and visualizations. This file is "
        "generated by `virtual_rodent.provenance`; run "
        "`python -m virtual_rodent.provenance index` to rebuild it.\n"
    )

    if not records:
        parts.append("_No experiments recorded yet._\n")
        return "\n".join(parts)

    # Sort newest first.
    ordered = sorted(records, key=lambda r: r.created_at, reverse=True)

    parts.append(f"**{len(ordered)}** experiment(s) recorded.\n")
    parts.append("| Experiment | Created | Commit | Headline metric | Tags |")
    parts.append("| --- | --- | --- | --- | --- |")
    for record in ordered:
        link = f"[{record.name or record.experiment_id}]({record.experiment_id}/)"
        commit = f"`{record.git.short_commit}`" if record.git.short_commit else "—"
        if record.git.dirty:
            commit += " ⚠️"
        primary = record.primary_metric()
        metric = f"`{primary[0]}` = {_fmt_value(primary[1])}" if primary else "—"
        tags = ", ".join(f"`{t}`" for t in record.tags) if record.tags else "—"
        created = record.created_at.replace("T", " ").replace("+00:00", "Z")
        parts.append(f"| {link} | {created} | {commit} | {metric} | {tags} |")

    parts.append(
        "\n> ⚠️ marks runs recorded with a dirty working tree — the commit hash "
        "does not fully capture the code that ran.\n"
    )
    return "\n".join(parts)
