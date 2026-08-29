"""
Command-line interface for the provenance layer.

Examples::

    # Rebuild the experiments/README.md leaderboard from all manifests
    python -m virtual_rodent.provenance index

    # List recorded experiments
    python -m virtual_rodent.provenance list

    # Print one experiment's rendered page to stdout
    python -m virtual_rodent.provenance show 20260829-142530_lstm-baseline

    # Record an experiment from JSON files (e.g. wiring in a non-Python tool)
    python -m virtual_rodent.provenance log \\
        --name "LSTM baseline" --description-file notes.md \\
        --config-file config.json --metrics-file metrics.json \\
        --figure outs/results/exp/per_session_r2.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import argparse
import json
import sys

from virtual_rodent.provenance.report import render_experiment_readme
from virtual_rodent.provenance.store import ExperimentStore, log_experiment


def _load_json(path: Optional[str]):
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def _read_text(path: Optional[str]) -> str:
    if not path:
        return ""
    return Path(path).read_text()


def _cmd_index(args: argparse.Namespace) -> int:
    store = ExperimentStore(args.root)
    path = store.rebuild_index()
    print(f"Wrote index: {path}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    store = ExperimentStore(args.root)
    records = store.list_records()
    if not records:
        print("No experiments recorded.")
        return 0
    records.sort(key=lambda r: r.created_at, reverse=True)
    for record in records:
        primary = record.primary_metric()
        metric = f"{primary[0]}={primary[1]:.4g}" if primary else "—"
        commit = record.git.short_commit or "—"
        dirty = " (dirty)" if record.git.dirty else ""
        print(f"{record.experiment_id}\t{commit}{dirty}\t{metric}\t{record.name}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    store = ExperimentStore(args.root)
    try:
        record = store.load(args.experiment_id)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(render_experiment_readme(record))
    return 0


def _cmd_log(args: argparse.Namespace) -> int:
    figures: List[str] = list(args.figure or [])
    record = log_experiment(
        name=args.name,
        description=_read_text(args.description_file) or (args.description or ""),
        config=_load_json(args.config_file),
        metrics=_load_json(args.metrics_file),
        structure=_read_text(args.structure_file) or (args.structure or None),
        tags=args.tag or None,
        figures=figures or None,
        copy_figures=args.copy_figures,
        experiment_id=args.id,
        store=ExperimentStore(args.root),
    )
    print(f"Recorded experiment: {record.experiment_id}")
    print(f"  -> {ExperimentStore(args.root).experiment_dir(record.experiment_id)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m virtual_rodent.provenance",
        description="Manage the VirtualRodent experiment provenance layer.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Path to the experiments/ directory (defaults to repo-level experiments/).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Rebuild the experiments/README.md leaderboard.")
    p_index.set_defaults(func=_cmd_index)

    p_list = sub.add_parser("list", help="List recorded experiments.")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Print an experiment's rendered page.")
    p_show.add_argument("experiment_id")
    p_show.set_defaults(func=_cmd_show)

    p_log = sub.add_parser("log", help="Record a new experiment from files/flags.")
    p_log.add_argument("--name", required=True)
    p_log.add_argument("--id", default=None, help="Explicit experiment id.")
    p_log.add_argument("--description", default=None)
    p_log.add_argument("--description-file", default=None)
    p_log.add_argument("--structure", default=None)
    p_log.add_argument("--structure-file", default=None)
    p_log.add_argument("--config-file", default=None, help="JSON config file.")
    p_log.add_argument("--metrics-file", default=None, help="JSON metrics file.")
    p_log.add_argument("--tag", action="append", help="Repeatable tag.")
    p_log.add_argument("--figure", action="append", help="Repeatable figure path.")
    p_log.add_argument(
        "--copy-figures",
        action="store_true",
        help="Copy figures into git (default: reference by path, keeps the repo lean).",
    )
    p_log.set_defaults(func=_cmd_log)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
