# Experiment Provenance Layer

The provenance layer turns every prediction / forecasting experiment into a
small, **git-tracked** record so results stay reproducible and reviewable long
after the run. It answers, for each run: *what was tried, on what code, in what
environment, with what choices, and what came out.*

Records live under the top-level [`experiments/`](../experiments/) directory —
one folder per run — and are committed to git alongside the code. To keep the
repository lean, only small text files are committed; large binary outputs
(checkpoints, prediction arrays, **and figures by default**) stay in the
gitignored `outs/` and are referenced by path rather than copied in.

### Keeping the repository lean

A git-tracked provenance folder should stay text-first and small:

- **Figures are referenced by path, not copied** (`copy_figures=False`, the
  default). The manifest and README point at the PNGs in `outs/`; nothing
  binary is committed. Pass `copy_figures=True` only for the occasional run you
  want to archive or share as a fully self-contained folder.
- **Checkpoints and arrays are never copied** — only referenced via `artifacts`.
- **Config is stored once** (in `manifest.json`); the README renders it for
  humans but no extra copies are written.

A typical committed record is a few kilobytes of JSON and Markdown.

## What gets recorded

Each experiment folder contains:

```
experiments/
  README.md                      # generated leaderboard / index of all runs
  <experiment_id>/
    manifest.json                # machine-readable source of truth
    config.yaml                  # resolved config snapshot (optional)
    README.md                    # human-readable page (rendered from manifest)
    figures/                     # copied visualizations (PNGs, etc.)
```

The `manifest.json` (an `ExperimentRecord`) captures:

| Field | Meaning |
| --- | --- |
| `name`, `description`, `tags` | What the experiment was and why |
| `structure` | How it was structured — splits, architecture, training/eval protocol |
| `config` | Full parameter / training / eval configuration (verbatim) |
| `metrics` | Results (arbitrary, possibly nested numeric summaries) |
| `figures` | Paths to visualizations (referenced by default; copied only if `copy_figures=True`) |
| `artifacts` | Named references to out-of-git outputs (e.g. checkpoints) |
| `git` | **Commit hash**, branch, dirty flag, remote — for reproducibility |
| `environment` | Python version, platform, key package versions |
| `created_at`, `status` | When it ran and its lifecycle state |

The recorded **commit hash** lets you check out the exact code that produced a
result. A `dirty` flag records whether there were uncommitted changes at run
time (shown as ⚠️ in the index), so you always know whether the commit hash
tells the full story.

## Recording an experiment from Python

```python
from virtual_rodent.provenance import log_experiment

log_experiment(
    name="LSTM multimodal baseline",
    description="Per-session decoder, 15s neural history, 5s pose horizon.",
    structure={
        "split_strategy": "per_session",
        "model": {"rnn_type": "lstm", "hidden_dim": 128, "num_layers": 2},
        "neural_history": 750,
        "pose_horizon": 250,
    },
    config=resolved_config_dict,          # e.g. OmegaConf.to_container(cfg)
    metrics={"r2_mean": 0.71, "best_val_loss": 0.0123},
    tags=["baseline", "per_session"],
    artifacts={"checkpoint": "outs/checkpoints/exp/best.ckpt"},
    figures=["outs/results/exp/per_session_r2.png"],  # referenced by path
    # copy_figures=True,  # opt in to commit the PNGs into the folder
)
```

Git and environment state are captured automatically. The call writes the
experiment folder and refreshes `experiments/README.md`.

### Automatic recording during training

`scripts/train.py` records a provenance entry automatically at the end of each
run (see `record_provenance()`), so no extra steps are needed for the standard
training flow. Recording failures are caught and never abort a completed run.

## Command-line interface

```bash
# Rebuild the experiments/README.md leaderboard from all manifests
python -m virtual_rodent.provenance index

# List recorded experiments (id, commit, headline metric, name)
python -m virtual_rodent.provenance list

# Print one experiment's rendered page to stdout
python -m virtual_rodent.provenance show <experiment_id>

# Record an experiment from JSON/text files (e.g. from a non-Python tool)
python -m virtual_rodent.provenance log \
    --name "GRU sweep" \
    --description-file notes.md \
    --config-file config.json \
    --metrics-file metrics.json \
    --figure outs/results/exp/per_session_r2.png \
    --tag sweep --tag gru
```

## Reading records back

```python
from virtual_rodent.provenance import ExperimentStore

store = ExperimentStore()
for record in store.list_records():
    print(record.experiment_id, record.primary_metric())

record = store.load("<experiment_id>")
print(record.config, record.metrics, record.git.commit)
```

## Design notes

- **`manifest.json` is the source of truth.** The `README.md` in each folder is
  rendered from it, so never hand-edit the README — re-run the recorder or
  `... provenance index` instead.
- **Text-first and small.** Only lightweight, reviewable artifacts are committed
  (manifests, configs, figures). Multi-megabyte arrays and checkpoints stay in
  `outs/` and are referenced by path.
- **Never breaks training.** Provenance recording is best-effort; a failure is
  logged as a warning and the training run still completes.
