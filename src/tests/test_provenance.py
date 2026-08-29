"""
Tests for the experiment provenance layer.

These tests are self-contained: they write to a temporary store and do not need
the Virtual Rodent dataset, so they always run.
"""

import pytest

from virtual_rodent.provenance import (
    ExperimentRecord,
    ExperimentStore,
    capture_environment,
    capture_git_info,
    log_experiment,
    slugify,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("LSTM Multimodal Baseline") == "lstm-multimodal-baseline"

    def test_strips_and_collapses(self):
        assert slugify("  A/B  test!!  ") == "a-b-test"

    def test_empty_fallback(self):
        assert slugify("---") == "experiment"


class TestCapture:
    def test_git_info_does_not_raise(self):
        info = capture_git_info()
        # In this repo it should be available, but the contract is only that it
        # never raises and returns a well-formed object.
        assert isinstance(info.available, bool)
        assert isinstance(info.dirty, bool)

    def test_environment_capture(self):
        env = capture_environment()
        assert env.python_version
        assert env.platform
        assert isinstance(env.packages, dict)


class TestRecordSerialization:
    def test_round_trip(self):
        record = ExperimentRecord.create(
            name="Test run",
            description="desc",
            config={"a": 1, "nested": {"b": 2}},
            metrics={"r2_mean": 0.5, "test": {"mse": 0.01}},
            tags=["x"],
            capture_git=False,
            capture_env=False,
        )
        restored = ExperimentRecord.from_dict(record.to_dict())
        assert restored.name == record.name
        assert restored.config == record.config
        assert restored.metrics == record.metrics
        assert restored.git.to_dict() == record.git.to_dict()

    def test_primary_metric_prefers_r2(self):
        record = ExperimentRecord.create(
            name="m",
            metrics={"mse": 0.1, "r2_mean": 0.8},
            capture_git=False,
            capture_env=False,
        )
        assert record.primary_metric() == ("r2_mean", 0.8)

    def test_primary_metric_none_when_no_scalars(self):
        record = ExperimentRecord.create(
            name="m", metrics={"note": "text"}, capture_git=False, capture_env=False
        )
        assert record.primary_metric() is None

    def test_auto_id_is_slugged_and_timestamped(self):
        record = ExperimentRecord.create(name="My Run", capture_git=False, capture_env=False)
        assert record.experiment_id.endswith("_my-run")


class TestStore:
    def test_save_and_load(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = log_experiment(
            name="Baseline",
            description="A baseline run.",
            config={"lr": 1e-3},
            metrics={"r2_mean": 0.7},
            structure="70/15/15 split.",
            tags=["baseline"],
            store=store,
            capture_git=False,
            capture_env=False,
        )
        exp_dir = store.experiment_dir(record.experiment_id)
        assert (exp_dir / "manifest.json").exists()
        assert (exp_dir / "README.md").exists()

        loaded = store.load(record.experiment_id)
        assert loaded.name == "Baseline"
        assert loaded.metrics == {"r2_mean": 0.7}

    def test_index_lists_experiment(self, tmp_path):
        store = ExperimentStore(tmp_path)
        log_experiment(
            name="Run A",
            metrics={"r2_mean": 0.9},
            store=store,
            capture_git=False,
            capture_env=False,
        )
        index = (tmp_path / "README.md").read_text()
        assert "Run A" in index
        assert "r2_mean" in index

    def test_figures_are_copied(self, tmp_path):
        # Create a fake figure file.
        fig = tmp_path / "plot.png"
        fig.write_bytes(b"\x89PNG\r\n\x1a\n fake")
        store = ExperimentStore(tmp_path / "experiments")
        record = log_experiment(
            name="With figure",
            figures=[fig],
            store=store,
            capture_git=False,
            capture_env=False,
        )
        exp_dir = store.experiment_dir(record.experiment_id)
        assert (exp_dir / "figures" / "plot.png").exists()
        assert "figures/plot.png" in record.figures

    def test_missing_figure_is_skipped(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = log_experiment(
            name="Missing figure",
            figures=["/does/not/exist.png"],
            store=store,
            capture_git=False,
            capture_env=False,
        )
        assert record.figures == []

    def test_overwrite_guard(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = ExperimentRecord.create(
            name="dup", experiment_id="dup", capture_git=False, capture_env=False
        )
        store.save(record)
        with pytest.raises(FileExistsError):
            store.save(record, overwrite=False)
        # Overwrite succeeds.
        store.save(record, overwrite=True)

    def test_list_records_skips_non_experiment_dirs(self, tmp_path):
        store = ExperimentStore(tmp_path)
        (tmp_path / "not-an-experiment").mkdir()
        log_experiment(name="Real", store=store, capture_git=False, capture_env=False)
        records = store.list_records()
        assert len(records) == 1
        assert records[0].name == "Real"
