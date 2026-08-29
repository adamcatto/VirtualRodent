"""
Training script for Virtual Rodent neural decoding models.

Uses Hydra for configuration management. Run with:
    python scripts/train.py
    python scripts/train.py model=lstm_neural_only
    python scripts/train.py data=per_animal
    python scripts/train.py experiment=ablation
"""

import sys
from pathlib import Path

# Get project root (scripts/ parent directory) - must be before hydra changes cwd
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from omegaconf.base import ContainerMetadata
from omegaconf.nodes import ValueNode
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import json

# Allow OmegaConf types in checkpoint loading (PyTorch 2.6+ requires explicit allowlist)
torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata, ValueNode])

from virtual_rodent.models.rodent_agent import RodentAgent
from virtual_rodent.data.per_session_datamodule import PerSessionDataModule
from virtual_rodent.data.per_animal_datamodule import PerAnimalDataModule


def save_results(trainer, model, dm, cfg, output_dir):
    """Save training results for later analysis and website."""
    results = {
        "experiment_name": cfg.experiment_name,
        "strategy": cfg.data.strategy,
        "model_type": cfg.model.rnn_type,
        "input_mode": cfg.model.input_mode,
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score),
        "best_model_path": str(trainer.checkpoint_callback.best_model_path),
        "test_metrics": {k: float(v) for k, v in trainer.callback_metrics.items() if "test/" in k},
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    output_file = output_dir / "training_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")

    # Record a git-tracked provenance entry under experiments/ so the run
    # (commit, config, results) is reproducible and reviewable outside outs/.
    record_provenance(results, cfg)


def record_provenance(results, cfg):
    """Write a git-tracked provenance record for this training run.

    Curates the small, reviewable parts of the run (description, exact commit,
    full config, headline metrics, checkpoint reference) into
    ``experiments/<experiment_name>/`` via ``virtual_rodent.provenance``.
    Failure here must never abort a completed training run, so errors are
    caught and reported.
    """
    try:
        from virtual_rodent.provenance import log_experiment

        metrics = {"best_val_loss": results["best_val_loss"], **results["test_metrics"]}
        description = (
            f"{cfg.model.rnn_type.upper()} {cfg.model.input_mode} decoder trained with "
            f"the `{cfg.data.strategy}` strategy.\n\n"
            f"Neural history: {cfg.data.neural_history} frames "
            f"({cfg.data.neural_history / 50:.1f}s at 50Hz); "
            f"pose horizon: {cfg.data.pose_horizon} frames "
            f"({cfg.data.pose_horizon / 50:.1f}s)."
        )
        structure = {
            "split_strategy": cfg.data.strategy,
            "model": {
                "rnn_type": cfg.model.rnn_type,
                "input_mode": cfg.model.input_mode,
                "hidden_dim": cfg.model.hidden_dim,
                "num_layers": cfg.model.num_layers,
                "bidirectional": cfg.model.bidirectional,
            },
            "neural_history": cfg.data.neural_history,
            "pose_history": cfg.data.pose_history,
            "pose_horizon": cfg.data.pose_horizon,
        }
        record = log_experiment(
            name=cfg.experiment_name,
            experiment_id=cfg.experiment_name,
            description=description,
            structure=structure,
            config=results["config"],
            metrics=metrics,
            tags=[cfg.data.strategy, cfg.model.rnn_type, cfg.model.input_mode],
            artifacts={"checkpoint": results["best_model_path"]},
        )
        print(f"Provenance record written: experiments/{record.experiment_id}/")
    except Exception as exc:  # pragma: no cover - provenance must not break training
        print(f"[provenance] warning: failed to record experiment: {exc}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("VirtualRodent Neural Decoding Training")
    print("=" * 80)
    print(f"\nProject root: {PROJECT_ROOT}")
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Create output directories (use absolute paths from project root)
    checkpoint_dir = PROJECT_ROOT / cfg.paths.checkpoints / cfg.experiment_name
    results_dir = PROJECT_ROOT / cfg.paths.results / cfg.experiment_name
    logs_dir = PROJECT_ROOT / cfg.paths.logs
    data_dir = PROJECT_ROOT / cfg.data.data_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create DataModule based on strategy
    print(f"\nInitializing DataModule with strategy: {cfg.data.strategy}")

    if cfg.data.strategy == "per_session":
        dm = PerSessionDataModule(
            data_dir=str(data_dir),
            brain_regions=cfg.data.brain_regions,
            animals=cfg.data.animals,
            neural_history=cfg.data.neural_history,
            pose_history=cfg.data.pose_history,
            pose_horizon=cfg.data.pose_horizon,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            split_ratios=cfg.data.split_ratios,
            seed=cfg.seed,
        )
    elif cfg.data.strategy == "per_animal":
        dm = PerAnimalDataModule(
            data_dir=str(data_dir),
            brain_regions=cfg.data.brain_regions,
            animals=cfg.data.animals,
            neural_history=cfg.data.neural_history,
            pose_history=cfg.data.pose_history,
            pose_horizon=cfg.data.pose_horizon,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unknown data strategy: {cfg.data.strategy}")

    # Setup data
    dm.setup("fit")

    print(f"\nDataModule setup complete:")
    print(dm)

    # Create model
    print(f"\nInitializing model:")
    print(f"  Architecture: {cfg.model.rnn_type.upper()}")
    print(f"  Input mode: {cfg.model.input_mode}")
    print(f"  Hidden dim: {cfg.model.hidden_dim}")
    print(f"  Num layers: {cfg.model.num_layers}")
    print(f"  Bidirectional: {cfg.model.bidirectional}")

    model = RodentAgent(
        input_dim_neural=dm.input_dim,
        input_dim_pose=69,  # Flattened keypoints
        output_dim=69,
        neural_history=cfg.data.neural_history,
        pose_history=cfg.data.pose_history,
        pose_horizon=cfg.data.pose_horizon,
        **cfg.model,
    )

    # Callbacks (use absolute paths)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=cfg.trainer.patience,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, early_stopping, lr_monitor]

    # Logger (use absolute path)
    logger = TensorBoardLogger(
        save_dir=str(logs_dir),
        name=cfg.experiment_name,
        default_hp_metric=False,
    )

    # Trainer
    print(f"\nInitializing Trainer:")
    print(f"  Max epochs: {cfg.trainer.max_epochs}")
    print(f"  Accelerator: {cfg.trainer.accelerator}")
    print(f"  Devices: {cfg.trainer.devices}")

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=True,
        val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches=cfg.trainer.limit_val_batches,
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    trainer.fit(model, dm)

    # Test - manually load best checkpoint to avoid PyTorch 2.6 weights_only issue
    print("\n" + "=" * 80)
    print("Running test evaluation...")
    print("=" * 80 + "\n")

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path:
        checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    trainer.test(model, dm)

    # Save results
    save_results(trainer, model, dm, cfg, results_dir)

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    train()
