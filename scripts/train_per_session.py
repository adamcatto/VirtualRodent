"""
Per-session training script for Virtual Rodent neural decoding models.

Trains a separate model for each session in the dataset.

Usage:
    # Train all sessions
    python scripts/train_per_session.py

    # Train specific session
    python scripts/train_per_session.py session_id=Rat_A_Day1

    # Train with different model config
    python scripts/train_per_session.py model=lstm_neural_only
"""

import sys
from pathlib import Path

# Get project root (scripts/ parent directory) - must be before hydra changes cwd
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gc
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
from virtual_rodent.data.single_session_datamodule import SingleSessionDataModule, get_all_session_ids


def clear_memory():
    """Clear GPU/MPS memory between sessions to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        # MPS doesn't have empty_cache, but gc.collect helps
        pass


def train_single_session(cfg: DictConfig, session_id: str, project_root: Path):
    """
    Train a model on a single session.

    Args:
        cfg: Hydra configuration
        session_id: ID of the session to train on
        project_root: Absolute path to project root directory

    Returns:
        Dict with training results
    """
    print(f"\n{'='*80}")
    print(f"Training session: {session_id}")
    print(f"{'='*80}")

    # Create output directories for this session (use absolute paths from project root)
    experiment_name = f"{cfg.experiment_name}/{session_id}"
    checkpoint_dir = project_root / cfg.paths.checkpoints / experiment_name
    results_dir = project_root / cfg.paths.results / experiment_name
    logs_dir = project_root / cfg.paths.logs
    data_dir = project_root / cfg.data.data_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create DataModule for single session
    dm = SingleSessionDataModule(
        data_dir=str(data_dir),
        session_id=session_id,
        neural_history=cfg.data.neural_history,
        pose_history=cfg.data.pose_history,
        pose_horizon=cfg.data.pose_horizon,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        split_ratios=cfg.data.split_ratios,
        seed=cfg.seed,
    )

    # Setup data
    dm.setup("fit")
    print(f"\n{dm}")

    # Check if we have enough data
    if len(dm.train_dataset) < cfg.data.batch_size:
        print(f"Warning: Not enough training samples ({len(dm.train_dataset)}). Skipping session.")
        return None

    # Create model
    model = RodentAgent(
        input_dim_neural=dm.input_dim,
        input_dim_pose=69,  # Flattened keypoints
        output_dim=69,
        neural_history=cfg.data.neural_history,
        pose_history=cfg.data.pose_history,
        pose_horizon=cfg.data.pose_horizon,
        **cfg.model,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=False,
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=cfg.trainer.patience,
        mode="min",
        verbose=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, early_stopping, lr_monitor]

    # Logger (use absolute path)
    logger = TensorBoardLogger(
        save_dir=str(logs_dir),
        name=experiment_name,
        default_hp_metric=False,
    )

    # Trainer
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
        enable_progress_bar=True,
        max_steps=cfg.trainer.max_steps if 'max_steps' in cfg.trainer else None,
        limit_test_batches=cfg.trainer.limit_test_batches if 'limit_test_batches' in cfg.trainer else None,
    )

    # Train
    trainer.fit(model, dm)

    # Test - manually load best checkpoint to avoid PyTorch 2.6 weights_only issue
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path:
        # Load checkpoint with weights_only=False to handle OmegaConf objects
        checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    # Set predictions save path before running test
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model.predictions_save_path = str(predictions_dir / "test_predictions.npz")

    trainer.test(model, dm)

    # Collect results
    results = {
        "session_id": session_id,
        "experiment_name": experiment_name,
        "strategy": "per_session",
        "model_type": cfg.model.rnn_type,
        "input_mode": cfg.model.input_mode,
        "session_metadata": dm.session_metadata,
        "num_train_samples": len(dm.train_dataset),
        "num_val_samples": len(dm.val_dataset),
        "num_test_samples": len(dm.test_dataset),
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score) if trainer.checkpoint_callback.best_model_score else None,
        "best_model_path": str(trainer.checkpoint_callback.best_model_path) if trainer.checkpoint_callback.best_model_path else None,
        "test_metrics": {k: float(v) for k, v in trainer.callback_metrics.items() if 'test/' in k},
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    # Save results
    output_file = results_dir / "training_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSession {session_id} complete:")
    print(f"  Best val loss: {results['best_val_loss']:.4f}")
    print(f"  Test metrics: {results['test_metrics']}")
    print(f"  Results saved to: {output_file}")

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    """
    Main training function for per-session training.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("VirtualRodent Per-Session Neural Decoding Training")
    print("=" * 80)
    print(f"\nProject root: {PROJECT_ROOT}")
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Use absolute path for data directory
    data_dir = PROJECT_ROOT / cfg.data.data_dir

    # Check if specific session is requested
    if hasattr(cfg, 'session_id') and cfg.session_id is not None:
        session_ids = [cfg.session_id]
        print(f"\nTraining single session: {cfg.session_id}")
    else:
        # Get all session IDs
        session_ids = get_all_session_ids(str(data_dir))
        print(f"\nFound {len(session_ids)} sessions to train")

    # Train each session
    all_results = []
    successful = 0
    failed = 0

    for i, session_id in enumerate(session_ids):
        print(f"\n[{i+1}/{len(session_ids)}] Processing session: {session_id}")

        try:
            results = train_single_session(cfg, session_id, PROJECT_ROOT)
            if results:
                all_results.append(results)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error training session {session_id}: {e}")
            failed += 1

        # Clear memory between sessions to prevent OOM
        clear_memory()
        print("  Memory cleared.")

    # Save summary results (use absolute path)
    summary_dir = PROJECT_ROOT / cfg.paths.results / cfg.experiment_name
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_name": cfg.experiment_name,
        "strategy": "per_session",
        "total_sessions": len(session_ids),
        "successful": successful,
        "failed": failed,
        "session_results": [
            {
                "session_id": r["session_id"],
                "best_val_loss": r["best_val_loss"],
                "test_metrics": r["test_metrics"],
                "session_metadata": r["session_metadata"],
            }
            for r in all_results
        ],
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    summary_file = summary_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("Per-Session Training Complete!")
    print("=" * 80)
    print(f"\nSuccessful: {successful}/{len(session_ids)}")
    print(f"Failed: {failed}/{len(session_ids)}")
    print(f"\nSummary saved to: {summary_file}")

    # Print per-session summary
    if all_results:
        print("\nPer-Session Results:")
        print("-" * 60)
        for r in all_results:
            val_loss = r["best_val_loss"]
            test_loss = r["test_metrics"].get("test/loss", 0)
            test_r2 = r["test_metrics"].get("test/r2", 0)
            print(f"  {r['session_id']:30s} val={val_loss:.4f} test={test_loss:.4f} R2={test_r2:.4f}")


if __name__ == "__main__":
    train()
