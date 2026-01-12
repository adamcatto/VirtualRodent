"""
Evaluation script for trained VirtualRodent models.

Generates predictions, computes metrics, and creates visualizations.

Usage:
    python scripts/evaluate.py checkpoint_path=outs/checkpoints/experiment_name/best.ckpt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm

from virtual_rodent.models.rodent_agent import RodentAgent
from virtual_rodent.data.per_session_datamodule import PerSessionDataModule
from virtual_rodent.data.per_animal_datamodule import PerAnimalDataModule
from virtual_rodent.analysis.metrics import (
    compute_session_metrics,
    aggregate_per_animal_metrics,
    save_metrics_plots,
)


def generate_predictions(model, dataloader, device):
    """Generate predictions for entire dataset."""
    model.eval()
    model.to(device)

    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            neural, pose = batch
            neural = neural.to(device)
            pose = pose.to(device)

            # Generate predictions
            pred = model(neural, None)  # Assuming neural-only for simplicity

            all_predictions.append(pred.cpu().numpy())
            all_ground_truth.append(pose.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)

    return predictions, ground_truth


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig):
    """
    Main evaluation function.

    Args:
        cfg: Hydra configuration (must include checkpoint_path)
    """
    print("=" * 80)
    print("VirtualRodent Model Evaluation")
    print("=" * 80)

    # Load checkpoint
    checkpoint_path = cfg.get('checkpoint_path')
    if not checkpoint_path:
        raise ValueError("Must specify checkpoint_path in config or command line")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading model from: {checkpoint_path}")
    model = RodentAgent.load_from_checkpoint(checkpoint_path)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    results_dir = Path(cfg.paths.results) / cfg.experiment_name
    viz_dir = Path(cfg.paths.visualizations)
    results_dir.mkdir(parents=True, exist_ok=True)
    (viz_dir / "plots").mkdir(parents=True, exist_ok=True)
    (viz_dir / "videos").mkdir(parents=True, exist_ok=True)
    (viz_dir / "neural").mkdir(parents=True, exist_ok=True)

    # Create DataModule
    print(f"\nSetting up DataModule with strategy: {cfg.data.strategy}")

    if cfg.data.strategy == "per_session":
        dm = PerSessionDataModule(**cfg.data)
    elif cfg.data.strategy == "per_animal":
        dm = PerAnimalDataModule(**cfg.data)
    else:
        raise ValueError(f"Unknown strategy: {cfg.data.strategy}")

    dm.setup("test")

    # Generate predictions on test set
    print("\nGenerating predictions on test set...")
    test_loader = dm.test_dataloader()
    predictions, ground_truth = generate_predictions(model, test_loader, device)

    print(f"Generated {len(predictions)} predictions")

    # Save predictions
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        predictions_dir / "test_predictions.npz",
        predictions=predictions,
        ground_truth=ground_truth,
    )
    print(f"Predictions saved to: {predictions_dir}/test_predictions.npz")

    # Compute overall metrics
    print("\nComputing metrics...")
    from virtual_rodent.analysis.metrics import compute_session_metrics

    overall_metrics = compute_session_metrics(
        predictions,
        ground_truth,
        session_id="overall",
        output_dir=results_dir,
    )

    print("\nOverall Test Metrics:")
    print(f"  MSE: {overall_metrics['mse']:.6f}")
    print(f"  MAE: {overall_metrics['mae']:.6f}")
    print(f"  RMSE: {overall_metrics['rmse']:.6f}")
    print(f"  R² (mean): {overall_metrics['r2_mean']:.4f}")
    print(f"  Velocity MSE: {overall_metrics['velocity_mse']:.6f}")

    # Generate plots
    print("\nGenerating visualization plots...")
    save_metrics_plots(
        all_session_metrics=[overall_metrics],
        animal_metrics={},
        output_dir=viz_dir / "plots",
    )

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Results saved to: {results_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate()
