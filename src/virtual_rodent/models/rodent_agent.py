"""
Rodent Agent: LSTM-based neural decoding model for pose prediction.

This module implements a PyTorch Lightning model that predicts future pose
trajectories from neural signals and optionally past pose history.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from sklearn.metrics import r2_score


class RodentAgent(pl.LightningModule):
    """
    LSTM-based neural decoding agent for pose prediction.

    This model predicts future 3D pose trajectories from neural signals
    and optionally past pose history. It uses:
    - Bidirectional LSTM to encode neural activity history
    - Optional bidirectional LSTM to encode pose history (multimodal mode)
    - Autoregressive LSTM decoder to generate future pose frames
    - Combined position + velocity loss for smooth predictions

    Input modes:
    - 'multimodal': Uses both neural and pose history (default)
    - 'neural_only': Uses only neural signals (pure decoding)
    - 'pose_only': Uses only pose history (ablation control)

    Example:
        >>> model = RodentAgent(
        ...     input_dim_neural=256,
        ...     input_dim_pose=69,
        ...     output_dim=69,
        ...     neural_history=750,
        ...     pose_horizon=250,
        ...     input_mode='multimodal'
        ... )
        >>> neural = torch.randn(batch, 750, 256)
        >>> pose_history = torch.randn(batch, 750, 69)
        >>> predictions = model(neural, pose_history)  # (batch, 250, 69)
    """

    def __init__(
        self,
        input_dim_neural: int,
        input_dim_pose: int,
        output_dim: int,
        neural_history: int = 750,  # 15 seconds at 50Hz
        pose_history: int = 750,
        pose_horizon: int = 250,  # 5 seconds at 50Hz
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        rnn_type: str = "lstm",  # or "gru"
        input_mode: str = "multimodal",  # or "neural_only", "pose_only"
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        lambda_velocity: float = 0.1,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
    ):
        """
        Initialize the Rodent Agent.

        Args:
            input_dim_neural: Dimension of neural input (num_neurons)
            input_dim_pose: Dimension of pose input (typically 69 for flattened keypoints)
            output_dim: Dimension of pose output (typically 69)
            neural_history: Number of past frames in neural input
            pose_history: Number of past frames in pose input
            pose_horizon: Number of future frames to predict
            hidden_dim: Hidden dimension for LSTM layers
            num_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM for encoders
            rnn_type: Type of RNN ('lstm' or 'gru')
            input_mode: Input configuration ('multimodal', 'neural_only', 'pose_only')
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            lambda_velocity: Weight for velocity loss component
            weight_decay: L2 regularization weight
            scheduler_patience: Patience for ReduceLROnPlateau scheduler
            scheduler_factor: Factor for learning rate reduction
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Validate input mode
        valid_modes = ['multimodal', 'neural_only', 'pose_only']
        if input_mode not in valid_modes:
            raise ValueError(f"input_mode must be one of {valid_modes}, got '{input_mode}'")

        # Select RNN class
        RNN = nn.LSTM if rnn_type == 'lstm' else nn.GRU

        # Build neural encoder (if used)
        if input_mode in ['multimodal', 'neural_only']:
            self.neural_encoder = RNN(
                input_dim_neural,
                hidden_dim,
                num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
            neural_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        else:
            self.neural_encoder = None
            neural_output_dim = 0

        # Build pose encoder (if used)
        if input_mode in ['multimodal', 'pose_only']:
            # Project pose to intermediate dimension
            self.pose_projection = nn.Linear(input_dim_pose, 128)

            self.pose_encoder = RNN(
                128,
                64,
                1,  # Single layer for pose encoding
                bidirectional=bidirectional,
                batch_first=True,
            )
            pose_output_dim = 128 if bidirectional else 64
        else:
            self.pose_projection = None
            self.pose_encoder = None
            pose_output_dim = 0

        # Fusion dimension
        self.fusion_dim = neural_output_dim + pose_output_dim

        if self.fusion_dim == 0:
            raise ValueError("At least one input mode must be enabled")

        # Build decoder
        self.decoder_rnn = RNN(
            output_dim,  # Decoder input is previous predicted pose
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Initial decoder input projection (from fused encoding)
        self.init_projection = nn.Linear(self.fusion_dim, output_dim)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Prediction storage for saving during test
        self.test_predictions = []
        self.predictions_save_path = None

    def encode_neural(self, neural: torch.Tensor) -> torch.Tensor:
        """
        Encode neural activity history.

        Args:
            neural: (batch, neural_history, input_dim_neural)

        Returns:
            (batch, neural_output_dim) encoding
        """
        # Run through BiLSTM
        output, (h_n, c_n) = self.neural_encoder(neural)

        # Concatenate final hidden states from both directions
        if self.hparams.bidirectional:
            # h_n: (num_layers * 2, batch, hidden_dim)
            # Take last layer's forward and backward hidden states
            encoding = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            encoding = h_n[-1]

        return encoding

    def encode_pose(self, pose_history: torch.Tensor) -> torch.Tensor:
        """
        Encode pose history.

        Args:
            pose_history: (batch, pose_history, input_dim_pose)

        Returns:
            (batch, pose_output_dim) encoding
        """
        # Project to intermediate dimension
        pose_proj = F.relu(self.pose_projection(pose_history))

        # Encode with BiLSTM
        output, (h_p, c_p) = self.pose_encoder(pose_proj)

        # Concatenate final hidden states
        if self.hparams.bidirectional:
            encoding = torch.cat([h_p[-2], h_p[-1]], dim=1)
        else:
            encoding = h_p[-1]

        return encoding

    def decode_autoregressive(
        self,
        initial_input: torch.Tensor,
        num_steps: int,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Autoregressively decode future pose frames.

        Args:
            initial_input: (batch, output_dim) - First decoder input
            num_steps: Number of frames to generate
            hidden: Optional initial hidden state for decoder

        Returns:
            (batch, num_steps, output_dim) predicted pose sequence
        """
        batch_size = initial_input.shape[0]
        predictions = []

        # First input
        decoder_input = initial_input.unsqueeze(1)  # (batch, 1, output_dim)

        for t in range(num_steps):
            # Decode one step
            out, hidden = self.decoder_rnn(decoder_input, hidden)

            # Project to output
            pred = self.output_projection(out)  # (batch, 1, output_dim)
            predictions.append(pred)

            # Use prediction as next input (free-running)
            decoder_input = pred

        return torch.cat(predictions, dim=1)

    def decode_teacher_forcing(
        self,
        initial_input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode with teacher forcing (used during training).

        Args:
            initial_input: (batch, output_dim) - First decoder input
            target: (batch, num_steps, output_dim) - Ground truth for teacher forcing

        Returns:
            (batch, num_steps, output_dim) predicted pose sequence
        """
        # Prepend initial input to target sequence
        # decoder_inputs: (batch, num_steps, output_dim)
        decoder_inputs = torch.cat(
            [initial_input.unsqueeze(1), target[:, :-1]], dim=1
        )

        # Run through decoder
        output, _ = self.decoder_rnn(decoder_inputs)

        # Project to output dimension
        predictions = self.output_projection(output)

        return predictions

    def forward(
        self,
        neural: Optional[torch.Tensor] = None,
        pose_history: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            neural: (batch, neural_history, input_dim_neural) or None
            pose_history: (batch, pose_history, input_dim_pose) or None
            target: (batch, pose_horizon, output_dim) or None (for teacher forcing)

        Returns:
            (batch, pose_horizon, output_dim) predicted pose sequence
        """
        # Encode inputs
        encodings = []

        if neural is not None and self.neural_encoder is not None:
            neural_encoding = self.encode_neural(neural)
            encodings.append(neural_encoding)

        if pose_history is not None and self.pose_encoder is not None:
            pose_encoding = self.encode_pose(pose_history)
            encodings.append(pose_encoding)

        if not encodings:
            raise ValueError("At least one input (neural or pose_history) must be provided")

        # Fuse encodings
        fused = torch.cat(encodings, dim=1)  # (batch, fusion_dim)

        # Project to initial decoder input
        initial_input = self.init_projection(fused)  # (batch, output_dim)

        # Decode
        if self.training and target is not None:
            # Use teacher forcing during training
            predictions = self.decode_teacher_forcing(initial_input, target)
        else:
            # Autoregressive generation during inference
            predictions = self.decode_autoregressive(
                initial_input, self.hparams.pose_horizon
            )

        return predictions

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined position + velocity loss.

        Args:
            pred: (batch, horizon, output_dim) predictions
            target: (batch, horizon, output_dim) ground truth

        Returns:
            Dict with 'total', 'position', and 'velocity' losses
        """
        # Position loss (MSE on positions)
        position_loss = F.mse_loss(pred, target)

        # Velocity loss (penalize non-smooth trajectories)
        if pred.shape[1] > 1:
            pred_vel = pred[:, 1:] - pred[:, :-1]
            target_vel = target[:, 1:] - target[:, :-1]
            velocity_loss = F.mse_loss(pred_vel, target_vel)
        else:
            velocity_loss = torch.tensor(0.0, device=pred.device)

        # Combined loss
        total_loss = position_loss + self.hparams.lambda_velocity * velocity_loss

        return {
            'total': total_loss,
            'position': position_loss,
            'velocity': velocity_loss,
        }

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics.

        Args:
            pred: (batch, horizon, output_dim) predictions
            target: (batch, horizon, output_dim) ground truth

        Returns:
            Dict with various metrics
        """
        # Convert to numpy for sklearn metrics
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Flatten batch and time dimensions
        pred_flat = pred_np.reshape(-1, pred_np.shape[-1])
        target_flat = target_np.reshape(-1, target_np.shape[-1])

        # R² per coordinate
        r2_scores = []
        for i in range(pred_flat.shape[1]):
            r2 = r2_score(target_flat[:, i], pred_flat[:, i])
            r2_scores.append(r2)

        # Mean absolute error
        mae = torch.mean(torch.abs(pred - target))

        # Root mean squared error
        rmse = torch.sqrt(F.mse_loss(pred, target))

        return {
            'mae': mae,
            'rmse': rmse,
            'r2_mean': torch.tensor(np.mean(r2_scores)),
        }

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Unpack batch based on dataset configuration
        if len(batch) == 3:
            # (neural, pose_history, pose_target) from dataset - multimodal format
            neural, pose_history, pose_target = batch
        elif len(batch) == 2:
            # (neural, pose_target) - neural-only format
            neural, pose_target = batch
            pose_history = None
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        # Forward pass
        pred = self(
            neural if self.hparams.input_mode in ['multimodal', 'neural_only'] else None,
            pose_history if self.hparams.input_mode in ['multimodal', 'pose_only'] else None,
            target=pose_target,
        )

        # Compute loss
        losses = self.compute_loss(pred, pose_target)

        # Log metrics
        self.log('train/loss', losses['total'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/position_loss', losses['position'], on_step=False, on_epoch=True)
        self.log('train/velocity_loss', losses['velocity'], on_step=False, on_epoch=True)

        return losses['total']

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        """Validation step."""
        # Unpack batch based on dataset configuration
        if len(batch) == 3:
            # (neural, pose_history, pose_target) from dataset - multimodal format
            neural, pose_history, pose_target = batch
        elif len(batch) == 2:
            # (neural, pose_target) - neural-only format
            neural, pose_target = batch
            pose_history = None
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        # Forward pass (no teacher forcing)
        pred = self(
            neural if self.hparams.input_mode in ['multimodal', 'neural_only'] else None,
            pose_history if self.hparams.input_mode in ['multimodal', 'pose_only'] else None,
            target=None,
        )

        # Compute loss and metrics
        losses = self.compute_loss(pred, pose_target)
        metrics = self.compute_metrics(pred, pose_target)

        # Store for epoch end
        self.validation_step_outputs.append({
            'loss': losses['total'],
            'position_loss': losses['position'],
            'velocity_loss': losses['velocity'],
            **metrics,
        })

        # Log metrics
        self.log('val/loss', losses['total'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/position_loss', losses['position'], on_step=False, on_epoch=True)
        self.log('val/velocity_loss', losses['velocity'], on_step=False, on_epoch=True)
        self.log('val/mae', metrics['mae'], on_step=False, on_epoch=True)
        self.log('val/rmse', metrics['rmse'], on_step=False, on_epoch=True)
        self.log('val/r2', metrics['r2_mean'], on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """Aggregate validation metrics."""
        if not self.validation_step_outputs:
            return

        # Clear outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        """Test step."""
        # Unpack batch based on dataset configuration
        if len(batch) == 3:
            # (neural, pose_history, pose_target) from dataset - multimodal format
            neural, pose_history, pose_target = batch
        elif len(batch) == 2:
            # (neural, pose_target) - neural-only format
            neural, pose_target = batch
            pose_history = None
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        # Forward pass
        pred = self(
            neural if self.hparams.input_mode in ['multimodal', 'neural_only'] else None,
            pose_history if self.hparams.input_mode in ['multimodal', 'pose_only'] else None,
            target=None,
        )

        # Compute loss and metrics
        losses = self.compute_loss(pred, pose_target)
        metrics = self.compute_metrics(pred, pose_target)

        # Store for epoch end
        self.test_step_outputs.append({
            'loss': losses['total'],
            'position_loss': losses['position'],
            'velocity_loss': losses['velocity'],
            **metrics,
        })

        # Store predictions for saving (detach and move to CPU)
        self.test_predictions.append({
            'predictions': pred.detach().cpu().numpy(),
            'ground_truth': pose_target.detach().cpu().numpy(),
            'neural': neural.detach().cpu().numpy() if neural is not None else None,
            'pose_history': pose_history.detach().cpu().numpy() if pose_history is not None else None,
        })

        # Log metrics
        self.log('test/loss', losses['total'], on_step=False, on_epoch=True)
        self.log('test/position_loss', losses['position'], on_step=False, on_epoch=True)
        self.log('test/mae', metrics['mae'], on_step=False, on_epoch=True)
        self.log('test/rmse', metrics['rmse'], on_step=False, on_epoch=True)
        self.log('test/r2', metrics['r2_mean'], on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        """Aggregate test metrics and save predictions."""
        if not self.test_step_outputs:
            return

        # Save predictions if path is set
        if self.predictions_save_path and self.test_predictions:
            self._save_predictions()

        # Clear outputs
        self.test_step_outputs.clear()
        self.test_predictions.clear()

    def _save_predictions(self):
        """Save test predictions to file."""
        import os

        # Concatenate all batches
        all_predictions = np.concatenate([p['predictions'] for p in self.test_predictions], axis=0)
        all_ground_truth = np.concatenate([p['ground_truth'] for p in self.test_predictions], axis=0)

        # Neural data (may be large, so only save a subset)
        neural_data = [p['neural'] for p in self.test_predictions if p['neural'] is not None]
        if neural_data:
            all_neural = np.concatenate(neural_data, axis=0)
        else:
            all_neural = np.array([])

        # Create predictions directory
        pred_dir = os.path.dirname(self.predictions_save_path)
        os.makedirs(pred_dir, exist_ok=True)

        # Save as npz
        np.savez_compressed(
            self.predictions_save_path,
            predictions=all_predictions,
            ground_truth=all_ground_truth,
            neural=all_neural,
        )
        print(f"Predictions saved to: {self.predictions_save_path}")
        print(f"  Predictions shape: {all_predictions.shape}")
        print(f"  Ground truth shape: {all_ground_truth.shape}")
        print(f"  Neural shape: {all_neural.shape}")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1,
            },
        }

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Prediction step for generating predictions.

        Returns both predictions and ground truth for visualization.
        """
        # Unpack batch based on dataset configuration
        if len(batch) == 3:
            # (neural, pose_history, pose_target) from dataset - multimodal format
            neural, pose_history, pose_target = batch
        elif len(batch) == 2:
            # (neural, pose_target) - neural-only format
            neural, pose_target = batch
            pose_history = None
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        # Generate predictions
        pred = self(
            neural if self.hparams.input_mode in ['multimodal', 'neural_only'] else None,
            pose_history if self.hparams.input_mode in ['multimodal', 'pose_only'] else None,
            target=None,
        )

        return {
            'predictions': pred,
            'ground_truth': pose_target,
            'neural': neural if neural is not None else torch.tensor([]),
            'pose_history': pose_history if pose_history is not None else torch.tensor([]),
        }
