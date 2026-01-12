# VirtualRodent Neural Decoding System

Complete implementation of a neural-to-pose prediction pipeline with LSTM-based models, dual evaluation strategies, and interactive visualization website.

## 🎯 Overview

This system enables:
- **Pose prediction** from neural signals (15 seconds → 5 seconds future)
- **Dual evaluation strategies**: Per-session temporal splits and per-animal chronological splits
- **Multimodal learning**: Configurable neural-only, pose-only, or multimodal input
- **Interactive website**: Real-time pose playback, metrics dashboards, neural visualizations
- **Comprehensive metrics**: Position MSE, R², velocity smoothness, temporal coherence

## 📁 Project Structure

```
VirtualRodent/
├── configs/                          # Hydra configurations
│   ├── config.yaml                  # Main config
│   ├── model/                       # Model configs (LSTM/GRU, multimodal/neural-only)
│   ├── data/                        # Data configs (per-session/per-animal)
│   └── experiment/                  # Experiment configs
│
├── src/virtual_rodent/
│   ├── environment/
│   │   └── session_env.py          # Session management with temporal splits
│   ├── models/
│   │   └── rodent_agent.py         # LSTM-based LightningModule
│   ├── data/
│   │   ├── per_session_datamodule.py
│   │   └── per_animal_datamodule.py
│   ├── visualization/
│   │   ├── animations.py           # Pose animation generation
│   │   └── neural_viz.py           # Neural population visualizations
│   ├── analysis/
│   │   └── metrics.py              # Metrics computation and plotting
│   └── website/
│       ├── app.py                  # Flask backend
│       └── frontend/               # React frontend (Vite + Three.js)
│
├── scripts/
│   ├── train.py                    # Training script with Hydra
│   └── evaluate.py                 # Evaluation and visualization generation
│
└── outs/                           # Generated outputs
    ├── checkpoints/                # Model checkpoints
    ├── results/                    # Metrics JSON files
    └── visualizations/             # Plots, videos, neural viz
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies (already in pyproject.toml)
pip install -e ".[dev,website]"

# Install React frontend dependencies
cd src/website/frontend
npm install
cd ../../..
```

### 2. Training

Train a model with default configuration (per-session, LSTM multimodal):

```bash
python scripts/train.py
```

Train with different configurations:

```bash
# Per-animal evaluation
python scripts/train.py data=per_animal

# Neural-only input (pure decoding)
python scripts/train.py model=lstm_neural_only

# GRU instead of LSTM
python scripts/train.py model=gru_multimodal

# Custom experiment name
python scripts/train.py experiment_name=my_experiment
```

### 3. Evaluation

After training, evaluate the model:

```bash
python scripts/evaluate.py checkpoint_path=outs/checkpoints/EXPERIMENT_NAME/best.ckpt
```

This will:
- Generate predictions on the test set
- Compute comprehensive metrics
- Save results to `outs/results/`
- Generate visualization plots in `outs/visualizations/`

### 4. Website

Launch the visualization website:

**Terminal 1 - Backend (Flask):**
```bash
python src/website/app.py
```

**Terminal 2 - Frontend (React):**
```bash
cd src/website/frontend
npm run dev
```

Open browser to `http://localhost:3000`

## 📊 Model Architecture

### RodentAgent (LSTM-based)

**Encoder:**
- **Neural pathway**: BiLSTM(256, 2 layers) encodes 750 frames (15s) of neural activity
- **Pose pathway**: Linear(128) + BiLSTM(64) encodes 750 frames of pose history
- **Fusion**: Concatenate encodings (640-dim)

**Decoder:**
- Autoregressive LSTM generates 250 future pose frames (5 seconds)
- Teacher forcing during training, free-running during inference

**Loss:**
- Position MSE + λ×Velocity MSE (λ=0.1 default)
- Encourages temporally smooth predictions

**Input Modes:**
- `multimodal` (default): neural + pose → pose
- `neural_only`: neural → pose (pure decoding)
- `pose_only`: pose → pose (ablation control)

## 🎓 Evaluation Strategies

### Per-Session Temporal Splits

Each session is split temporally:
- **60% train** / **1% gap** / **19% val** / **1% gap** / **19% test**
- Gap regions prevent temporal leakage
- Tests model on later time points within same session

### Per-Animal Chronological Splits

Sessions split chronologically per animal:
- **First 60% sessions → train**
- **Middle 20% → val**
- **Last 20% → test**
- Tests generalization to later recording sessions
- Critical for neuroscience applications (neural drift over time)

## 📈 Metrics

### Position Metrics
- **MSE**: Mean squared error on 3D positions
- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error
- **R²**: Per-coordinate coefficient of determination

### Temporal Metrics
- **Velocity MSE**: Smoothness of predicted trajectories
- **Temporal Coherence**: Measures prediction jitter (lower is better)

### Expected Performance
Based on neural decoding literature:
- **Position MSE**: 5-10mm (good), <5mm (excellent)
- **R²**: >0.7 (good), >0.85 (excellent)
- **Generalization**: Per-animal test error <20% higher than per-session

## 🌐 Website Features

### Interactive Pose Playback
- Side-by-side 3D visualization (ground truth vs prediction)
- Play/pause controls, frame scrubbing
- Three.js rendering with orbit controls

### Metrics Dashboard
- Real-time metrics display (MSE, MAE, R², RMSE)
- Interactive charts (Recharts)
- Per-session and per-animal comparisons

### Visualizations
- Training loss curves
- Per-keypoint error breakdown
- R² distribution across coordinates
- Neural activity heatmaps
- PCA of LSTM hidden states

## ⚙️ Configuration

All configurations managed via Hydra. Key parameters:

### Model Config

```yaml
model:
  hidden_dim: 256        # LSTM hidden dimension
  num_layers: 2          # Number of LSTM layers
  bidirectional: true    # Use BiLSTM
  rnn_type: "lstm"       # "lstm" or "gru"
  input_mode: "multimodal"  # "multimodal", "neural_only", "pose_only"
  dropout: 0.2
  learning_rate: 1e-3
  lambda_velocity: 0.1   # Weight for velocity loss
```

### Data Config

```yaml
data:
  data_dir: "data/Virtual_Rodent"
  strategy: "per_session"  # or "per_animal"
  neural_history: 750      # 15 seconds at 50Hz
  pose_history: 750
  pose_horizon: 250        # 5 seconds
  batch_size: 32
  num_workers: 4
```

### Trainer Config

```yaml
trainer:
  max_epochs: 100
  accelerator: "auto"      # auto-detect GPU/MPS/CPU
  devices: 1
  gradient_clip_val: 1.0
  patience: 15             # Early stopping patience
```

## 🔬 Visualizations

All visualizations are saved to `outs/visualizations/` for file browser viewing:

### Animation Videos (`videos/`)
- MP4 files of prediction playback
- Side-by-side ground truth vs predicted skeletons
- Generated with matplotlib animations

### Metrics Plots (`plots/`)
- `per_session_mse.png` - MSE comparison across sessions
- `per_session_r2.png` - R² comparison across sessions
- `per_animal_comparison.png` - Aggregate metrics per animal
- `r2_distribution.png` - Distribution of R² across all coordinates

### Neural Visualizations (`neural/`)
- `{session_id}_heatmap.png` - Neural activity heatmap (time × neurons)
- `{session_id}_pca.png` - PCA of LSTM hidden states
- `{session_id}_correlation.png` - Neuron-pose correlation matrix

## 🧪 Example Usage

### Train per-session model
```bash
python scripts/train.py experiment_name=session_multimodal
```

### Train per-animal model
```bash
python scripts/train.py data=per_animal experiment_name=animal_multimodal
```

### Evaluate and generate visualizations
```bash
python scripts/evaluate.py \\
  checkpoint_path=outs/checkpoints/session_multimodal/best.ckpt \\
  experiment_name=session_multimodal
```

### Launch website
```bash
# Terminal 1
python src/website/app.py

# Terminal 2
cd src/website/frontend && npm run dev
```

## 📝 Output Files

### Training Outputs
- `outs/checkpoints/{experiment_name}/` - Model checkpoints
  - `best.ckpt` - Best validation loss checkpoint
  - `last.ckpt` - Last epoch checkpoint
  - `epoch=XX-val_loss=X.XXXX.ckpt` - Top-3 checkpoints

- `outs/results/{experiment_name}/` - Training results
  - `training_results.json` - Metrics, config, checkpoint path

- `logs/{experiment_name}/` - TensorBoard logs
  - View with: `tensorboard --logdir logs/`

### Evaluation Outputs
- `outs/results/{experiment_name}/predictions/` - Predictions
  - `test_predictions.npz` - Compressed numpy arrays

- `outs/results/{experiment_name}/` - Metrics
  - `overall_metrics.json` - Test set metrics
  - `{session_id}_metrics.json` - Per-session metrics
  - `{animal}_summary.json` - Per-animal aggregated metrics

- `outs/visualizations/` - All visualizations
  - Accessible via website and file browser

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'virtual_rodent'**
```bash
# Ensure you're in project root and package is installed
pip install -e .
```

**MPS (Apple Silicon) Issues**
- DataModule automatically disables pin_memory for MPS
- If issues persist, set `data.num_workers=0`

**Out of Memory**
- Reduce `data.batch_size`
- Reduce `data.neural_history` or `data.pose_horizon`
- Set `data.num_workers=0` to use less RAM

**Website Frontend Not Loading**
```bash
# Ensure dependencies are installed
cd src/website/frontend
npm install

# Check that Flask backend is running on port 5000
curl http://localhost:5000/api/health
```

## 🔍 Key Design Decisions

### 1. Temporal Leakage Prevention
- Gap regions (1%) between train/val/test splits
- Margin filtering (respect neural_history and pose_horizon)
- No sample can use future data from different split

### 2. Variable Neuron Count Handling
- Automatic padding to max_neurons across sessions
- LSTM naturally handles effective variable-length input
- Consistent tensor shapes for batching

### 3. Autoregressive Decoding
- Teacher forcing during training (faster, more stable)
- Free-running during inference (realistic)
- Velocity loss encourages smooth predictions

### 4. Chronological Per-Animal Splits
- Tests temporal generalization
- More realistic for neuroscience (neural drift)
- Prevents data leakage across time

## 📚 References

- VirtualRodent Paper: Neural control of 3D pose
- PyTorch Lightning: Training framework
- Hydra: Configuration management
- React + Three.js: 3D visualization
- Flask: REST API backend

## 🎉 Summary

This complete neural decoding system provides:
✅ **Training**: Dual evaluation strategies with configurable models
✅ **Evaluation**: Comprehensive metrics and visualizations
✅ **Website**: Interactive exploration of results
✅ **Visualizations**: Saved to files for easy viewing
✅ **Configuration**: Hydra-based experiment management
✅ **Documentation**: Comprehensive guides and examples

All results are saved to files (JSON, NPZ, PNG, MP4) for viewing in file browser, in addition to being served via the website!
