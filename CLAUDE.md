# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VirtualRodent is a neural-to-pose prediction pipeline for neuroscience research. It enables forecasting of 3D body pose trajectories from neural signals recorded in motor cortex and dorsolateral striatum (DLS). The project is inspired by the Virtual Rodent paper and provides PyTorch/Lightning datasets for training neural decoding models.

## Development Commands

### Installation
```bash
# Install package in editable mode with all dependencies
pip install -e ".[dev,notebook,website]"

# Install only specific extras
pip install -e ".[dev]"           # Development tools
pip install -e ".[notebook]"       # Jupyter notebooks
pip install -e ".[website]"        # Flask visualization app
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest src/tests/test_data.py

# Run with coverage
pytest --cov=virtual_rodent --cov-report=html

# Run a specific test class or function
pytest src/tests/test_data.py::TestDataset::test_dataset_creation
```

**Note**: Many tests require the Virtual Rodent dataset to be present in [data/Virtual_Rodent/](data/Virtual_Rodent/). Tests are automatically skipped if data is not available.

### Code Quality
```bash
# Format code (line length: 100)
black src/
isort src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Data Architecture

### HDF5 File Structure
Each session file follows this structure:
- `ephys/spike_counts`: Shape `(T, N)` - Neural spike counts per 20ms bin
- `pose/keypoints`: Shape `(T, 3, 23)` - 3D positions of 23 body keypoints
- `pose/qpos`: Shape `(T, 74)` - Full skeletal model joint positions
- `behavior/motion_mapper`: Shape `(T,)` - Behavior labels

**Key Constants**:
- Sampling rate: 50 Hz (20ms bins)
- Keypoints: 23 body landmarks in 3D (xyz)
- Brain regions: `DLS` (dorsolateral striatum) and `motor_cortex`
- Animals: Named after jazz musicians
  - DLS: art, bud, coltrane
  - motor_cortex: duke, freddie, gerry

### Data Pipeline Layers

The data pipeline is organized in three abstraction layers:

1. **Low-level loading** ([src/virtual_rodent/data/loader.py](src/virtual_rodent/data/loader.py)):
   - `SessionLoader`: Loads individual HDF5 files with optional preloading and caching
   - `DatasetIndex`: Efficiently maps global sample indices to (session, frame) tuples
   - `discover_sessions()`: Scans data directory and builds index of all sessions

2. **Dataset layer** ([src/virtual_rodent/data/dataset.py](src/virtual_rodent/data/dataset.py)):
   - `VirtualRodentDataset`: PyTorch Dataset that handles temporal windows
     - Configurable neural history (past frames) and pose horizon (future frames to predict)
     - Automatic padding to handle variable neuron counts across sessions
     - LRU caching of session loaders
   - `VirtualRodentSequenceDataset`: Variant for sequence-to-sequence models

3. **DataModule layer** ([src/virtual_rodent/data/datamodule.py](src/virtual_rodent/data/datamodule.py)):
   - `VirtualRodentDataModule`: PyTorch Lightning DataModule
   - Three split strategies:
     - `random`: Random split across all samples
     - `session`: Split by session (prevents data leakage)
     - `animal`: Split by animal (tests generalization to new subjects)

### Preprocessing Pipeline

[src/virtual_rodent/data/preprocessing.py](src/virtual_rodent/data/preprocessing.py) provides:

**Neural normalization**:
- `zscore`: Zero-mean, unit variance
- `sqrt`: Square-root transform (variance-stabilizing for Poisson)
- `log`: Log transform
- `minmax`: Min-max scaling

**Pose normalization**:
- `center`: Center relative to reference keypoint (default: spine_lower at index 14)
- `zscore`: Standardize across dataset

**Smoothing**: Gaussian, boxcar, and exponential smoothing for neural signals

**Utilities**: Flatten/unflatten keypoints, compute velocities, create temporal windows

## Important Design Patterns

### Temporal Context Windows
Datasets support flexible temporal configurations:
- `neural_history`: Number of past neural frames to include (e.g., 10 = 200ms of history)
- `pose_history`: Number of past pose frames to include (typically 0)
- `pose_horizon`: Number of future pose frames to predict (e.g., 5 = 100ms ahead)

The dataset automatically computes valid indices that respect these temporal margins.

### Variable Neuron Counts
Different animals have different numbers of recorded neurons. The dataset:
1. Computes `max_neurons` across all sessions
2. Pads neural inputs to this maximum with zeros
3. Returns consistent tensor shapes for batching

This is handled automatically in `VirtualRodentDataset.__getitem__()`.

### Session Caching
To balance memory usage and I/O:
- `SessionLoader` can optionally preload data into memory
- `VirtualRodentDataset` maintains an LRU cache of `SessionLoader` instances (default: 4 sessions)
- Configure via `cache_sessions` parameter

### MPS Backend Compatibility
The DataModule automatically detects MPS (Apple Silicon) backend and disables `pin_memory` as it's not supported. See `_should_pin_memory()` in [src/virtual_rodent/data/datamodule.py](src/virtual_rodent/data/datamodule.py).

## Common Workflows

### Loading a Single Dataset
```python
from virtual_rodent.data import VirtualRodentDataset

dataset = VirtualRodentDataset(
    data_dir="data/Virtual_Rodent",
    brain_regions=["motor_cortex"],
    neural_history=10,        # 200ms of neural history
    pose_horizon=5,           # Predict 100ms into future
    normalize_neural_method="sqrt",
    normalize_pose_method="center",
)

neural, pose = dataset[0]  # Returns tensors
```

### Training with PyTorch Lightning
```python
from virtual_rodent.data import VirtualRodentDataModule

dm = VirtualRodentDataModule(
    data_dir="data/Virtual_Rodent",
    batch_size=64,
    split_strategy="session",  # Prevent data leakage
    neural_history=10,
    pose_horizon=5,
)

dm.setup("fit")
# Access via dm.train_dataloader(), dm.val_dataloader()
```

### Accessing Sample Metadata
```python
sample = dataset.get_sample_with_metadata(idx)
# Returns dict with: neural, pose, behavior, session_id, animal, brain_region, frame_idx
```

## File Organization

- [src/virtual_rodent/data/](src/virtual_rodent/data/): Core data pipeline
- [src/tests/](src/tests/): Unit tests (requires dataset to run)
- [notebooks/](notebooks/): Jupyter notebooks for exploration
- [scripts/](scripts/): Utility scripts
- [data/Virtual_Rodent/](data/Virtual_Rodent/): Expected location for HDF5 dataset files
- [logs/](logs/): Training logs
- [outs/](outs/): Model outputs

## Code Style

- Line length: 100 characters
- Formatter: Black with isort integration
- Type hints encouraged (checked with mypy)
- Docstrings: NumPy/Google style with type annotations in Args/Returns sections
