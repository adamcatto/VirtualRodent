# VirtualRodent

Neural-to-pose prediction pipeline inspired by the Virtual Rodent paper. This project enables forecasting of 3D body pose trajectories from neural signals recorded in motor cortex and dorsolateral striatum (DLS).

## Overview

This project provides:
- **Data pipelines** for loading and preprocessing neural-pose paired data
- **PyTorch/Lightning datasets** for training neural decoding models
- **Interactive website** for visualizing neural signals and pose trajectories
- **Analysis tools** for exploring the relationship between neural activity and behavior

## Project Structure

```
VirtualRodent/
├── data/
│   └── Virtual_Rodent/          # Raw data from Virtual Rodent dataset
│       ├── DLS/                 # Dorsolateral striatum recordings
│       └── motor_cortex/        # Motor cortex recordings
├── docs/
│   └── ai/                      # AI-generated documentation
├── logs/                        # Training logs
├── notebooks/                   # Jupyter notebooks for analysis
├── outs/                        # Model outputs
├── scripts/                     # Utility scripts
└── src/
    ├── virtual_rodent/          # Main Python package
    │   └── data/                # Data loading and preprocessing
    ├── tests/                   # Unit tests
    └── website/                 # Interactive visualization app
```

## Installation

```bash
# Clone the repository
git clone https://github.com/adamcatto/VirtualRodent.git
cd VirtualRodent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e ".[dev,notebook,website]"
```

## Quick Start

```python
from virtual_rodent.data import VirtualRodentDataset

# Load the dataset
dataset = VirtualRodentDataset(
    data_dir="data/Virtual_Rodent",
    brain_region="motor_cortex",
    animal="duke"
)

# Get a sample
neural_signal, pose = dataset[0]
```

## Data

The dataset contains paired recordings of:
- **Neural signals**: Multi-unit activity from motor cortex and DLS
- **3D Pose**: Body keypoint positions tracked over time

Animals are named after jazz musicians: art, bud, coltrane (DLS) and duke, freddie, gerry (motor cortex).

## License

MIT License
