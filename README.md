# El fantastico

**Autonomous Vehicle Trajectory Prediction**

*Proyecto Integrador · Maestría en Inteligencia Artificial Aplicada · Tecnológico de Monterrey*

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-EE4C2C?logo=pytorch&logoColor=white)
![ROS2](https://img.shields.io/badge/ROS2-MCAP-22314E?logo=ros&logoColor=white)
![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?logo=uv&logoColor=white)

## Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run a script
uv run python training/training.py

# Run Jupyter notebooks
uv run jupyter notebook

# Add a new dependency
uv add <package>
```

## Project structure
```text
├── README.md                  # Project documentation
├── crips-dm                   # Methodology analysis
├── data                       # Folder for processed data, datasets, or intermediate files
├── dataset
│   ├── builder.py             # Script for building and handling datasets
│   ├── output-examples        # Example output samples from the dataset (e.g., processed clips)
│   │   ├── clip_sample.pt     # Tensor sample
│   │   └── sample.npz         # Numpy sample
│   └── outputs                # Generated outputs from dataset processing or model runs
├── extras
│   └── topics_list.txt        # Additional files, such as lists of ROS topics or other metadata
├── inference
│   └── inference.ipynb        # Notebooks or scripts for running inference using trained models
├── model-checkpoint           # Directory to store model checkpoints during or after training
├── notebooks
│   ├── 1_clip_analysis.ipynb     # Jupyter notebooks for exploration and analysis
│   └── 2_inputs_transformation.ipynb  
├── pyproject.toml             # Project dependencies and configuration
├── raw-data
│   ├── image                  # Raw image data and scripts for handling them
│   │   └── images.py
│   ├── orientation            # Raw orientation sensor data and utilities
│   │   └── orientation.py
│   └── pose                   # Raw pose/GPS data and related scripts
│       └── pose.py
├── training
│   ├── __init__.py
│   ├── parameter_trainable.py # Utilities for model parameter handling, e.g., LoRA and freezing layers
│   └── training.py            # Model training logic
└── uv.lock                    # Lock file for reproducible dependencies
```
