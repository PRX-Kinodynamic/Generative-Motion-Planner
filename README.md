# GenMoPlan: Generative Motion Planner

A framework for motion planning using generative machine learning methods

## Overview

GenMoPlan is a research framework for training diffusion-based and flow-matching generative models for trajectory generation and motion planning. It supports various models for generating system trajectories from initial conditions.

## Features

- Training diffusion models for trajectory generation
- Flow matching models for conditional trajectory generation
- Support for any control systems along with manifold operations (limited to flow matching)
- Visualization tools for datasets and generated trajectories
- Region of Attraction (RoA) estimation

## Project Structure

```
Generative-Motion-Planner/
├── genMoPlan/         # Main package
│   ├── models/         # Model implementations
│   ├── datasets/       # Dataset implementations
│   └── utils/          # Utility functions
├── scripts/            # Training and visualization scripts
├── config/             # Configuration files (<dataset>.py)
├── data_trajectories/  # Dataset storage
└── experiments/        # Experiment results organized as <dataset>/<method_name>/
```

## Data Structure

The `data_trajectories` directory contains trajectory datasets organized as follows:

```
data_trajectories/
├── <dataset_name>/        # e.g., pendulum_lqr_50k
│   ├── trajectories/      # Essential folder containing individual trajectory files
│   │   └── sequence_<n>.txt  # Individual trajectory files with state data
│   └── roa_labels.txt     # Labels for Region of Attraction estimation (only required for RoA estimation)
```

Each trajectory file (`sequence_<n>.txt`) contains comma-separated state variables (e.g., angle and angular velocity for pendulum systems) with one state per line representing a timestep in the sequence.

## Installation

```bash
# Clone the repository
git clone https://github.com/PRX-Kinodynamic/Generative-Motion-Planner.git
cd Generative-Motion-Planner

# Create conda environment and install packages
conda env create -f environment.yml

# Activate the environment
conda activate genMoPlan

# Install flow_matching dependency
git clone https://github.com/Ewerton-Vieira/flow_matching.git
cd flow_matching
conda env update --name genMoPlan --file environment.yml
pip install -e .
cd ../Generative-Motion-Planner

# Install genMoPlan as an editable package
pip install -e .
```

Note: PyTorch installation may fail. In that case, remove the corresponding line from the requirements.txt file and refer to [PyTorch's website](https://pytorch.org/get-started/locally/) to install it.

## Usage

### Training a Diffusion Model

To train a diffusion model

```bash
python scripts/train_trajectory.py --dataset pendulum_lqr_50k --method diffusion
```

### Training a Flow-Matching Model

To train a flow matching model

```bash
python scripts/train_trajectory.py --dataset pendulum_lqr_50k --method flow_matching
```


### Using Configuration Variations

You can specify predefined variations to modify the base configuration:

```bash
python scripts/train_trajectory.py --dataset pendulum_lqr_50k --method diffusion --variations fewer_steps
```

### Visualizing Models and Datasets

```bash
# Visualize trajectories from a trained model
python scripts/viz_model.py --dataset pendulum_lqr_50k --model_path path/to/model

# Visualize dataset trajectories
python scripts/viz_dataset.py --dataset pendulum_lqr_50k
```

### Estimating Region of Attraction

```bash
# Estimate the region of attraction for a trained model
python scripts/estimate_roa.py --dataset pendulum_lqr_50k --model_path path/to/model
```

## Configuration

Configuration files are located in the `config/` directory with naming convention `<dataset>.py`. Each configuration file defines parameters for:

- Dataset loading and preprocessing
- Method (diffusion, flow_matching) Params
- Model (U-Net, Transformer) architecture
- Training process
- Evaluation metrics

## Experiment Results

Experiment results are organized in the following structure:
```
experiments/
├── <dataset>/          # Dataset used for the experiment
│   ├── <method_name>/  # Method used (diffusion, flow_matching, etc.)
│   │   └── ...         # Experiment outputs and model checkpoints
```

## Requirements

Main dependencies:
- Python 3.9
- PyTorch with CUDA 11.8
- NumPy
- Matplotlib
- tqdm
- einops
- scipy
- [flow_matching](https://github.com/Ewerton-Vieira/flow_matching.git)

See `environment.yml` and `requirements.txt` for complete dependency information.

## TODO

- [ ] Add support for custom dataset readers to allow for different data formats and structures
