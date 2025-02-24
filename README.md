# mg_diffuse


## Installation

```bash
# Create env and install conda packages
conda env create -f environment.yml

# Activate env
conda activate mg_diffuse

# Install mg_diffuse as a package
pip install -e .
```

## Usage

Running the base config:
```bash
python scripts/train_diffusion.py --config config/pendulum_lqr_50k.py --dataset pendulum_lqr_50k
```

Running a variation of the base config:
```bash
python scripts/train_diffusion.py --config config/pendulum_lqr_50k.py --dataset pendulum_lqr_50k --variation <variation_name>
```

