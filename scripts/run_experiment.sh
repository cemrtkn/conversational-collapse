#!/bin/bash -l
#
#SBATCH --chdir=.
#SBATCH --output logs/slurm-%x-%j.out
#SBATCH --error logs/slurm-%x-%j.out
#SBATCH --job-name collapse_experiment
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
# Wall clock limit (max is 24 hours):
#SBATCH --time=00:30:00

# defaults to configs/test_config.yaml
CONFIG_FILE=${1:-configs/test_config.yaml}


source .venv/bin/activate

source .env

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "from pyarrow import __version__; print('PyArrow version:', __version__)"

python src/main.py "$CONFIG_FILE"
