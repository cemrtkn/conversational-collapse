#!/bin/bash -l
#
#SBATCH --chdir=.
#SBATCH --output logs/test/slurm-%x-%j.out
#SBATCH --error logs/test/slurm-%x-%j.out
#SBATCH --job-name interpretability_testing
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
#SBATCH --time=00:10:00

source .venv/bin/activate

source .env

python scripts/interpretability/test.py
