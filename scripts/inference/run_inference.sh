#!/bin/bash -l
#
#SBATCH --output logs/slurm-%x-%j.out
#SBATCH --error logs/slurm-%x-%j.out
#SBATCH -D ./
#SBATCH --job-name run_inference
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

module load apptainer

source .env

INFERENCE_IMAGE="container_images/inference.sif"
HF_CACHE_PATH="/ptmp/$USER/huggingface"

apptainer exec --nv \
    -B .:"$HOME" \
    -B "$HF_CACHE_PATH:/huggingface" \
    --env HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN}" \
    --env HF_HUB_CACHE="/huggingface/hub" \
    "${INFERENCE_IMAGE}" \
    bash -c 'pip install --user -e "$HOME" --no-deps && python scripts/inference/run_inference.py'