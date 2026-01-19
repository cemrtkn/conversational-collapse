#!/bin/bash -l
#
#SBATCH --output logs/slurm-%x-%j.out
#SBATCH --error logs/slurm-%x-%j.out
#SBATCH -D ./
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

DEBUG_FLAG=$1
# defaults to configs/test_config.yaml
CONFIG_FILE=${2:-configs/test_config.yaml}

# Build debug argument
if [ "$DEBUG_FLAG" = "debug" ]; then
    DEBUG_ARG="--debug"
else
    DEBUG_ARG=""
fi

module load apptainer

source .env

INFERENCE_IMAGE="container_images/inference.sif"

apptainer exec --nv \
    -B .:"$HOME" \
    -B ./data:/data \
    -B "$HF_CACHE_PATH:/huggingface" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env HF_HUB_CACHE="/huggingface/hub" \
    "${INFERENCE_IMAGE}" \
    bash -c 'pip install --user -e "$HOME" --no-deps && python src/main.py '"$CONFIG_FILE"' '"$DEBUG_ARG"
