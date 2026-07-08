#!/bin/bash -l

#SBATCH --job-name=qb-lmjudge-4b
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=scripts/logs/%x_%A_%a.out
#SBATCH --error=scripts/logs/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

CONFIGS=(
  "templates/qwen3_4b_experiment_lmjudge_matrix_split/bf16.yaml"
  "templates/qwen3_4b_experiment_lmjudge_matrix_split/awq-w4a16-asym.yaml"
  "templates/qwen3_4b_experiment_lmjudge_matrix_split/int4.yaml"
  "templates/qwen3_4b_experiment_lmjudge_matrix_split/int8.yaml"
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo " >> Array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo " >> Config: ${CONFIG}"
echo " >> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

srun python -c "from src import run; run('${CONFIG}')"
