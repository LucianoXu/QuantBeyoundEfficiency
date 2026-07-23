#!/bin/bash -l
# Run the Qwen3-1.7B quantization utility matrix as 3 independent single-A100 jobs, on the MPCDF Raven cluster.

#SBATCH --job-name=qwen3-1.7b-util
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=scripts/logs/%x_%A_%a.out
#SBATCH --error=scripts/logs/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

# For dataset, fail loudly if a dataset is somehow still missing, rather than racing to regenerate it.
export HF_DATASETS_OFFLINE=1


CONFIGS=(
  "templates/qwen3_1.7b_quant_utility_split/bf16.yaml"
  "templates/qwen3_1.7b_quant_utility_split/int8.yaml"
  "templates/qwen3_1.7b_quant_utility_split/nf4.yaml"
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo " >> Array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo " >> Config: ${CONFIG}"
echo " >> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

srun python -c "from src import run; run('${CONFIG}')"
