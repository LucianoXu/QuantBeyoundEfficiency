#!/bin/bash -l
# Run the Qwen3-1.7B experiment_1 bench matrix as 4 independent single-A100 jobs
# (one per quantization variant: bf16 / int8 / int4 / awq-w4a16-asym) on Raven.

#SBATCH --job-name=qb-exp1
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --time=24:00:00
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

# HumanEval's code_eval need this.
export HF_ALLOW_CODE_EVAL=1

set -a; [ -f .env ] && source .env; set +a

CONFIGS=(
  "templates/qwen_3_experiment1_split/bf16.yaml"
  "templates/qwen_3_experiment1_split/int8.yaml"
  "templates/qwen_3_experiment1_split/int4.yaml"
  "templates/qwen_3_experiment1_split/awq-w4a16-asym.yaml"
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo " >> Array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo " >> Config: ${CONFIG}"
echo " >> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

srun python -c "from src import run; run('${CONFIG}')"
