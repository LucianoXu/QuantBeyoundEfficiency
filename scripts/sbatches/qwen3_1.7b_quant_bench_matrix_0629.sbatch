#!/bin/bash -l
# Run the Qwen3-1.7B 0629 quant bench matrix as 4 independent single-A100 jobs
# (one per quantization variant), on the MPCDF Raven cluster.

#SBATCH --job-name=qb-matrix-0629
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

# Datasets must be prefetched (scripts/prefetch_datasets.py); fail loudly instead of
# racing to re-download on a flaky compute-node network (the 0629 single-job run died
# on a transient HF Hub 504 while fetching BBH).
export HF_DATASETS_OFFLINE=1

CONFIGS=(
  "templates/qwen3_1.7b_quant_bench_matrix_0629_split/awq-w4a16-asym.yaml"
  "templates/qwen3_1.7b_quant_bench_matrix_0629_split/bf16.yaml"
  "templates/qwen3_1.7b_quant_bench_matrix_0629_split/int8.yaml"
  "templates/qwen3_1.7b_quant_bench_matrix_0629_split/nf4.yaml"
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo " >> Array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo " >> Config: ${CONFIG}"
echo " >> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

srun python -c "from src import run; run('${CONFIG}')"
