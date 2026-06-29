#!/bin/bash -l
# Generic single-A100 job: run any experiment config on the MPCDF Raven cluster.

#SBATCH --job-name=qb-run
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --time=08:00:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

CONFIG="${1:-${CONFIG:-}}"
if [[ -z "${CONFIG}" ]]; then
  echo "ERROR: no config given. Usage: sbatch $0 <path/to/config.yaml>" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 1
fi

echo " >> Job ${SLURM_JOB_ID} on $(hostname)"
echo " >> Config: ${CONFIG}"
echo " >> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

srun python -c "from src import run; run('${CONFIG}')"
