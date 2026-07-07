#!/bin/bash -l

#SBATCH --job-name=adv-diff-4gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=480000
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/adv-4gpu-%j.log
#SBATCH --error=scripts/logs/adv-4gpu-%j.log

cd "$SLURM_SUBMIT_DIR"
module load gcc/13 2>/dev/null || true  # quanto int4 kernels need C++20
export CC=$(command -v gcc) CXX=$(command -v g++)
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======== GPUs on this node ========"
nvidia-smi -L
python -c "import torch; print('torch sees', torch.cuda.device_count(), 'GPUs')"

echo "======== [gate] multi-GPU sharding smoke ========"
rm -rf trials/adv_diff_smoke
if ! python -c "from src import run; run('templates/adv_diff_smoke.yaml')"; then
    echo "SMOKE FAILED -- aborting before the real run (node released)."
    exit 1
fi

echo "======== [real] differential GA, 4-GPU sharded ========"
python -c "from src import run; run('templates/qwen3_4b_adv_diff.yaml')"

echo "======== DONE ========"
