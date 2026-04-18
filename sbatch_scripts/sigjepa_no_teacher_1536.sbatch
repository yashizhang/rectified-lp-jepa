#!/bin/bash
#SBATCH --job-name=sigjepa-no-teacher-1536
#SBATCH --account=AI4D-CORE-132
#SBATCH --partition=TrixieLong
#SBATCH --gres=gpu:3
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --output=/home/ext.yashi.zhang/work/rectified-lp-jepa/slurm_logs/%x-%j.out
#SBATCH --error=/home/ext.yashi.zhang/work/rectified-lp-jepa/slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/ext.yashi.zhang/work/rectified-lp-jepa"
cd "$REPO_ROOT"

source ./.venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_VISIBLE_DEVICES=0,1,2

echo "Host: $(hostname)"
echo "Started: $(date)"

srun python main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100 \
  --config-name=sigjepa_no_teacher_1536.yaml \
  'devices=[0,1,2]' \
  optimizer.batch_size=64 \
  data.preload=true \
  strategy=ddp_find_unused_parameters_true \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

echo "Finished: $(date)"
