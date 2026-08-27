#!/bin/bash
#SBATCH --job-name="iql_fuji_discount_0999_bins_128_expectile_09_steps_100K_holdout_02_eval_every_5000"
#SBATCH --gres=gpu:1               # 노드당 GPU 장수
#SBATCH --nodes=1                  # 노드 수
#SBATCH --partition=rlwrld
#SBATCH --time=24:00:00            # 최대 48:00:00 (2 days)
#SBATCH --wckey=project-short-name:rd
#SBATCH --output=/rlwrld2/home/junmo_cho/ws/rd-rl/out/critic_%j.out
#SBATCH --error=/rlwrld2/home/junmo_cho/ws/rd-rl/out/critic_%j.err
# MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/junmo_cho/ 이거 붙여

cd /rlwrld2/home/junmo_cho/ws/rd-rl/
source configs/kakao_path.sh
export WANDB_API_KEY=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
# PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" NO_ALBUMENTATIONS_UPDATE=1 third_party/RLDX-1/.venv/bin/python -u -m rl.offline_iql \
#   --exp fuji --data rl-dataset/fuji-rl-dataset --checkpoints checkpoints \
#   --discount 0.999 --bins 128 --expectile 0.9 \
#   --steps 100000 --holdout 0.2 --eval-every 5000

PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" NO_ALBUMENTATIONS_UPDATE=1 \
third_party/RLDX-1/.venv/bin/python -u -m rl.offline_iql \
  --exp fuji --data rl-dataset/fuji-rl-dataset --checkpoints checkpoints \
  --features cogfeat.npy \
  --discount 0.999 --bins 128 --expectile 0.7 \
  --steps 100000 --holdout 0.2 --eval-every 5000
