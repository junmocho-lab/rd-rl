#!/bin/bash
#SBATCH --job-name="offline_critic_0 openarm_rim (0815_openarm_rh56f1_inference, steps=3000)"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --output=/rlwrld2/home/junmo_cho/ws/rd-rl/out/critic_%j.out
#SBATCH --error=/rlwrld2/home/junmo_cho/ws/rd-rl/out/critic_%j.err

cd /rlwrld2/home/junmo_cho/ws/rd-rl/
source configs/kakao_path.sh

PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" NO_ALBUMENTATIONS_UPDATE=1 \
third_party/RLDX-1/.venv/bin/python -u -m rl.extract_cogfeat \
  --exp openarm_rim \
  --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints \
  --batch 64 --resume

PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" NO_ALBUMENTATIONS_UPDATE=1 \
third_party/RLDX-1/.venv/bin/python -u -m rl.offline_iql \
  --exp openarm_rim \
  --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints \
  --features cogfeat.npy \
  --discount 0.995 --bins 128 --expectile 0.7 \
  --steps 30000 --holdout 0.1 --eval-every 5000 --video-eps 30
# --train-eps success 성공 에피소드만 학습하고 싶다면 추가

PYTHONPATH="$L_PYTHONPATH" $L_PY -u -m rl.offline_critic_0 --exp openarm_rim --data $L_DS/0815_openarm_rh56f1_inference --checkpoints $L_CKPT/sbatch