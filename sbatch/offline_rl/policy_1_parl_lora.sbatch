#!/bin/bash
#SBATCH --job-name="1/4 parl_lora — iql critic + action-expert LoRA distillation"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=rlwrld
#SBATCH --time=24:00:00
#SBATCH --wckey=project-short-name:rd
#SBATCH --output=/rlwrld2/home/junmo_cho/ws/rd-rl/out/policy_1_parl_lora_%j.out
#SBATCH --error=/rlwrld2/home/junmo_cho/ws/rd-rl/out/policy_1_parl_lora_%j.err
# MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/junmo_cho/ sbatch ./sbatch/offline_rl/policy_1_parl_lora.sbatch

cd /rlwrld2/home/junmo_cho/ws/rd-rl/
source configs/kakao_path.sh
set -euo pipefail

DATA=rl-dataset/0825_openarm_f1_inference
export PYTHONPATH="$PWD/third_party/RLDX-1:$PWD"
export NO_ALBUMENTATIONS_UPDATE=1
PY=third_party/RLDX-1/.venv/bin/python

CRITIC=iql-cog-dist128-t07-g0995-q10all-s0-h01/critic_latest.pt

# (4) 액션 relabel: base 후보 M=32 → top-K=10(+로그 액션) → ∇_a Q 상승 10스텝 → argmax
#     → checkpoints/openarm_rim-critic/parl_actions.npy (raw 액션 (T,A))
$PY -u -m rl.relabel_parl \
  --exp openarm_rim --data $DATA --checkpoints checkpoints \
  --features cogfeat.npy --critic $CRITIC \
  --num-samples 32 --num-keep 10 --num-steps 10 --auto-step 0.05 --temp 0 \
  --batch 8 --no-parquet

# (5) action expert LoRA distillation. 손실은 RLDX-1 원본 flow matching (rldx.py:438).
#     타깃이 **고정 라벨**이라 30k x 16 = 58 epoch 은 과적합 가능 — 11개 체크포인트에서 고를 것.
$PY -u -m rl.train_policy --method parl \
  --exp openarm_rim --data $DATA --checkpoints checkpoints \
  --targets parl_actions.npy \
  --steps 30000 --batch 16 --lr 1e-4 --lora-rank 16 --lora-alpha 32 \
  --holdout 0.1 --save-every 3000 --wandb-run parl-lora

echo "[done 1/4] checkpoints/openarm_rim-policy/parl-lora16-lr0.0001-s0/lora_latest.pt"
