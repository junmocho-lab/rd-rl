#!/bin/bash
#SBATCH --export=ALL,MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/junmo_cho/0827-rby1-f1-fuji-s180-rldx
#SBATCH --job-name=0827-rby1-f1-fuji-s180-rldx
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --out=/fsx/rlwrld/junmo_cho/logs/%x-%j.out
#SBATCH --err=/fsx/rlwrld/junmo_cho/logs/%x-%j.err

set -euo pipefail
REPO=/fsx/rlwrld/junmo_cho/ws/rd-rl/third_party/RLDX-1
DATA=/fsx/rlwrld/junmo_cho/ws/dataset/fuji

SECRETS="${RLDX_SECRETS:-$HOME/.rldx_secrets.sh}"
if [[ -f "$SECRETS" ]]; then
    set +x
    source "$SECRETS"
fi
: "${HF_TOKEN:?HF_TOKEN unset — create $SECRETS (see comment above)}"
: "${HUGGINGFACE_HUB_TOKEN:=$HF_TOKEN}"
export HF_TOKEN HUGGINGFACE_HUB_TOKEN
: "${WANDB_API_KEY:?WANDB_API_KEY unset — create $SECRETS, or drop --use-wandb below}"
export WANDB_API_KEY
export HF_HOME=/fsx/rlwrld/junmo_cho/hf_cache
export NO_ALBUMENTATIONS_UPDATE=1
export MODEL_OUTPUT_DIR
NUM_GPUS=$(nvidia-smi -L | wc -l)
export NUM_GPUS
echo "checkpoints -> $MODEL_OUTPUT_DIR"

cd "$REPO"
source .venv/bin/activate

# Single merged dataset: rby1m_rh56f1_teleop_s180_20260827 (teleop, not inference).
# 98 ep / 64058 frames / 60236 effective steps (frames - ep x (action_horizon - 1);
# RLDX-1 drops the last 39 frames of each episode at horizon 40). 30 fps, episode
# length 550-849. Read from meta/info.json on 2026-08-27.
#   epoch = 50000 x 128 / 60236 = 106.2
#
# meta/ has stats.json and relative_stats.json, so lerobot_episode_loader.py's stats
# assert is satisfied. modality.json matches rby1_f1_config.py exactly: the same 6 joint
# groups (torso 4:10 / right_arm 10:17 / right_hand 17:23 / left_arm 23:30 /
# left_hand 30:36 / neck 36:38) and the same 3 cameras.
#
# No --dataset-mix-ratios here: with one --dataset-paths entry there is only one dataset
# spec, so its weight is 1.0 regardless.
torchrun --nproc_per_node="$NUM_GPUS" --standalone \
    rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1-PT-IMG \
    --backbone-path RLWRLD/RLDX-1-VLM \
    --dataset-paths $DATA/rby1m_rh56f1_teleop_s180_20260827 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path rldx/configs/data/rby1_f1_config.py \
    --output-dir "$MODEL_OUTPUT_DIR" \
    --video-length 1 \
    --n-cog-tokens 64 \
    --color-jitter-params brightness 0.6 contrast 0.5 saturation 0.5 hue 0.08 \
    --global-batch-size 128 \
    --learning-rate 1e-4 \
    --max-steps 50000 \
    --dataloader-num-workers 16 \
    --experiment-name "fuji_0827_s180_98ep_rby1_f1_rldx_ptimg_drop05_rtc8tr_bs128_50k_${NUM_GPUS}gpu" \
    --use-wandb --wandb-project RLDX-1 \
    --action-horizon 40 \
    --rtc-training-max-delay 8 \
    --state-dropout-prob 0.5 \
    --num-gpus "$NUM_GPUS" \
    --random-crop-fraction 0.95 \
    --save-only-model \
    --save-steps 10000
