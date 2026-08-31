#!/bin/bash
#SBATCH --export=ALL,MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/junmo_cho/0828-openarm-duck-rldx
#SBATCH --job-name=0828-openarm-duck-rldx
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --out=/fsx/rlwrld/junmo_cho/logs/%x-%j.out
#SBATCH --err=/fsx/rlwrld/junmo_cho/logs/%x-%j.err

# Slurm port of the MLXP k8s Job junmo-cho-0818-0821-openarm-rh56f1-rldx-ptimg
# (sbatch/openarm_duck/junmo-cho-openarm-0818-0821-rldx-ptimg.yaml), with the dataset
# swapped from the 14-subset openarm teleop mix to the single duckdoll_20260828 set.
# Structure follows sbatch/0827/rby1_f1_fuji_s180.sbatch; every openarm-specific flag
# below is carried over from the yaml unchanged.

set -euo pipefail
REPO=/fsx/rlwrld/junmo_cho/ws/rd-rl/third_party/RLDX-1
DATA=/fsx/rlwrld/junmo_cho/ws/dataset/openarm_duck/duckdoll_20260828

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

# Single dataset: duckdoll_20260828_320x192fit.
# 100 ep / 17732 frames / 16232 effective steps (frames - ep x (action_horizon - 1);
# RLDX-1 drops the last 15 frames of each episode at horizon 16). 20 fps, episode
# length 143-234. Read from meta/info.json on 2026-08-28.
#   epoch = 30000 x 128 / 16232 = 236.6
#
# meta/ has stats.json, which is what lerobot_episode_loader.py asserts on.
# relative_stats.json is NOT needed here: openarm_inspire_config.py uses
# ActionRepresentation.DELTA for the neck/arm groups, and only RELATIVE groups read it
# (stats.py:253, state_action_processor.py:365).
#
# modality.json matches openarm_inspire_config.py: the same 5 joint groups
# (neck 0:2 / left_arm 2:9 / left_hand 9:15 / right_arm 15:22 / right_hand 22:28) and
# the same 2 ego cameras — note openarm is ego-stereo only, no wrist camera (the rby1_f1
# configs carry a third wrist_left view).
#
# No --dataset-mix-ratios here: with one --dataset-paths entry there is only one dataset
# spec, so its weight is 1.0 regardless. (The yaml needed ratios for its 14 subsets.)
#
# --action-horizon 16 must match openarm_inspire_config.py's action
# delta_indices=list(range(16)), or assembly.py's
# _validate_action_horizon_matches_modality raises ValueError immediately. This dataset
# is 20 fps, so 16 steps = 0.8 s. (The rby1_f1 fuji runs are 30 fps / 40 steps = 1.33 s.)
#
# --embodiment-tag GENERAL_EMBODIMENT (projector id 0, the warm pretrained slot) is what
# openarm_inspire_config.py registers under, so it must not be changed to NEW_EMBODIMENT
# the way the rby1_f1 runs did.
torchrun --nproc_per_node="$NUM_GPUS" --standalone \
    rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1-PT-IMG \
    --backbone-path RLWRLD/RLDX-1-VLM \
    --dataset-paths $DATA/duckdoll_20260828_320x192fit \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path rldx/configs/data/openarm_inspire_config.py \
    --output-dir "$MODEL_OUTPUT_DIR" \
    --video-length 1 \
    --n-cog-tokens 64 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --global-batch-size 128 \
    --learning-rate 1e-4 \
    --max-steps 50000 \
    --dataloader-num-workers 16 \
    --experiment-name "openarm_duck_0828_100ep_inspire_egostereo_ptimg_drop03_rtc12tr_bs128_50k_${NUM_GPUS}gpu" \
    --use-wandb --wandb-project RLDX-1 \
    --action-horizon 16 \
    --rtc-training-max-delay 5 \
    --state-dropout-prob 0.3 \
    --num-gpus "$NUM_GPUS" \
    --random-crop-fraction 0.95 \
    --save-only-model \
    --save-steps 10000
