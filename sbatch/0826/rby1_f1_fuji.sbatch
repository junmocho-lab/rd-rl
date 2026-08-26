#!/bin/bash
#SBATCH --export=ALL,MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/junmo_cho/0826-rby1-f1-fuji-rldx
#SBATCH --job-name=0826-rby1-f1-fuji-rldx
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
DATA=/fsx/rlwrld/junmo_cho/ws/dataset/fuji/fuji_0810_0824_340x192_success

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

# Subset selection: ep >= 5 only. The 8 subsets with 1-4 successful episodes were
# dropped (23 ep / 21067 eff steps, 8.5% of the success set) — under equal-probability
# sampling a 1-episode subset would be drawn as often as a 30-episode one and simply be
# memorised. The retry_* subsets were removed from the dataset dir separately.
#
# --dataset-mix-ratios (frame-proportional sampling) — computed, left commented out.
# RLDX-1 builds one spec per --dataset-paths entry, so without this all 21 subsets are
# sampled with equal probability (1/21 each) regardless of size: the 5-episode subsets
# would still carry the same weight as the two 30-episode ones.
# Values = effective steps = frames - ep x (action_horizon - 1); RLDX-1 drops the last
# 39 frames of each episode at horizon 40. Read from meta/info.json on 2026-08-26.
# All 21 subsets are 30 fps and have meta/stats.json.
#   rby1m_rh56f1_inference_20260810_113635     ep= 15   14064  ( 6.2%)
#   rby1m_rh56f1_inference_20260810_135951     ep= 16   15260  ( 6.7%)
#   rby1m_rh56f1_inference_20260810_150626     ep=  5    4629  ( 2.0%)
#   rby1m_rh56f1_inference_20260810_151515     ep=  5    4715  ( 2.1%)
#   rby1m_rh56f1_inference_20260810_153125     ep=  5    4698  ( 2.1%)
#   rby1m_rh56f1_inference_20260812_145925     ep= 30   28924  (12.7%)
#   rby1m_rh56f1_inference_20260812_163525     ep=  7    6854  ( 3.0%)
#   rby1m_rh56f1_inference_20260812_170242     ep=  5    5142  ( 2.3%)
#   rby1m_rh56f1_inference_20260812_221036     ep=  7    6643  ( 2.9%)
#   rby1m_rh56f1_inference_20260813_104055     ep=  5    4812  ( 2.1%)
#   rby1m_rh56f1_inference_20260813_131439     ep= 20   18834  ( 8.3%)
#   rby1m_rh56f1_inference_20260813_172132     ep= 30   31686  (13.9%)
#   rby1m_rh56f1_inference_20260813_214147     ep= 10   11572  ( 5.1%)
#   rby1m_rh56f1_inference_20260813_234038     ep=  9   10036  ( 4.4%)
#   rby1m_rh56f1_inference_20260814_091242     ep= 20   20293  ( 8.9%)
#   rby1m_rh56f1_inference_20260814_130546     ep= 10   10638  ( 4.7%)
#   rby1m_rh56f1_inference_20260824_083132     ep=  7    6510  ( 2.9%)
#   rby1m_rh56f1_inference_20260824_093258     ep=  8    6638  ( 2.9%)
#   rby1m_rh56f1_inference_20260824_101111     ep=  5    4101  ( 1.8%)
#   rby1m_rh56f1_inference_20260824_113025     ep=  6    5253  ( 2.3%)
#   rby1m_rh56f1_inference_20260824_114321     ep=  7    6270  ( 2.8%)
#   total  21 subsets / 232 ep / 236620 frames / 227572 effective steps
#   epoch = 60000 x 128 / 227572 = 33.7
#   --dataset-mix-ratios 14064 15260 4629 4715 4698 28924 6854 5142 6643 4812 18834 31686 11572 10036 20293 10638 6510 6638 4101 5253 6270 \
torchrun --nproc_per_node="$NUM_GPUS" --standalone \
    rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1-PT-IMG \
    --backbone-path RLWRLD/RLDX-1-VLM \
    --dataset-paths $DATA/rby1m_rh56f1_inference_20260810_113635 \
                    $DATA/rby1m_rh56f1_inference_20260810_135951 \
                    $DATA/rby1m_rh56f1_inference_20260810_150626 \
                    $DATA/rby1m_rh56f1_inference_20260810_151515 \
                    $DATA/rby1m_rh56f1_inference_20260810_153125 \
                    $DATA/rby1m_rh56f1_inference_20260812_145925 \
                    $DATA/rby1m_rh56f1_inference_20260812_163525 \
                    $DATA/rby1m_rh56f1_inference_20260812_170242 \
                    $DATA/rby1m_rh56f1_inference_20260812_221036 \
                    $DATA/rby1m_rh56f1_inference_20260813_104055 \
                    $DATA/rby1m_rh56f1_inference_20260813_131439 \
                    $DATA/rby1m_rh56f1_inference_20260813_172132 \
                    $DATA/rby1m_rh56f1_inference_20260813_214147 \
                    $DATA/rby1m_rh56f1_inference_20260813_234038 \
                    $DATA/rby1m_rh56f1_inference_20260814_091242 \
                    $DATA/rby1m_rh56f1_inference_20260814_130546 \
                    $DATA/rby1m_rh56f1_inference_20260824_083132 \
                    $DATA/rby1m_rh56f1_inference_20260824_093258 \
                    $DATA/rby1m_rh56f1_inference_20260824_101111 \
                    $DATA/rby1m_rh56f1_inference_20260824_113025 \
                    $DATA/rby1m_rh56f1_inference_20260824_114321 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path rldx/configs/data/rby1_f1_config.py \
    --output-dir "$MODEL_OUTPUT_DIR" \
    --video-length 1 \
    --n-cog-tokens 64 \
    --color-jitter-params brightness 0.6 contrast 0.5 saturation 0.5 hue 0.08 \
    --global-batch-size 128 \
    --learning-rate 1e-4 \
    --max-steps 60000 \
    --dataloader-num-workers 16 \
    --experiment-name "fuji_0810_0824_min5ep232ep_rby1_f1_rldx_ptimg_drop05_rtc8tr_bs128_60k_${NUM_GPUS}gpu" \
    --use-wandb --wandb-project RLDX-1 \
    --action-horizon 40 \
    --rtc-training-max-delay 8 \
    --state-dropout-prob 0.5 \
    --num-gpus "$NUM_GPUS" \
    --random-crop-fraction 0.95 \
    --save-only-model \
    --save-steps 10000
