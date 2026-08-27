#!/bin/bash
#SBATCH --job-name=dexjoco-hammer-nail-bc
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/rlwrld2/home/junmo_cho/ws/rd-rl/out/dexjoco-bc_%j.out
#SBATCH --error=/rlwrld2/home/junmo_cho/ws/rd-rl/out/dexjoco-bc_%j.err

# RLDX-1 BC finetune on DexJoCo hammer_nail (rand_obj, 100 teleop demos).
#
# Why hammer_nail: shortest horizon of the 11 tasks (215.7 frames = 7.2 s mean,
# 149 min / 522 max) and the DexJoCo paper's Table 2 puts GR00T N1.5 — the closest
# architectural relative of RLDX-1 — at 67.3% on it under rand-obj, i.e. a clear
# ~30 pt headroom for the critic/test-time-guidance experiment. click_mouse (85.3%)
# and pick_bucket (72.0%) leave less room; pinch_tongs (12.7%) and fold_glasses
# (27.3%) are too weak for a BC baseline to improve on; the bimanual tasks are both
# longer-horizon and near 0.
#
# Dataset is built by:
#   /workspace/junmo_cho/dexjoco/venv/bin/python sim/dexjoco/convert_raw_to_rldx.py \
#       --input /workspace/junmo_cho/dexjoco/raw/dexjoco_raw_datasets/hammer_nail \
#       --output $DATA --task hammer_nail
#
# Nothing lands on /rlwrld2 (69 GB free, 100% used): dataset on /workspace,
# checkpoints on /rlwrld-unified-checkpoints (both effectively unbounded).
#
#   sbatch sbatch/dexjoco/bc_hammer_nail.sbatch

set -euo pipefail

REPO=/rlwrld2/home/junmo_cho/ws/rd-rl/third_party/RLDX-1
DATA=/workspace/junmo_cho/dexjoco/lerobot/hammer_nail_rand_obj
OUTPUT_DIR=/rlwrld-unified-checkpoints/junmo_cho/dexjoco/hammer_nail_bc_ptimg

# ---------------------------------------------------------------- secrets
# RLWRLD/RLDX-1-PT-IMG and RLDX-1-VLM are private, and this node has no cached
# blobs for them, so HF_TOKEN is mandatory. Keep it out of the file:
#   umask 077; cat > ~/.rldx_secrets.sh <<'EOF'
#   export HF_TOKEN=hf_...
#   export WANDB_API_KEY=...
#   EOF
SECRETS="${RLDX_SECRETS:-$HOME/.rldx_secrets.sh}"
if [[ -f "$SECRETS" ]]; then
    source "$SECRETS"
fi
: "${HF_TOKEN:?HF_TOKEN unset — create $SECRETS (see comment above)}"
: "${HUGGINGFACE_HUB_TOKEN:=$HF_TOKEN}"
export HF_TOKEN HUGGINGFACE_HUB_TOKEN

# ---------------------------------------------------------------- environment
# The 13 GB base checkpoint must not land on /rlwrld2 either.
export HF_HOME=/workspace/junmo_cho/hf_cache
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

mkdir -p "$OUTPUT_DIR" "$HF_HOME"

NUM_GPUS=$(nvidia-smi -L | wc -l)
CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)}
NUM_WORKERS=$(( 16 < (CPUS / NUM_GPUS) ? 16 : (CPUS / NUM_GPUS) ))

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=$NUM_GPUS workers/rank=$NUM_WORKERS"
echo "data        -> $DATA"
echo "checkpoints -> $OUTPUT_DIR"

cd "$REPO"
source .venv/bin/activate
python -c "
import json, rldx
info = json.load(open('$DATA/meta/info.json'))
mod  = json.load(open('$DATA/meta/modality.json'))
assert info['codebase_version'] == 'v2.1', info['codebase_version']
assert set(mod['video']) == {'camera_front', 'camera_wrist'}, mod['video']
print('rldx', rldx.__version__, '|', info['total_episodes'], 'ep /', info['total_frames'], 'frames')
"

# --max-steps 10000 at global batch 128 over ~18.5k effective steps
# (100 ep x (215.7 - 31); RLDX-1 drops the last action_horizon-1 frames of every
# episode) = ~69 epochs, the same epoch budget the cjl_0819_v9 rby1 run used.
# --save-steps 2500 keeps 4 intermediate checkpoints (~13 GB each) so the BC
# success rate can be measured as a function of training length and the best one
# picked as the frozen base policy for the critic stage.
#
# --action-horizon 32 must equal len(action delta_indices) in
# dexjoco_panda_allegro_config.py or assembly.py raises immediately. 32 / 30 fps
# = 1.07 s, and the rollout client executes ~16 of those per replan.
torchrun --nproc_per_node="$NUM_GPUS" --standalone \
    rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1-PT-IMG \
    --backbone-path RLWRLD/RLDX-1-VLM \
    --dataset-paths "$DATA" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path rldx/configs/data/dexjoco_panda_allegro_config.py \
    --output-dir "$OUTPUT_DIR" \
    --video-length 1 \
    --n-cog-tokens 64 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --global-batch-size 128 \
    --learning-rate 1e-4 \
    --max-steps 10000 \
    --dataloader-num-workers "$NUM_WORKERS" \
    --experiment-name "dexjoco_hammer_nail_randobj_100ep_ptimg_h32_rtc8tr_drop03_bs128_10k_${NUM_GPUS}gpu" \
    --use-wandb --wandb-project RLDX-1 \
    --action-horizon 32 \
    --rtc-training-max-delay 8 \
    --state-dropout-prob 0.3 \
    --num-gpus "$NUM_GPUS" \
    --random-crop-fraction 0.95 \
    --save-only-model \
    --save-steps 2500

echo "=== $(date -Is) done, checkpoints in $OUTPUT_DIR"
