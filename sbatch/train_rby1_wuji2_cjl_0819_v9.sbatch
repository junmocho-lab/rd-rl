#!/bin/bash
#SBATCH --job-name=rby1-wuji2-cjl-v9-rldx-ptimg
#SBATCH --partition=h100-260818-261117
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8                 # p5.48xlarge = 8x H100. NUM_GPUS is derived from this.
# No --cpus-per-task / --mem: this cluster rejects both and applies the partition
# defaults DefCpuPerGPU=24 / DefMemPerGPU=245000M, i.e. 8 GPUs = 192 CPU + ~1.9TiB.
#SBATCH --time=24:00:00              # the openarm 8-GPU run does 30k steps in ~6h50m at
                                     # action_horizon 16; horizon 40 here is heavier. max 48:00:00
#SBATCH --out=/fsx/rlwrld/junmo_cho/logs/%x-%j.out
#SBATCH --err=/fsx/rlwrld/junmo_cho/logs/%x-%j.err
# The submit plugin requires MODEL_OUTPUT_DIR in the submitted environment, under
# /fsx/rlwrld-unified-checkpoints/$USER/. --export carries it there. ($USER does not
# expand in #SBATCH lines, hence the literal junmo_cho.)
#SBATCH --export=ALL,MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/junmo_cho/0819-rby1-wuji2-cjl-v9-rldx-ptimg

# RLDX-1 finetune for rby1 + wujihand2 on cjl_0819_v9 (191 ep / 62804 frames usable).
# Slurm port of the MLXP k8s Job junmo-cho-0814-rby1-wuji2-cjl-v9-rldx-ptimg, with the
# dataset swapped from cjl_0814_full_v9 (2 subsets) to cjl_0819_v9 (10 subsets).
#
#   sbatch -p h100-260818-261117 /fsx/rlwrld/junmo_cho/ws/sbatch/train_rby1_wuji2_cjl_0819_v9.sbatch
#
# The -p on the command line is required, not optional: /etc/profile.d exports
# SBATCH_PARTITION=rlwrld-gpu (a partition that does not exist on this cluster) and
# Slurm gives env vars precedence over #SBATCH directives, so a bare
# `sbatch <this file>` is rejected. A CLI flag outranks the env var.

set -euo pipefail

# Requires the junmo/cjl-v6 branch checked out here: both the rby1 wujihand2 modality
# config and --save-only-model (train_config.py, training/training_config.py,
# assembly.py, experiment.py) live only on that branch. main has neither, and the
# startup assert below fails fast if the checkout drifts back.
REPO=/fsx/rlwrld/junmo_cho/ws/RLDX-1

DATA=/fsx/rlwrld/junmo_cho/ws/dataset/cjl_0819_v9/CJL
: "${MODEL_OUTPUT_DIR:=/fsx/rlwrld-unified-checkpoints/$USER/0819-rby1-wuji2-cjl-v9-rldx-ptimg}"
FALLBACK_OUTPUT_DIR=/fsx/rlwrld/junmo_cho/checkpoints/0819-rby1-wuji2-cjl-v9-rldx-ptimg

# ---------------------------------------------------------------- secrets
# HF_TOKEN / HUGGINGFACE_HUB_TOKEN (RLWRLD/RLDX-1-* are private) + WANDB_API_KEY.
# Kept out of this file so it stays safe to commit and never lands in the logs.
#   umask 077; cat > ~/.rldx_secrets.sh <<'EOF'
#   export HF_TOKEN=hf_...
#   export HUGGINGFACE_HUB_TOKEN=hf_...
#   export WANDB_API_KEY=wandb_v1_...
#   EOF
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

# ---------------------------------------------------------------- environment
export HF_HOME=/fsx/rlwrld/junmo_cho/hf_cache
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

# Node-local NVMe for compile/codec scratch — keeps this off the shared /fsx mount.
SCRATCH=/scratch/$USER/$SLURM_JOB_NAME-$SLURM_JOB_ID
if mkdir -p "$SCRATCH" 2>/dev/null; then
    export TMPDIR="$SCRATCH"
    export TRITON_CACHE_DIR="$SCRATCH/triton"
    export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor"
else
    echo "[warn] /scratch unavailable, falling back to default TMPDIR"
    SCRATCH=""
fi

# /fsx/rlwrld-unified-checkpoints/ is root-owned and has no junmo_cho/ subdir (job 14
# hit this too). Fall back rather than lose the run to a mkdir.
if ! mkdir -p "$MODEL_OUTPUT_DIR" 2>/dev/null; then
    echo "[warn] cannot create $MODEL_OUTPUT_DIR — ask an admin for"
    echo "[warn] /fsx/rlwrld-unified-checkpoints/$USER. Falling back to $FALLBACK_OUTPUT_DIR"
    MODEL_OUTPUT_DIR="$FALLBACK_OUTPUT_DIR"
    mkdir -p "$MODEL_OUTPUT_DIR"
fi
export MODEL_OUTPUT_DIR
echo "checkpoints -> $MODEL_OUTPUT_DIR"

NUM_GPUS=$(nvidia-smi -L | wc -l)          # follows --gres, so the two can't drift
export NUM_GPUS
# Allocated CPUs / NUM_GPUS ranks, capped at the MLXP job's 32 per rank.
CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)}                       # 24 per GPU (DefCpuPerGPU)
NUM_WORKERS=$(( 32 < (CPUS / NUM_GPUS) ? 32 : (CPUS / NUM_GPUS) ))
export OMP_NUM_THREADS=4

echo "=== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID gpus=$NUM_GPUS workers/rank=$NUM_WORKERS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
command -v ffmpeg >/dev/null || echo "[warn] ffmpeg not on PATH"

cd "$REPO"
source .venv/bin/activate
echo "repo commit: $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
# Fail in seconds, not hours, if this checkout is on the wrong branch.
python -c "
import rldx, torch
from rldx.configs.train_config import TrainConfig
assert 'save_only_model' in TrainConfig.__dataclass_fields__, 'wrong rldx: missing save_only_model'
print('rldx', rldx.__version__, 'from', rldx.__file__)
print('torch', torch.__version__, '| gpus', torch.cuda.device_count())
"

# --dataset-mix-ratios: frame-proportional sampling.
# RLDX-1 builds one spec per --dataset-paths entry, so factory.py's
# dataset_relative_lengths is always [1.0] -> weight = 1.0 x mix_ratio. Omitting the
# ratios would sample all 10 subsets with equal probability (10% each) regardless of
# size, which here would over-weight the 4-episode subsets ~10x.
# Values are effective steps = frames - ep x (action_horizon - 1); RLDX-1 drops the
# last 39 frames of every episode at horizon 40. Counted from meta/episodes.jsonl
# on 2026-08-19:
#   143842        13ep   4650f   4143   7.5%   (ep len 331-400)
#   150035        13ep   4771f   4264   7.7%   (ep len 343-403)
#   0818_091851   19ep   7957f   7216  13.0%   (ep len 379-447)
#   0818_101214    7ep   2617f   2344   4.2%   (ep len 100-442)
#   0819_105807   27ep   9935f   8882  16.0%   (ep len 220-483)
#   front_130940  74ep  18139f  15253  27.6%   (ep len 219-292)
#   front_134800   6ep   1456f   1222   2.2%   (ep len 226-257)
#   side_1_142441 13ep   5377f   4870   8.8%   (ep len 348-495)
#   side_1_163139 15ep   6376f   5791  10.5%   (ep len 333-519)
#   side_1_180500  4ep   1526f   1370   2.5%   (ep len 328-425)
#   total        191ep  62804f  55355
# epoch = 30000 x 128 / 55355 = 69.4
#
# 0814_full_v9_20260814_153724 (4ep / 1410f) is EXCLUDED: it has no meta/stats.json,
# the same reason the MLXP job dropped it. To include it, generate stats first with
# rldx/data/stats.py. Every other subset has stats.json, 30 fps, and the same
# camera_ego_left / camera_ego_right pair, so front_* and side_1_* differ only in
# scene layout and mix in fine under the egostereo modality config.
#
# 30 fps dataset, so --action-horizon 40 = 1.33 s. It must match
# rby1_wujihand2_data_config_egostereo_general_embodiment.py's action
# delta_indices=list(range(40)) or assembly.py raises immediately.
torchrun --nproc_per_node="$NUM_GPUS" --standalone \
    rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1-PT-IMG \
    --backbone-path RLWRLD/RLDX-1-VLM \
    --dataset-paths $DATA/0814_full_v9_20260814_143842_320x192 \
                    $DATA/0814_full_v9_20260814_150035_320x192 \
                    $DATA/0814_full_v9_20260818_091851_320x192 \
                    $DATA/0814_full_v9_20260818_101214_320x192 \
                    $DATA/0814_full_v9_20260819_105807_320x192 \
                    $DATA/0814_full_v9_front_20260819_130940_320x192 \
                    $DATA/0814_full_v9_front_20260819_134800_320x192 \
                    $DATA/0814_full_v9_side_1_20260819_142441_320x192 \
                    $DATA/0814_full_v9_side_1_20260819_163139_320x192 \
                    $DATA/0814_full_v9_side_1_20260819_180500_320x192 \
    --dataset-mix-ratios 4143 4264 7216 2344 8882 15253 1222 4870 5791 1370 \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path rldx/configs/data/rby1_wujihand2_data_config_egostereo_general_embodiment.py \
    --output-dir "$MODEL_OUTPUT_DIR" \
    --video-length 1 \
    --n-cog-tokens 64 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --global-batch-size 128 \
    --learning-rate 1e-4 \
    --max-steps 30000 \
    --dataloader-num-workers "$NUM_WORKERS" \
    --experiment-name "cjl_0819_v9_all191ep_wuji2_egostereo_ptimg_framewt_drop03_rtc12tr_bs128_30k_${NUM_GPUS}gpu_slurm" \
    --use-wandb --wandb-project RLDX-1 \
    --action-horizon 40 \
    --rtc-training-max-delay 12 \
    --state-dropout-prob 0.3 \
    --num-gpus "$NUM_GPUS" \
    --random-crop-fraction 0.95 \
    --save-only-model \
    --save-steps 9999999

echo "=== $(date -Is) done, checkpoints in $MODEL_OUTPUT_DIR"
[[ -n "$SCRATCH" ]] && rm -rf "$SCRATCH"
exit 0
