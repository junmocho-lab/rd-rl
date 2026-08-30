#!/bin/bash
#SBATCH --job-name=dexjoco-hammer-nail-distill-lora-finetune-action-expert-from-relabeled-actions
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --no-requeue
#SBATCH --time=4:00:00
#SBATCH --output=out/dexjoco-distill-lora_%j.out
#SBATCH --error=out/dexjoco-distill-lora_%j.err
#
# action distillation 3단계: 개선된 액션이 들어간 데이터셋으로 action expert 만 LoRA 파인튜닝.
# MSAT DiT 를 freeze 하고 PEFT LoRA 만 주입한다 (rldx/model/core/rldx.py:219).
# 손실은 원래 flow matching MSE 그대로다 — 바뀐 것은 데이터의 action 컬럼뿐이다.
#
#   sbatch --export=ALL,DATA=rl-dataset/dexjoco/distill_arm2_relabel_success,TAG=arm2 \
#          sbatch/dexjoco/distill/lora_train.sbatch
#
# **max-steps 를 작게 잡을 것.** 오래 돌리면 critic 오차에 과적합한다. 성공 123ep =
# 30,953프레임, 배치 64 면 484 스텝/epoch → 2000 스텝 ≈ 4 epoch.

# 제출 플러그인이 MODEL_OUTPUT_DIR 를 요구하는데 머신마다 허용 접두사가 다르다.
# ★ 플러그인은 제출 시점의 환경을 본다 — sbatch 로 던질 때는 셸에서 미리 export 할 것.
if [ -z "${MODEL_OUTPUT_DIR:-}" ]; then
    for _p in /fsx/rlwrld-unified-checkpoints /rlwrld-unified-checkpoints; do
        if [ -d "$_p/$USER" ]; then MODEL_OUTPUT_DIR="$_p/$USER/rd-rl"; break; fi
    done
fi
export MODEL_OUTPUT_DIR

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
if [ ! -d "$REPO/configs/exp" ]; then
    echo "REPO 가 레포 루트가 아니다: $REPO"; exit 2
fi
cd "$REPO"

EXP=${EXP:-dexjoco_hammer_nail_d2r8_s0}
DATA=${DATA:?DATA 를 줘야 한다 (학습할 LeRobot 데이터셋, 레포 상대경로)}
TAG=${TAG:?TAG 를 줘야 한다 (arm1 / arm2 / arm4)}
STEPS=${STEPS:-2000}
BATCH=${BATCH:-64}
LR=${LR:-1e-4}
OUT=$REPO/checkpoints/${EXP%%_*}_distill/${TAG}

BASE=$REPO/checkpoints/$(awk '/^base_policy:/{print $2; exit}' configs/exp/$EXP.yaml)
# ★ 전부 exp yaml / 모델 config 에서 온다. 하드코딩하면 태스크를 옮길 때 조용히 틀린다
#   (dexjoco 는 panda_allegro / 32 / 8, fuji 는 rby1_f1 / 40 / 4).
MODCFG=$(awk '/^rldx_data_config:/{print $2; exit}' configs/exp/$EXP.yaml)
AH=$(awk '/^action_horizon:/{print $2; exit}' configs/exp/$EXP.yaml)
RTCD=$(python3 -c "import json;print(json.load(open('$BASE/config.json'))['rtc_training_max_delay'])")
[ -d "$BASE" ] || { echo "base policy 없음: $BASE"; exit 3; }
[ -d "$REPO/$DATA" ] || { echo "데이터 없음: $REPO/$DATA"; exit 3; }

export HF_HOME=${HF_HOME:-/workspace/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
[ -f "$HOME/.rldx_secrets.sh" ] && source "$HOME/.rldx_secrets.sh"

NGPU=$(nvidia-smi -L | wc -l)
echo "=== $(date -Is)  distill LoRA  tag=$TAG  gpus=$NGPU"
echo "    base $BASE"
echo "    data $REPO/$DATA"
echo "    out  $OUT"
echo "    modality=$MODCFG  horizon=$AH  rtc_max_delay=$RTCD"
"$REPO/third_party/RLDX-1/.venv/bin/python" -c "
import json;i=json.load(open('$REPO/$DATA/meta/info.json'))
print(f\"    데이터셋 {i['total_episodes']}ep / {i['total_frames']}프레임 → \"
      f\"배치 $BATCH 에서 {i['total_frames']//$BATCH} 스텝/epoch, $STEPS 스텝 = {$STEPS/(i['total_frames']//$BATCH):.1f} epoch\")"

cd "$REPO/third_party/RLDX-1"
# use_wandb 기본값이 False 라 (train_config.py:390) 따로 끄지 않는다.
torchrun --nproc_per_node="$NGPU" --standalone \
    rldx/experiment/launch_train.py \
    --base-model-path "$BASE" \
    --dataset-path "$REPO/$DATA" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path "$MODCFG" \
    --video-length 1 --n-cog-tokens 64 \
    --action-horizon "$AH" --rtc-training-max-delay "$RTCD" \
    --action-model-use-lora True --save-trainable-only True \
    --tune-projector False --tune-llm False --tune-visual False \
    --global-batch-size "$BATCH" --learning-rate "$LR" \
    --max-steps "$STEPS" --save-steps $((STEPS/4)) \
    --num-gpus "$NGPU" --dataloader-num-workers 8 \
    --output-dir "$OUT" \
    --experiment-name "dexjoco_distill_${TAG}" || exit 1
echo "=== $(date -Is)  완료  $OUT"
