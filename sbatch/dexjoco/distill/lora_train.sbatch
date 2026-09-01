#!/bin/bash
#SBATCH --job-name=dexjoco-hammer-nail-distill-lora-finetune-action-expert-from-relabeled-actions
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4   # 할당된 만큼 전부 쓴다 (NGPU 를 nvidia-smi 로 세어 torchrun 에 넘긴다).
                       # 더/덜 쓰려면 제출할 때 sbatch --gres=gpu:N 으로 덮으면 된다.
                       # --global-batch-size 는 "전역" 이라 GPU 수가 바뀌어도 유효 배치는 같다
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
WANDB=${WANDB:-1}                       # 0 이면 끈다
WANDB_PROJECT=${WANDB_PROJECT:-rd-rl-distill}
BATCH=${BATCH:-128}                     # 전역 배치. GPU 4장이면 장당 32
# ★ 전역 배치라 **GPU 수가 곧 장당 배치를 정한다**. fsx H100 80GB 실측(잡 792):
#   장당 128 -> OOM (79.0GiB 사용 후 280MiB 부족).  즉 GPU 1장 + BATCH 128 은 불가.
#   장당 64 (GPU 2장) 이 안전한 하한이고, GPU 4장(장당 32)이면 여유롭다.
#   GPU 를 줄이려면 BATCH 도 같이 줄여야 한다 — 유효 배치가 바뀌므로 실험이 달라진다.
# 저장 간격. LoRA 어댑터만 저장하므로(--save-trainable-only) 체크포인트가 작다 —
# 촘촘히 남겨 두면 나중에 "몇 스텝이 최적인가" 를 재학습 없이 고를 수 있다.
# dexjoco 에서 critic 스텝이 성능을 18pp 갈랐고 세팅마다 최적점이 달랐다. 같은 일이
# distillation 스텝에서도 일어난다면 여기 남겨둔 체크포인트가 그대로 답이 된다.
# --save-only-model 을 켜서 optimizer/scheduler/RNG 상태를 안 남긴다. 그게 없으면
# 체크포인트 하나가 8.7GB 가 되어 (preflight 실측) 30개면 260GB 다. 어댑터만 남기면
# 훨씬 작아진다 — 그래야 촘촘히 저장하는 것이 의미가 있다.
SAVE=${SAVE:-1000}
[ "$SAVE" -gt "$STEPS" ] && SAVE=$STEPS
# ★ save_total_limit 기본값이 5 라 그냥 두면 **마지막 5개만 남고 나머지는 지워진다.**
#   촘촘히 저장하는 목적이 "나중에 최적 스텝을 고르는 것" 이므로 전부 남겨야 한다.
# ★ --tune-top-llm-layers 기본값이 **4** 다. tune_llm=False 여도 백본 LLM 상단 4개 층이
#   학습 대상이 되어 (adapter.py:242) 800M 파라미터가 풀 파인튜닝된다 — 체크포인트가
#   2.9GB 가 된 이유이자, "action expert 만 LoRA" 라는 설계와 다른 실험이 되는 이유다.
#   0 으로 꺼야 어댑터만 학습된다. 백본까지 흔들고 싶으면 TOPLLM=4 로 되돌린다.
TOPLLM=${TOPLLM:-0}
NCK=$((STEPS / SAVE))
SAVE_LIMIT=${SAVE_LIMIT:-$NCK}
LR=${LR:-1e-4}
OUT=$REPO/checkpoints/${EXP%%_*}_distill/${TAG}

BASE=$REPO/checkpoints/$(awk '/^base_policy:/{print $2; exit}' configs/exp/$EXP.yaml)
# ★ 전부 exp yaml / 모델 config 에서 온다. 하드코딩하면 태스크를 옮길 때 조용히 틀린다
#   (dexjoco 는 panda_allegro / 32 / 8, fuji 는 rby1_f1 / 40 / 4).
MODCFG=$(awk '/^rldx_data_config:/{print $2; exit}' configs/exp/$EXP.yaml)
AH=$(awk '/^action_horizon:/{print $2; exit}' configs/exp/$EXP.yaml)
RTCD=$(python3 -c "import json;print(json.load(open('$BASE/config.json'))['rtc_training_max_delay'])")
# ★ embodiment tag 는 **BC 학습이 쓴 값과 같아야 한다.** 하드코딩하면 태스크를 옮길 때
#   조용히 틀린다 (dexjoco=general_embodiment, fuji=new_embodiment).
#   conf.yaml 은 소문자 값을, CLI 는 enum 이름(대문자)을 쓴다.
EMB=$(grep -m1 -oP '^\s*embodiment_tag:\s*\K\S+' "$BASE/experiment_cfg/conf.yaml" 2>/dev/null | tr 'a-z' 'A-Z')
EMB=${EMB:-GENERAL_EMBODIMENT}
[ -d "$BASE" ] || { echo "base policy 없음: $BASE"; exit 3; }
[ -d "$REPO/$DATA" ] || { echo "데이터 없음: $REPO/$DATA"; exit 3; }

# LeRobot 데이터셋 경로 목록을 만든다.
#   dexjoco : 루트 자체가 하나의 데이터셋      → 1개
#   fuji    : 루트 아래에 세션 디렉토리 여러 개 → N개
# relabel/make_subset 출력도 입력과 같은 구조를 유지하므로 여기서 한 번에 처리한다.
# launch_train 은 --dataset-paths (복수) 가 --dataset-path 보다 우선한다.
collect_datasets() {
    local root=$1
    if [ -f "$root/meta/info.json" ]; then
        echo "$root"
    else
        local d
        for d in "$root"/*/; do
            [ -f "$d/meta/info.json" ] && echo "${d%/}"
        done
    fi
}
mapfile -t DPATHS < <(collect_datasets "$REPO/$DATA")
[ ${#DPATHS[@]} -gt 0 ] || { echo "LeRobot 데이터셋을 못 찾았다: $REPO/$DATA"; exit 3; }

export HF_HOME=${HF_HOME:-/workspace/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# 단편화로 인한 OOM 을 줄인다 (OOM 메시지가 직접 권하는 설정).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
[ -f "$HOME/.rldx_secrets.sh" ] && source "$HOME/.rldx_secrets.sh"

# wandb 는 있으면 쓰고 없으면 조용히 끈다. 30K 스텝짜리 학습이 로깅 키 하나 없다고
# 죽으면 안 된다 — 밤새 도는 체인에서는 특히.
WB_ARGS=""
if [ "$WANDB" = 1 ]; then
    if [ -n "${WANDB_API_KEY:-}" ]; then
        WB_ARGS="--use-wandb --wandb-project $WANDB_PROJECT"
        echo "    wandb: $WANDB_PROJECT / ${EXP%%_*}_distill_${TAG}"
    else
        echo "    [경고] WANDB=1 인데 WANDB_API_KEY 가 없다 → 로깅 없이 진행한다"
        echo "           ~/.rldx_secrets.sh 에 export WANDB_API_KEY=... 를 넣을 것"
    fi
fi

NGPU=$(nvidia-smi -L | wc -l)
echo "=== $(date -Is)  distill LoRA  tag=$TAG  gpus=$NGPU"
echo "    base $BASE"
echo "    data $REPO/$DATA"
echo "    out  $OUT"
echo "    modality=$MODCFG  horizon=$AH  rtc_max_delay=$RTCD  embodiment=$EMB  top_llm_layers=$TOPLLM"
echo "    steps=$STEPS  batch=$BATCH (장당 $((BATCH/NGPU)))  save마다=$SAVE"
echo "    체크포인트 ${NCK}개 보존 (save_total_limit=$SAVE_LIMIT) — preflight 실측 2.9GB/개 기준 약 $((NCK * 29 / 10))GB"
df -h "$(dirname "$OUT")" 2>/dev/null | tail -1 | awk '{print "    디스크 여유 "$4" ("$5" 사용)"}'
echo "    데이터셋 ${#DPATHS[@]}개:"
printf '      %s\n' "${DPATHS[@]}"
"$REPO/third_party/RLDX-1/.venv/bin/python" -c "
import json,sys
ps=sys.argv[1:]
ep=sum(json.load(open(p+'/meta/info.json'))['total_episodes'] for p in ps)
fr=sum(json.load(open(p+'/meta/info.json'))['total_frames'] for p in ps)
spe=max(fr//$BATCH,1)
print(f'    합계 {ep}ep / {fr}프레임 → 배치 $BATCH 에서 {spe} 스텝/epoch, '
      f'$STEPS 스텝 = {$STEPS/spe:.1f} epoch')" "${DPATHS[@]}"

cd "$REPO/third_party/RLDX-1"
# ★ torchrun 은 **venv 안의 것**을 절대경로로 부른다. 맨 `torchrun` 은 PATH 에 의존해서
#   venv 를 활성화하지 않은 클러스터에서 "torchrun: command not found" 로 2초 만에 죽는다
#   (fsx 실측: 잡 783). 다른 단계들은 .venv/bin/python 을 절대경로로 쓰는데 여기만
#   빠져 있었다.
TORCHRUN="$REPO/third_party/RLDX-1/.venv/bin/torchrun"
[ -x "$TORCHRUN" ] || { echo "torchrun 이 없다: $TORCHRUN"; exit 3; }
# use_wandb 기본값이 False 라 (train_config.py:390) WANDB=0 이면 아무것도 안 붙인다.
# tyro 파서는 불리언을 --flag / --no-flag 로 받는다. "--flag True" 로 주면
#   Unrecognized arguments: True True False False False
# 로 죽는다 (PARL-DISTILL.md 의 예시가 그 형식이라 그대로 옮기면 안 된다).
"$TORCHRUN" --nproc_per_node="$NGPU" --standalone \
    rldx/experiment/launch_train.py \
    --base-model-path "$BASE" \
    --dataset-paths "${DPATHS[@]}" \
    --embodiment-tag "$EMB" \
    --modality-config-path "$MODCFG" \
    --video-length 1 --n-cog-tokens 64 \
    --action-horizon "$AH" --rtc-training-max-delay "$RTCD" \
    --action-model-use-lora --save-trainable-only \
    --save-only-model \
    --tune-top-llm-layers "$TOPLLM" \
    --no-tune-projector --no-tune-llm --no-tune-visual \
    --global-batch-size "$BATCH" --learning-rate "$LR" \
    --max-steps "$STEPS" --save-steps "$SAVE" --save-total-limit "$SAVE_LIMIT" \
    --num-gpus "$NGPU" --dataloader-num-workers 8 \
    --output-dir "$OUT" \
    --experiment-name "${EXP%%_*}_distill_${TAG}" \
    $WB_ARGS || exit 1

# ── 어댑터를 serve 의 --artifacts 형식(.pt)으로 변환 ─────────────────────────
# merge(13GB/arm) 대신 이걸 쓴다. ExpoServer._load (rl/vla_rldx.py:540) 가
# sd["lora"] 를 읽어 base 위에 LoRA 를 주입하는 경로가 이미 있고 (EXPO-FT 가 쓰던 것),
# 키 이름도 그대로 맞는다. 체크포인트당 10MB 라 전 스텝을 남겨도 부담이 없고,
# 그래야 "몇 스텝이 최적인가" 를 재학습·재merge 없이 고를 수 있다.
"$REPO/third_party/RLDX-1/.venv/bin/python" "$REPO/utils/adapter_to_artifacts.py" \
    --arm "$TAG" --all-steps || echo "  [경고] artifacts 변환 실패 — 나중에 수동으로 돌릴 것"

echo "=== $(date -Is)  완료  $OUT"
