#!/bin/bash
#SBATCH --job-name=fuji-rby1m-rh56f1-distill-lora-action-expert
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --time=48:00:00   # 파티션 상한. 짧게 걸면 backfill 에는 유리하지만 30K 스텝이
                          # 중간에 잘린다 — 어댑터 변환이 학습 뒤에 붙어 있어서 잘리면
                          # adapter.pt 가 하나도 안 만들어진다.
                          # 참고 실측: dexjoco 잡 835 = 15,000스텝 / 장당배치 64 /
                          # 06:54:32 → 1.66s/스텝. 30,000스텝이면 ~14h.
#
# ★ 중요한 것은 GPU 수가 아니라 **장당 배치**다. --global-batch-size 는 "전역" 이라
#   장당 = BATCH/NGPU 가 되고, fsx H100 80GB 실측(잡 792)에서 **장당 128 이 OOM**
#   (79.0GiB 쓴 뒤 280MiB 부족), 장당 64 는 통과였다.
#   그래서 1장이면 BATCH=64, 2장이면 BATCH=128 이 같은 장당 배치이고 같은 속도다.
#   유효 배치가 달라지므로 **GPU 수를 바꾸면 실험이 달라진다** (지금은 64).
#SBATCH --output=out/fuji-distill-lora_%j.out
#SBATCH --error=out/fuji-distill-lora_%j.err
#
# action distillation 마지막 단계: relabel 된 액션으로 **action expert 만** LoRA 파인튜닝.
# MSAT DiT 를 freeze 하고 PEFT LoRA 만 주입한다. 손실은 원래 flow matching MSE 그대로고
# 바뀐 것은 데이터의 action 컬럼뿐이다 (PA-RL train.py:249 와 같은 발상).
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   sbatch --export=ALL,EXP=fuji_d4r16_0902_fuji_rc_success,\
#     DATA=rl-dataset/fuji/relabel_fuji_d4r16_0902_fuji_rc_success__success_ens_full@25k_success_ss3n10 \
#     sbatch/fuji/distill/lora_train.sbatch
#
# TAG 를 안 주면 DATA 이름의 '__' 뒤쪽에서 만든다 (@ 는 -). 즉 어떤 critic·어떤
# step_size 로 relabel 한 것인지가 출력 디렉토리에 그대로 남는다.
#
# ── dexjoco 판(sbatch/dexjoco/distill/lora_train.sbatch)과 다른 점 ──────────
#   partition   rlwrld -> dedicated-rd
#   gres        4장 -> 1장 (지금 쓸 수 있는 것이 1장. 장당 배치 64 는 그대로 유지)
#   HF_HOME     /workspace/... -> /fsx/rlwrld/junmo_cho/hf_cache  (fsx 클러스터)
#   EMB 폴백     GENERAL_EMBODIMENT -> NEW_EMBODIMENT
#               (rby1_f1_config.py:68 이 등록하는 것이 NEW_EMBODIMENT 다. GENERAL 로
#                주면 MODALITY_CONFIGS KeyError 로 죽는다 — 잡 초기에 겪은 그것이다)
#   adapter 변환 --root 를 명시한다 (그 스크립트 기본값이 fuji_distill 이지만 의존하지 않는다)
#
# ── STEPS 근거 ──────────────────────────────────────────────────────────────
# 224,097프레임 / batch 64 = 3,501 스텝/epoch.  30,000스텝 = 8.6 epoch.
# **오래 돌리면 critic 오차에 과적합한다.** 이번 relabel 의 변화량은 자연변화의
# 2.2배(관절당 1.75°/스텝)로 작아서, 최적 스텝이 어디인지 사전에 알 수 없다.
# 그래서 SAVE=2000 으로 15개를 남긴다 — 어댑터가 21MB 라 315MB 다.
# 나중에 serve_distill.sh 의 LORA_STEP 으로 갈아끼우며 고르면 된다 (재학습 불필요).
# 이전 fuji arm 들도 2000 마다 저장해 16000 까지 남겼다 — 같은 간격이라 비교가 된다.

# 제출 플러그인이 MODEL_OUTPUT_DIR 를 요구하는데 머신마다 허용 접두사가 다르다.
# ★ 플러그인은 **제출 시점의 환경**을 본다 — sbatch 로 던질 때 셸에서 미리 export 할 것.
if [ -z "${MODEL_OUTPUT_DIR:-}" ]; then
    for _p in /fsx/rlwrld-unified-checkpoints /rlwrld-unified-checkpoints; do
        if [ -d "$_p/$USER" ]; then MODEL_OUTPUT_DIR="$_p/$USER/rd-rl"; break; fi
    done
fi
export MODEL_OUTPUT_DIR

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
[ -d "$REPO/configs/exp" ] || { echo "REPO 가 레포 루트가 아니다: $REPO"; exit 2; }
cd "$REPO"

EXP=${EXP:-fuji_d4r16_0902_fuji_rc_success}
DATA=${DATA:?DATA 를 줘야 한다 (relabel 이 만든 데이터셋, 레포 상대경로)}
YAML=configs/exp/$EXP.yaml
[ -f "$YAML" ] || { echo "exp yaml 이 없다: $YAML"; exit 2; }
[ -d "$REPO/$DATA" ] || { echo "데이터 없음: $REPO/$DATA"; exit 3; }

# TAG: 안 주면 DATA 이름에서 만든다. relabel 출력이
#   relabel_<exp>__<ctag>@<N>k_<eps>_ss<S>n<G>
# 이므로 '__' 뒤가 곧 "어떤 critic / 어떤 step_size" 다. @ 는 경로에서 다루기
# 번거로우니 - 로 바꾼다.
if [ -z "${TAG:-}" ]; then
    TAG=$(basename "$DATA")
    TAG=${TAG#*__}
    TAG=${TAG//@/-}
    [ "$TAG" = "$(basename "$DATA")" ] && TAG=${TAG#relabel_}   # '__' 가 없으면 폴백
fi

STEPS=${STEPS:-30000}
SAVE=${SAVE:-2000}
[ "$SAVE" -gt "$STEPS" ] && SAVE=$STEPS
BATCH=${BATCH:-64}                      # 전역 배치. GPU 1장 기준 = 장당 64 (안전한 상한)
LR=${LR:-1e-4}
# ★ --tune-top-llm-layers 기본값이 **4** 다. tune_llm=False 여도 백본 LLM 상단 4개 층이
#   학습 대상이 되어 (adapter.py:242) 800M 파라미터가 풀 파인튜닝된다 — 체크포인트가
#   커지는 이유이자 "action expert 만 LoRA" 라는 설계와 다른 실험이 되는 이유다.
TOPLLM=${TOPLLM:-0}
# ── cog_emb 을 학습에 포함할지 ────────────────────────────────────────────────
# 기본값 0(= 학습함) 은 launch_train 의 기본값(freeze_cog_tokens=False)과 같다.
# 그래서 어댑터에는 lora 64개 텐서(4,784,128) 외에 backbone.cog_emb (64x4096 =
# 262,144) 이 함께 들어간다 — 저장 파라미터 5,046,272 중 5.2%.
#
# ★ 언제 1 로 켜야 하는가: **증류된 정책 위에 critic 을 다시 얹을 때** (serve 의
#   METHOD=parl/sel32, 즉 --lora 와 --artifacts 동시 사용).
#   critic 은 base policy 가 낸 cog feature (cogfeat.npy) 로 학습됐는데, LoRA 가
#   cog_emb 을 흔들면 배포 시 정책이 내는 cog feature 가 critic 이 본 것과 달라져
#   Q 가 조용히 엉뚱해진다 (에러는 안 난다 — 그래서 위험하다).
#   METHOD=bc 로만 서빙할 거면 상관없다.
#
# 2026-09-02 의 두 arm(ss1n10 / ss3n10)은 0 으로 돌렸다 — 그때는 bc 서빙만 계획했다.
# 값을 바꾸면 arm 간 비교가 깨지므로, 기존 arm 과 대조할 런은 같은 값을 쓸 것.
FREEZE_COG=${FREEZE_COG:-0}
if [ "$FREEZE_COG" = 1 ]; then COG_ARG=--freeze-cog-tokens; else COG_ARG=--no-freeze-cog-tokens; fi
NCK=$((STEPS / SAVE))
SAVE_LIMIT=${SAVE_LIMIT:-$NCK}          # 기본값 5 면 마지막 5개만 남는다 — 전부 남긴다
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-rd-rl-distill}

DISTILL_ROOT=checkpoints/${EXP%%_*}_distill          # fuji_distill
OUT=$REPO/$DISTILL_ROOT/$TAG

# ★ 전부 exp yaml / 모델 config 에서 온다. 하드코딩하면 조용히 틀린다.
BASE=$REPO/checkpoints/$(awk '/^base_policy:/{print $2; exit}' "$YAML")
MODCFG=$(awk '/^rldx_data_config:/{print $2; exit}' "$YAML")
AH=$(awk '/^action_horizon:/{print $2; exit}' "$YAML")
[ -d "$BASE" ] || { echo "base policy 없음: $BASE"; exit 3; }
# rtc_training_max_delay 는 **base 모델 config 에서** 읽는다 (사전학습과 맞춰야 한다).
# exp yaml 의 inference_latency 와 다를 수 있다 — critic 은 yaml 값으로 창을 잡으므로
# 어긋나면 배포 시 어느 한쪽이 맞지 않는다. 조용히 지나가지 않게 경고를 낸다.
RTCD=$(python3 -c "import json;print(json.load(open('$BASE/config.json'))['rtc_training_max_delay'])")
LAT_YAML=$(awk '/^inference_latency:/{print $2; exit}' "$YAML")
if [ "$RTCD" != "$LAT_YAML" ]; then
  echo "★ 경고: base 모델 rtc_training_max_delay=$RTCD 인데 $YAML 의"
  echo "   inference_latency=$LAT_YAML 이다. LoRA 는 base 값($RTCD)으로 학습하지만,"
  echo "   critic 은 yaml 값($LAT_YAML)으로 창을 잡았다. 실기 배포 시 rrc 의"
  echo "   inference_latency_steps 를 무엇으로 맞출지 결정해야 한다."
fi
# embodiment tag 는 **BC 학습이 쓴 값과 같아야 한다.** conf.yaml 은 소문자, CLI 는 대문자.
EMB=$(grep -m1 -oP '^\s*embodiment_tag:\s*\K\S+' "$BASE/experiment_cfg/conf.yaml" 2>/dev/null | tr 'a-z' 'A-Z')
EMB=${EMB:-NEW_EMBODIMENT}

# LeRobot 데이터셋 경로 목록. fuji 는 루트 아래에 세션 디렉토리가 여러 개다
# (relabel 출력도 같은 구조를 유지한다). launch_train 은 --dataset-paths 를 우선한다.
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

export HF_HOME=${HF_HOME:-/fsx/rlwrld/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
[ -f "$HOME/.rldx_secrets.sh" ] && source "$HOME/.rldx_secrets.sh"

# wandb 는 있으면 쓰고 없으면 조용히 끈다. 16K 스텝짜리 학습이 로깅 키 하나 없다고
# 죽으면 안 된다.
WB_ARGS=""
if [ "$WANDB" = 1 ]; then
    if [ -n "${WANDB_API_KEY:-}" ]; then
        WB_ARGS="--use-wandb --wandb-project $WANDB_PROJECT"
        echo "    wandb: $WANDB_PROJECT / ${EXP%%_*}_distill_${TAG}  (키 확인됨)"
    else
        echo "    [경고] WANDB=1 인데 WANDB_API_KEY 가 없다 → 로깅 없이 진행한다"
        echo "           ~/.rldx_secrets.sh 에 export WANDB_API_KEY=... 를 넣을 것"
    fi
fi

NGPU=$(nvidia-smi -L | wc -l)
echo "=== $(date -Is)  fuji distill LoRA  tag=$TAG  gpus=$NGPU  job ${SLURM_JOB_ID:-local}"
echo "    exp    $EXP"
echo "    base   $BASE"
echo "    data   $REPO/$DATA"
echo "    out    $OUT"
echo "    modality=$MODCFG  horizon=$AH  rtc_max_delay=$RTCD  embodiment=$EMB  top_llm=$TOPLLM"
echo "    cog_emb $([ "$FREEZE_COG" = 1 ] && echo 'freeze (증류판 위에 critic 을 얹을 수 있다)' \
                                          || echo '학습함 (critic 재사용 불가 — bc 서빙 전용)')"
echo "    steps=$STEPS  batch=$BATCH (장당 $((BATCH/NGPU)))  save마다=$SAVE  lr=$LR"
echo "    체크포인트 ${NCK}개 보존 (save_total_limit=$SAVE_LIMIT) — 어댑터 21MB/개 ≈ $((NCK * 21))MB"
echo "    예상 소요 ≈ $((STEPS * 166 / 100 / 3600))h  (1.66s/스텝, dexjoco 잡 835 실측)"
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
# OOM 판정은 장당 배치로 한다 (GPU 수 자체는 무관하다).
if [ $((BATCH / NGPU)) -gt 64 ]; then
  echo "    ★ 경고: 장당 배치 $((BATCH/NGPU)) > 64. 잡 792 에서 장당 128 이 OOM 이었다"
  echo "      BATCH 를 $((64 * NGPU)) 로 줄이거나 GPU 를 늘릴 것"
fi
echo

cd "$REPO/third_party/RLDX-1"
# ★ torchrun 은 **venv 안의 것**을 절대경로로 부른다. 맨 `torchrun` 은 PATH 에 의존해서
#   venv 를 활성화하지 않은 클러스터에서 "command not found" 로 2초 만에 죽는다 (잡 783).
TORCHRUN="$REPO/third_party/RLDX-1/.venv/bin/torchrun"
[ -x "$TORCHRUN" ] || { echo "torchrun 이 없다: $TORCHRUN"; exit 3; }
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
    "$COG_ARG" \
    --global-batch-size "$BATCH" --learning-rate "$LR" \
    --max-steps "$STEPS" --save-steps "$SAVE" --save-total-limit "$SAVE_LIMIT" \
    --num-gpus "$NGPU" --dataloader-num-workers 8 \
    --output-dir "$OUT" \
    --experiment-name "${EXP%%_*}_distill_${TAG}" \
    $WB_ARGS || exit 1

# ── 어댑터 변환 (안전망) ─────────────────────────────────────────────────────
# 2026-09-03 부터 RLDX-1 의 CheckpointFormatCallback.on_save 가 **저장 시점마다**
# adapter.pt 를 만든다 (rldx/experiment/utils.py). 그래서 학습 중에도 앞쪽 체크포인트를
# 바로 서빙할 수 있고, 잡이 잘려도 그때까지의 어댑터가 남는다.
# 아래 한 번 더 도는 것은 안전망이다 — 콜백이 실패했거나(예외를 삼키고 계속 간다)
# 옛 체크포인트가 섞여 있을 때를 위한 것이고, 덮어쓰기라 부작용이 없다.
# ── 어댑터를 serve 의 --lora 형식(.pt)으로 변환 ──────────────────────────────
# merge(13GB/arm) 를 만들지 않는다. ExpoServer._load (rl/vla_rldx.py:540) 가 sd["lora"]
# 를 읽어 base 위에 주입하는 경로가 이미 있고 키 이름도 그대로 맞는다. 체크포인트당
# 10MB 라 전 스텝을 남겨도 부담이 없고, 그래야 "몇 스텝이 최적인가" 를 재학습 없이
# 고를 수 있다. adapter.pt 는 checkpoint-N/ **안에** 쓴다 (model.safetensors 옆).
cd "$REPO"
"$REPO/third_party/RLDX-1/.venv/bin/python" "$REPO/utils/adapter_to_artifacts.py" \
    --root "$DISTILL_ROOT" --arm "$TAG" --all-steps \
    || echo "  [경고] adapter 변환 실패 — 나중에 수동으로: utils/adapter_to_artifacts.py --root $DISTILL_ROOT --arm $TAG --all-steps"

echo
echo "=== $(date -Is)  완료  $OUT"
echo "서빙:  ARM=$TAG EXP=$EXP bash sbatch/fuji/rollout_w_critic/serve_distill.sh"
echo "스텝 스윕:  for s in 2000 6000 12000 20000 30000; do ARM=$TAG LORA_STEP=\$s EXP=$EXP bash sbatch/fuji/rollout_w_critic/serve_distill.sh; done"
