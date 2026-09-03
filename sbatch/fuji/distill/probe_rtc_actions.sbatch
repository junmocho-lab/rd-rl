#!/bin/bash
#SBATCH --job-name=fuji-probe-rtc-actions
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --time=3:00:00
#SBATCH --output=out/fuji-probe-rtc_%j.out
#SBATCH --error=out/fuji-probe-rtc_%j.err
#
# 에피소드 하나에서 RTC 결정 시점마다 액션 거리를 재고 비디오로 만든다.
#   리플레이(로그)  vs  BC 샘플 32개(mean±std)  vs  PA-RL 최적화 (step_size 를 쓸어가며)
#
# 지표는 **L1 평균** mean|Δ| over (right_arm 7관절 x 실행 16스텝) — "관절 하나가 한
# 스텝에 평균 몇만큼 다른가". raw 로 환산하면 그대로 평균 절대 관절 차이[rad] 다.
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   sbatch sbatch/fuji/distill/probe_rtc_actions.sbatch
#
#   # 에피소드 지정 / critic 스텝 변경 / step_size 목록 변경
#   sbatch --export=ALL,EPISODES=auto:10,CRITIC_STEP=25000 sbatch/fuji/distill/probe_rtc_actions.sbatch
#   sbatch --export=ALL,STEP_SIZES=0,0.03,0.1,0.3,1 sbatch/fuji/distill/probe_rtc_actions.sbatch
#
# ★ --export 는 콤마로 쪼개진다. STEP_SIZES 처럼 콤마가 들어간 값은 위처럼 그대로
#   이어 써도 sbatch 가 각각을 별개 변수로 보다가 마지막만 남는다. 콤마 있는 값은
#   **먼저 export 하고 --export=ALL 만** 쓸 것:
#     export STEP_SIZES=0,1e-4,1e-3,1e-2 && sbatch sbatch/fuji/distill/probe_rtc_actions.sbatch
#
# ── 무엇을 보려는 잡인가 ────────────────────────────────────────────────────
# step_size 를 정하기 전에 답해야 하는 질문이 두 개다:
#   1. 최적화된 액션이 리플레이 액션과 **의미 있게** 다른가?
#      "의미 있게" 의 단위는 리플레이 궤적 자체의 1프레임 변화량이다 (fuji 실측 0.0051).
#      거리가 그 1/10 이면 로봇은 차이를 느끼지 못한다.
#   2. 그 차이가 critic 때문인가, 그냥 정책 샘플링 노이즈인가?
#      BC 샘플 32개를 하나하나 리플레이와 비교해 mean±std 밴드를 그린다. 최적화
#      곡선이 그 밴드 안에 있으면 "다시 뽑은 것" 과 구별되지 않는다 = critic 은
#      아무 일도 하지 않았다. 32개는 후보 샘플링에서 이미 뽑는 것이라 공짜다.
# step_size=0 도 목록에 넣는다 — 상승을 끄고 **선택만** 한 결과라, 선택 효과와
# 상승 효과가 분리된다.
#
# ── 왜 GPU 를 요청하는가 (critic_grid 와 다른 점) ────────────────────────────
# critic_grid 는 cogfeat.npy 를 읽어 critic MLP 만 돌리므로 CPU 로 충분했다.
# 여기는 **base policy(1.24B VLA)로 후보를 직접 샘플링**한다 — GPU 가 필수다.
# 결정 39개 x 후보 32개면 VLA forward 가 5배치라 몇 분이면 끝난다.
#
# step_size 목록은 공짜에 가깝다: 비싼 쪽(VLA 샘플링)을 한 번만 하고 상승만 다시
# 돌린다. 그래서 6개를 쓸어도 1개와 시간이 거의 같다.

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
[ -d "$REPO/configs/exp" ] || { echo "REPO 가 레포 루트가 아니다: $REPO"; exit 2; }
cd "$REPO"

# 제출 플러그인이 MODEL_OUTPUT_DIR 를 요구한다. 이 잡은 거기에 아무것도 쓰지 않는다
# (산출물은 <exp>-critic/<tag>/plots/probe/ 아래). 플러그인 통과용이다.
if [ -z "${MODEL_OUTPUT_DIR:-}" ]; then
    for _p in /fsx/rlwrld-unified-checkpoints /rlwrld-unified-checkpoints; do
        if [ -d "$_p/$USER" ]; then MODEL_OUTPUT_DIR="$_p/$USER/rd-rl"; break; fi
    done
fi
export MODEL_OUTPUT_DIR

EXP=${EXP:-fuji_d4r16}
TAG=${TAG:-success}
CRITIC_STEP=${CRITIC_STEP:-10000}
EPISODES=${EPISODES:-auto:6}       # 세션을 돌아가며 성공 에피소드 6개
NUM_STEPS=${NUM_STEPS:-10}             # PA-RL 원본 기본값
NUM_SAMPLES=${NUM_SAMPLES:-32}
NUM_KEEP=${NUM_KEEP:-10}
BATCH=${BATCH:-8}
CAM_NAME=${CAM_NAME:-}
# 0 을 반드시 포함한다 (상승 없음 = 선택만). PA-RL 원본값 3e-4 가 우리 Q 스케일
# ([0,1], 원본은 [-100,0])에서 사실상 아무 일도 못 한다는 것을 눈으로 확인하려고
# 위아래로 두 자릿수씩 벌려 둔다.
STEP_SIZES=${STEP_SIZES:-0,3e-4,1e-2,1,10,100}

YAML=$REPO/configs/exp/$EXP.yaml
[ -f "$YAML" ] || { echo "exp yaml 이 없다: $YAML"; exit 2; }
WORK=$REPO/checkpoints/${EXP}-critic
CF=$([ "$CRITIC_STEP" -eq 0 ] && echo critic_latest.pt || printf 'critic_%06d.pt' "$CRITIC_STEP")
if [ ! -f "$WORK/$TAG/$CF" ]; then
  echo "critic 이 없다: $WORK/$TAG/$CF"
  echo "있는 것:"; ls "$WORK/$TAG" 2>/dev/null | grep '^critic_' || echo "  (없음)"
  exit 3
fi
# actnorm.npy / images.mm / cogfeat.npy 는 critic 학습이 만든 캐시다. 없으면 여기서
# 새로 만들다가 1.5시간을 쓰게 되므로 미리 걸러 준다.
for f in actnorm.npy images.mm cogfeat.npy; do
  [ -e "$WORK/$f" ] || { echo "캐시가 없다: $WORK/$f — critic 을 한 번 학습할 것"; exit 3; }
done

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
export HF_HOME=${HF_HOME:-/fsx/rlwrld/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PY=${PY:-$REPO/third_party/RLDX-1/.venv/bin/python}
[ -x "$PY" ] || { echo "python 이 없다: $PY"; exit 3; }

echo "=== $(date -Is)  fuji probe-rtc-actions  job ${SLURM_JOB_ID:-local}"
echo "    exp        $EXP   ($(awk '/^dataset:/{print $2; exit}' "$YAML"))"
echo "    critic     $TAG/$CF"
echo "    에피소드   $EPISODES"
echo "    step_sizes $STEP_SIZES   num_steps=$NUM_STEPS"
echo "    후보       M=$NUM_SAMPLES K=$NUM_KEEP"
echo "    산출물     $WORK/$TAG/plots/probe/"
nvidia-smi -L 2>/dev/null | sed 's/^/    /'
echo

ARGS=(--exp "$EXP" --tag "$TAG" --critic-step "$CRITIC_STEP"
      --episodes "$EPISODES" --step-sizes "$STEP_SIZES" --num-steps "$NUM_STEPS"
      --num-samples "$NUM_SAMPLES" --num-keep "$NUM_KEEP" --batch "$BATCH")
[ -n "$CAM_NAME" ] && ARGS+=(--cam-name "$CAM_NAME")

"$PY" -u utils/probe_rtc_actions.py "${ARGS[@]}"
rc=$?

echo
echo "=== $(date -Is)  종료 rc=$rc"
if [ $rc -eq 0 ]; then
  echo "산출물:"
  ls -la "$WORK/$TAG/plots/probe/" 2>/dev/null | sed 's/^/  /'
fi
exit $rc
