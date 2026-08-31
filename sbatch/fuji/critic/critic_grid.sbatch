#!/bin/bash
#SBATCH --job-name=fuji-rby1m-rh56f1-critic-grid-holdout-q-by-checkpoint-step
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --time=1:00:00
#SBATCH --output=out/fuji-critic-grid_%j.out
#SBATCH --error=out/fuji-critic-grid_%j.err
#
# 학습이 끝난 critic 체크포인트들을 홀드아웃에서 비교한다 (utils/critic_grid.py).
# 산출물: <exp>-critic/<TAG>/plots/grid/
#   <TAG>_<steps>.mp4   행=에피소드. 왼쪽 카메라 + 오른쪽 Q 곡선(색=체크포인트 스텝)
#                       + 현재 프레임 빨간 커서. 스텝별 AUC 표는 로그에 찍힌다.
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   sbatch --export=ALL,EXP=fuji_d3r8,TAG=all     sbatch/fuji/critic/critic_grid.sbatch
#   sbatch --export=ALL,EXP=fuji_d3r8,TAG=success sbatch/fuji/critic/critic_grid.sbatch
#
# ── GPU 를 요청하지 않는 이유 ────────────────────────────────────────────────
# 이 잡은 VLA 백본을 올리지 않는다. cogfeat.npy(3GB)를 읽어 17.4M critic MLP 를
# 돌리는 것이 전부라 CPU 로 충분하다. 실측 (fuji_d3r8/all, 홀드아웃 29 에피소드,
# 체크포인트 6개): 1분 51초 / RSS ~11GB.
# 다만 로그인 노드에서 직접 돌리면 OOM 으로 죽는다 (공용이라 여유 메모리가 없다).
# 반드시 sbatch 로 던질 것.
# 그래서 --gres 없이 던진다 — GPU 노드를 붙잡지 않으므로 backfill 이 잘 되고,
# critic 학습 잡과 GPU 를 두고 경쟁하지 않는다. GPU 로 돌리고 싶으면
#   sbatch --gres=gpu:1 --export=ALL,...,DEVICE=cuda ...
# 로 던지면 된다 (스크립트의 --device 로 그대로 넘어간다).
#
# 메모리: cogfeat.npy 를 통째로 텐서에 올린다. 체크포인트를 바꿀 때마다 다시 올리므로
# (사이에 del 로 놓아준다) 피크는 체크포인트 수와 무관하게 ~11GB 다.
# --cpus-per-task / --mem 은 이 클러스터가 거부한다 (제출 플러그인이 막는다) —
# 파티션 기본값 DefCpuPerGPU / DefMemPerGPU 가 자동 적용된다.

# 제출 플러그인이 MODEL_OUTPUT_DIR 를 요구하는데 머신마다 허용 접두사가 다르다.
# 이 잡은 거기에 아무것도 쓰지 않는다 (산출물은 전부 $REPO 아래). 플러그인 통과용이다.
if [ -z "${MODEL_OUTPUT_DIR:-}" ]; then
    for _p in /fsx/rlwrld-unified-checkpoints /rlwrld-unified-checkpoints; do
        if [ -d "$_p/$USER" ]; then MODEL_OUTPUT_DIR="$_p/$USER/rd-rl"; break; fi
    done
fi
export MODEL_OUTPUT_DIR

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
if [ ! -d "$REPO/configs/exp" ]; then
    echo "REPO 가 레포 루트가 아니다: $REPO"
    echo "  레포 루트에서 sbatch 를 던지거나 --export=ALL,REPO=<루트> 를 줄 것"
    exit 2
fi
cd "$REPO"

EXP=${EXP:-fuji_d3r8}
TAG=${TAG:-all}                       # success | all — critic 학습의 --tag 와 같다
HOLDOUT=${HOLDOUT:-0.1}               # critic 학습에 쓴 값과 **반드시** 같아야 한다.
                                      # 다르면 학습에 쓴 에피소드를 홀드아웃이라 부르게 된다.
SET=${SET:-holdout}                   # holdout | train | all
MAX_ROWS=${MAX_ROWS:-8}
DEVICE=${DEVICE:-cpu}                 # cuda 로 쓰려면 --gres=gpu:1 도 같이 줄 것
CAM_NAME=${CAM_NAME:-}                # 비우면 modality 의 첫 카메라 (camera_ego_left)
# 쉼표를 쓰는 값은 이 파일 안에 둔다 (sbatch --export 가 쉼표로 항목을 자른다).
STEPS=${STEPS_OVERRIDE:-1000,5000,20000,50000,100000,200000}

RUN=checkpoints/${EXP}-critic/${TAG}
[ -d "$RUN" ] || { echo "critic 런이 없다: $RUN  (먼저 critic.sbatch 를 돌릴 것)"; exit 2; }

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
export HF_HOME=${HF_HOME:-/fsx/rlwrld/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
PY=$REPO/third_party/RLDX-1/.venv/bin/python

echo "=== $(date -Is)  critic-grid  $EXP / $TAG"
echo "    steps=$STEPS  set=$SET  holdout=$HOLDOUT  rows=$MAX_ROWS  device=$DEVICE"
echo "=== 있는 체크포인트"; ls "$RUN" | grep '^critic_[0-9]' | sort

ARGS=(--exp "$EXP" --tag "$TAG" --critic-steps "$STEPS"
      --holdout "$HOLDOUT" --set "$SET" --max-rows "$MAX_ROWS"
      --device "$DEVICE" --success-only)
[ -n "$CAM_NAME" ] && ARGS+=(--cam-name "$CAM_NAME")

$PY -u utils/critic_grid.py "${ARGS[@]}" || exit 1

echo "=== 산출물"; ls -la "$RUN/plots/grid"
echo "=== $(date -Is)  완료"
