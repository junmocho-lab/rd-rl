#!/bin/bash
#SBATCH --job-name=fuji-rby1m-rh56f1-distill-relabel-then-lora
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --no-requeue
#SBATCH --time=48:00:00
#SBATCH --output=out/fuji-distill-chain_%j.out
#SBATCH --error=out/fuji-distill-chain_%j.err
#
# relabel + LoRA 를 **한 잡 안에서** 순서대로 돌린다.
#
# ── 왜 afterok 두 잡이 아니라 한 잡인가 ─────────────────────────────────────
# afterok 로 엮으면 relabel 이 끝나는 순간 Slurm 이 GPU 를 **반납한다.** 뒤따르는 LoRA
# 잡은 그 시점에 다시 자원 경쟁을 하므로, 공용 클러스터에서는 relabel 하는 35분 사이에
# 남이 GPU 를 채워 버리면 LoRA 가 몇 시간을 기다린다 (실제로 잡 1093 이 Reason=Priority
# 로 밀렸다 — 앞의 4-GPU 잡이 backfill 창을 막고 있었다).
# 한 잡으로 묶으면 할당받은 GPU 2장을 relabel 부터 LoRA 끝까지 **놓지 않는다.**
# 대가: relabel 단계는 1장만 쓰므로 ~35분 동안 1장이 유휴다. 기다림 위험을 그 값에
# 사는 것이다.
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   sbatch --export=ALL,SSIZE=1 sbatch/fuji/distill/relabel_then_lora.sbatch
#   sbatch --export=ALL,SSIZE=3,STEPS=16000 sbatch/fuji/distill/relabel_then_lora.sbatch
#
# 이미 relabel 이 끝난 데이터가 있으면 SKIP_RELABEL=1 로 LoRA 만 돌린다.

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

export EXP=${EXP:-fuji_d4r16_0902_fuji_rc_success}
export CTAG=${CTAG:-success_ens_full}
export STEP=${STEP:-25000}
export SSIZE=${SSIZE:?SSIZE 를 줘야 한다 (gradient ascent step_size, 예: 1 또는 3)}
export GSTEPS=${GSTEPS:-10}
export EPS=${EPS:-success}
export BATCH=${BATCH:-128}          # 전역 배치. GPU 2장이면 장당 64 (잡 792 OOM 하한 아래)
export STEPS=${STEPS:-30000}
export SAVE=${SAVE:-2000}
SKIP_RELABEL=${SKIP_RELABEL:-0}

# relabel.sbatch 가 만드는 출력 경로를 **같은 규칙으로** 다시 만든다.
#   relabel.sbatch:95  SUF="${CTAG}@$((STEP/1000))k_${EPS}_ss${SSIZE}n${GSTEPS}"
# 규칙이 갈라지면 LoRA 가 없는 경로를 보고 죽으므로, 저기를 바꾸면 여기도 바꿀 것.
DSDIR=$(dirname "$(awk '/^dataset:/{print $2; exit}' configs/exp/$EXP.yaml)")
SUF="${CTAG}@$((STEP/1000))k_${EPS}_ss${SSIZE}n${GSTEPS}"
export DATA=$DSDIR/relabel_${EXP}__${SUF}

NGPU=$(nvidia-smi -L | wc -l)
echo "╔══════════════════════════════════════════════════════════════════════════"
echo "║ $(date -Is)  distill chain  job ${SLURM_JOB_ID:-local}  GPU $NGPU 장"
echo "║   exp    $EXP"
echo "║   critic $CTAG@$STEP   step_size=$SSIZE  num_steps=$GSTEPS  eps=$EPS"
echo "║   relabel -> $DATA"
echo "║   LoRA    steps=$STEPS  batch=$BATCH (장당 $((BATCH/NGPU)))  save=$SAVE"
echo "╚══════════════════════════════════════════════════════════════════════════"
[ "$NGPU" -lt 2 ] && echo "★ 경고: GPU $NGPU 장이면 장당 $((BATCH/NGPU)) — 잡 792 에서 장당 128 이 OOM 이었다"

# ── 1단계: relabel (GPU 1장만 쓴다) ─────────────────────────────────────────
if [ "$SKIP_RELABEL" = 1 ] && [ -d "$REPO/$DATA" ]; then
    echo; echo "### 1/2 relabel 건너뜀 (SKIP_RELABEL=1, 이미 있다): $DATA"
else
    echo; echo "### 1/2 relabel  $(date -Is)"
    bash sbatch/fuji/distill/relabel.sbatch || {
        echo "!!! relabel 실패 — LoRA 로 넘어가지 않는다"; exit 1; }
    [ -d "$REPO/$DATA" ] || {
        echo "!!! relabel 이 끝났는데 출력이 없다: $REPO/$DATA"
        echo "    relabel.sbatch 의 SUF 규칙이 이 스크립트와 갈라졌을 수 있다"; exit 1; }
fi

# ── 2단계: LoRA (GPU 전부) ──────────────────────────────────────────────────
echo; echo "### 2/2 LoRA  $(date -Is)"
bash sbatch/fuji/distill/lora_train.sbatch || { echo "!!! LoRA 실패"; exit 1; }

echo; echo "=== $(date -Is)  체인 완료"
