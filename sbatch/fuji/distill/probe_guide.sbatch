#!/bin/bash
#SBATCH --job-name=fuji-rby1m-rh56f1-probe-guide-move-sweep-dryrun
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --time=4:00:00
#SBATCH --output=out/fuji-probe-guide_%j.out
#SBATCH --error=out/fuji-probe-guide_%j.err
#
# relabel 의 guidance step size(--guide-move)를 **측정으로** 고른다.
# rl.relabel_parl --dry-run 은 아무것도 쓰지 않고 통계만 찍는데, 실제 relabel 과
# **같은 코드 경로**다. --limit 으로 결정 프레임 수를 줄여 여러 값을 빠르게 쓸어본다.
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   sbatch --export=ALL,EXP=fuji_d4r16,CTAG=success,STEP=10000,LIMIT=300 \
#          sbatch/fuji/distill/probe_guide.sbatch
#
# 판정 기준 (serve --guide-move / --ood-gate 도움말의 실측 근거):
#   · ΔQ / std >= 1            편집이 노이즈보다 큰가
#   · 앙상블 std 증가 < 2배     외삽으로 튀지 않았는가
#   · 로그된 액션이 이긴 비율 < 70%   critic 이 실제로 다른 액션을 원하는가
#   · A>0 비율                  advantage 가 양수인가 (실패 에피소드 쪽도 볼 것)
#
# --num-steps 는 쓸지 않는다: dexjoco sim/dexjoco/probe_steps.py 실측에서 steps
# 1/2/4/10/20 의 ΔQ 가 **소수 5자리까지 동일**했다 (이 범위에서 Q 가 선형이라 그래디언트
# 방향이 이동 중에 안 변한다). 실질적인 유일한 노브가 --guide-move 다.

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

EXP=${EXP:-fuji_d4r16}
CTAG=${CTAG:-success}
STEP=${STEP:-10000}
EPS=${EPS:-}                   # 비우면 CTAG 에 맞춘다 (success -> success, all -> all)
LIMIT=${LIMIT:-300}
NCAND=${NCAND:-32}
KEEP=${KEEP:-10}
GSTEPS=${GSTEPS:-4}
TEMP=${TEMP:-0.001}
# 쉼표를 쓰는 값은 이 파일 안에 둔다 (sbatch --export 가 쉼표로 항목을 자른다).
# 셸에서 export MOVES=... 한 뒤 --export=ALL 로 실어 보내면 덮을 수 있다.
MOVES=${MOVES:-0.005,0.01,0.02,0.05,0.1}
[ -n "$EPS" ] || EPS=$CTAG

DATA=$(awk '/^dataset:/{print $2; exit}' configs/exp/$EXP.yaml)
CRITIC=$REPO/checkpoints/${EXP}-critic/${CTAG}/$(printf 'critic_%06d.pt' "$STEP")
[ -f "$CRITIC" ] || { echo "critic 없음: $CRITIC"; exit 3; }

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
export HF_HOME=${HF_HOME:-/fsx/rlwrld/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
PY=$REPO/third_party/RLDX-1/.venv/bin/python

echo "=== $(date -Is)  probe guide_move  $EXP  critic=$CTAG@$STEP  eps=$EPS"
echo "    data=$DATA  limit=$LIMIT  후보 $NCAND -> keep $KEEP -> 상승 ${GSTEPS}스텝  temp=$TEMP"
echo "    쓸어볼 move: $MOVES"

for M in ${MOVES//,/ }; do
    echo
    echo "################ guide_move = $M ################"
    $PY -u -m rl.relabel_parl \
        --exp "$EXP" --data "$DATA" --checkpoints checkpoints \
        --critic "$CRITIC" --features cogfeat.npy \
        --eps "$EPS" \
        --num-samples "$NCAND" --num-keep "$KEEP" \
        --num-steps "$GSTEPS" --guide-move "$M" --temp "$TEMP" \
        --limit "$LIMIT" --dry-run || echo "  [실패] move $M"
done
echo
echo "=== $(date -Is)  완료"
