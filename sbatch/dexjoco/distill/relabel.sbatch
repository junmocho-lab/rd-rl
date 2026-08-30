#!/bin/bash
#SBATCH --job-name=dexjoco-hammer-nail-relabel-parl-action-optimization-for-distillation
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue   # preempt 되면 조용히 재시작하지 말고 죽어라
#SBATCH --time=6:00:00   # 결정 7300 x 후보 32. 요청 시간을 실측에 맞춰야
                       # Slurm backfill 이 빈 슬롯에 끼워 넣는다 (12h 를 걸면 대기가 길어진다)
#SBATCH --output=out/dexjoco-relabel_%j.out
#SBATCH --error=out/dexjoco-relabel_%j.err
#
# PA-RL 액션 최적화로 데이터셋의 action 컬럼을 갈아끼운다 (action distillation 1단계).
# 서빙(rl/vla_rldx.py)이 89% 를 낸 절차와 **같은 설정**이 기본값이다:
#   후보 32 -> Q(min) 상위 9 + 로그액션 1 -> ∇_A Q 상승 4스텝(keep-best) -> Categorical(Q/0.001)
#   선택/채택은 앙상블 min, 상승 방향만 mean.
#
#   sbatch --export=ALL,...,DRY=1,LIMIT=200 ...    # 통계만 (먼저 이걸 볼 것)
#   sbatch --export=ALL,...,CTAG=success,EPS=success ...
#
# 판정 기준: 로그 액션이 이긴 비율 < 70%. 넘으면 critic 이 base policy 를 못 이겨
# distillation 이 no-op 이 된다 (PARL-DISTILL.md 5절).

: "${MODEL_OUTPUT_DIR:=/rlwrld-unified-checkpoints/junmo_cho/dexjoco-rl}"
export MODEL_OUTPUT_DIR

set -uo pipefail
# ── 머신에 따라 달라지는 것은 전부 환경변수로 덮을 수 있다 ────────────────────
# REPO 는 sbatch 를 던진 디렉토리(= 레포 루트)에서 온다. 다른 머신에 레포가 다른
# 경로로 있어도 그대로 돌아간다. 명시하고 싶으면 --export=ALL,REPO=... 로 준다.
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
if [ ! -d "$REPO/configs/exp" ]; then
    echo "REPO 가 레포 루트가 아니다: $REPO"
    echo "  레포 루트에서 sbatch 를 던지거나 --export=ALL,REPO=<루트> 를 줄 것"
    exit 2
fi
cd "$REPO"

EXP=${EXP:-dexjoco_hammer_nail_d2r8_s0}
DATA=$(awk '/^dataset:/{print $2; exit}' configs/exp/$EXP.yaml)
CTAG=${CTAG:-success}          # critic 학습 태그
STEP=${STEP:-20000}
EPS=${EPS:-all}                # relabel 할 에피소드 (critic 학습 데이터와 맞출 것)
NCAND=${NCAND:-32}
KEEP=${KEEP:-10}
GSTEPS=${GSTEPS:-4}
GMOVE=${GMOVE:-0.05}
TEMP=${TEMP:-0.001}
LIMIT=${LIMIT:-0}
DRY=${DRY:-0}

CRITIC=$REPO/checkpoints/${EXP}-critic/${CTAG}/$(printf 'critic_%06d.pt' $STEP)
[ -f "$CRITIC" ] || { echo "critic 없음: $CRITIC"; exit 3; }
OUT=$REPO/rl-dataset/dexjoco/relabel_${EXP}__${CTAG}@$((STEP/1000))k${EPS:+_$EPS}

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
export HF_HOME=${HF_HOME:-/workspace/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1

echo "=== $(date -Is)  relabel  $EXP  critic=$CTAG@$STEP  eps=$EPS"
echo "    후보 $NCAND -> keep $KEEP -> 상승 $GSTEPS x move $GMOVE -> temp $TEMP"
echo "    out $OUT"

third_party/RLDX-1/.venv/bin/python -u -m rl.relabel_parl \
    --exp $EXP --data $DATA --checkpoints checkpoints \
    --critic "$CRITIC" --features cogfeat.npy \
    --eps $EPS \
    --num-samples $NCAND --num-keep $KEEP \
    --num-steps $GSTEPS --guide-move $GMOVE --temp $TEMP \
    --out "$OUT" \
    $([ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT") \
    $([ "$DRY" = 1 ] && echo "--dry-run")
echo "=== $(date -Is)  완료"
