#!/bin/bash
#SBATCH --job-name=dexjoco-hammer-nail-critic-qvgm-iql-eefpose-ablation-step-grid-checkpoints
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue   # preempt 되면 조용히 재시작하지 말고 죽어라
                       # (background 는 PreemptMode=REQUEUE 라 런이 섞인다)
#SBATCH --time=2:00:00   # 실측 19~20분, C 500K 는 ~55분. 요청 시간을 실측에 맞춰야
                       # Slurm backfill 이 빈 슬롯에 끼워 넣는다 (12h 를 걸면 대기가 길어진다)
#SBATCH --output=/rlwrld2/home/junmo_cho/ws/rd-rl/out/dexjoco-critic_%j.out
#SBATCH --error=/rlwrld2/home/junmo_cho/ws/rd-rl/out/dexjoco-critic_%j.err
#
# 학습은 ~150 step/s 라 200K 가 22분이다. --keep-steps 로 격자 스텝의 체크포인트를
# 지켜두면 스텝 ablation 이 한 번의 학습으로 전부 나온다 (따로 5번 돌릴 필요 없음).
#
#   sbatch --export=ALL,EXP=...,TRAIN_EPS=success sbatch/dexjoco/critic/critic_hammer_nail.sbatch
#
# ※ EXP 당 cogfeat.sbatch 를 먼저 한 번 돌린 뒤에 던질 것 (캐시 경쟁 방지).

: "${MODEL_OUTPUT_DIR:=/rlwrld-unified-checkpoints/junmo_cho/dexjoco-rl}"
export MODEL_OUTPUT_DIR

set -uo pipefail
REPO=/rlwrld2/home/junmo_cho/ws/rd-rl
cd "$REPO"

EXP=${EXP:-dexjoco_hammer_nail_d2r8_s0}
DATA=$(awk '/^dataset:/{print $2; exit}' configs/exp/$EXP.yaml)   # 데이터셋은 yaml 이 정한다
TRAIN_EPS=${TRAIN_EPS:-success}          # success | all | fail
STEPS=${STEPS:-200000}
# 격자를 하나로 합쳐 둔다 — STEPS 를 넘는 값은 그냥 안 생긴다. 그래서 인자가 필요 없다.
# (sbatch --export 는 쉼표로 항목을 자르므로 여기 쉼표를 넣는 건 이 파일 안에서만 안전하다)
KEEP_STEPS=1000,5000,20000,25000,50000,100000,200000,500000
KEEP_LAST=${KEEP_LAST:-3}
NUM_QS=${NUM_QS:-2}                      # 2 = clipped double Q (Q-VGM)
NUM_MIN_QS=${NUM_MIN_QS:-0}              # >0 이면 REDQ (예: NUM_QS=10 NUM_MIN_QS=2)
HOLDOUT=${HOLDOUT:-0.1}
TAG=${TAG:-${TRAIN_EPS}$([ "$NUM_MIN_QS" -gt 0 ] && echo _ens)}

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
export HF_HOME=/workspace/junmo_cho/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
PY=$REPO/third_party/RLDX-1/.venv/bin/python

echo "=== $(date -Is)  critic  $EXP / $TAG  steps=$STEPS keep=$KEEP_STEPS qs=$NUM_QS/$NUM_MIN_QS"

# cogfeat 이 이미 완성돼 있으면 건너뛴다. extract_cogfeat 는 캐시가 꽉 차 있어도
# VLA 백본을 통째로 올리느라 2분 가까이 쓰는데, 학습 잡 8개면 그게 다 낭비다.
NEED=$(python3 -c "import json;print(json.load(open('$DATA/meta/info.json'))['total_frames'])")
HAVE=$(python3 -c "import json,os;p='checkpoints/${EXP}-critic/cogfeat.json';print(json.load(open(p))['done'] if os.path.exists(p) else 0)")
if [ "$HAVE" = "$NEED" ]; then
    echo "[skip] cogfeat 이미 완료 ($HAVE 프레임)"
else
    echo "[cogfeat] $HAVE / $NEED 프레임 — 추출한다"
    $PY -u -m rl.extract_cogfeat --exp $EXP --data $DATA \
        --checkpoints checkpoints --batch 64 --resume || exit 1
fi

$PY -u -m rl.offline_iql_qvgm --exp $EXP --data $DATA \
    --checkpoints checkpoints --features cogfeat.npy \
    --action-groups eef_position,eef_rotation \
    --train-eps $TRAIN_EPS \
    --discount 0.995 --expectile 0.8 --num-qs $NUM_QS --num-min-qs $NUM_MIN_QS \
    --bins 128 --q-range 0,1 \
    --latent 2048 --state-latent 256 --hidden 1024,512 --no-stepwise \
    --steps $STEPS --holdout $HOLDOUT --eval-every 1000 \
    --keep-last $KEEP_LAST --keep-steps $KEEP_STEPS --no-wandb \
    --tag "$TAG" || exit 1

RUN=checkpoints/${EXP}-critic/${TAG}
echo "=== 남은 체크포인트"; ls "$RUN" | grep '^critic_[0-9]' | sort

$PY -u sim/dexjoco/critic_grid.py \
    --exp $EXP --data "$REPO/$DATA" --critic ${TAG}/critic_latest.pt \
    --holdout $HOLDOUT --set holdout --success-only --max-rows 8 \
    --out "$RUN/plots/holdout_success.mp4" || true
echo "=== $(date -Is)  완료"
