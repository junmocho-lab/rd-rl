#!/bin/bash
# 데이터 양에 따른 critic 품질 곡선: N = 100, 200, 300, 500, 1000 에피소드로 IQL 학습.
#
#   bash sim/dexjoco/run_scale.sh 100            # 하나만
#   bash sim/dexjoco/run_scale.sh 100 200 300    # 순서대로
#
# **반드시 오름차순으로 돌 것.** rl/data.py 의 build_images 와 rl/extract_cogfeat.py 는
# "기존 캐시가 새 에피소드 목록의 접두사일 때만" 이어붙인다 (rl/data.py:573). 큰 N 을
# 먼저 만들고 작은 N 을 돌리면 접두사 판정이 깨져 전체를 다시 만들 뿐 아니라
# f.truncate(nbytes) 가 images.mm 을 작은 크기로 잘라버린다.
# 오름차순이면 총 작업량이 N=1000 한 번과 같다 (cogfeat 약 2.6h, images.mm 106GB).
#
# 부분집합은 앞에서부터 N 개라 항상 이전 N 의 접두사가 된다 (make_subset.py).
set -u

cd /rlwrld2/home/junmo_cho/ws/rd-rl
export PYTHONPATH="$PWD/third_party/RLDX-1:$PWD"
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
export HF_HOME=/workspace/junmo_cho/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PY=third_party/RLDX-1/.venv/bin/python
SIM=/workspace/junmo_cho/dexjoco/venv/bin/python
EXP=dexjoco_hammer_nail
DATA=rl-dataset/dexjoco-scale
CKPT=checkpoints

# --keep-last 6: eval 주기(5000)마다 남는 체크포인트를 전부 보존한다. 기본값 1 이면
# 마지막 것만 남는데, N=100 실측에서 성공/실패 격차가 15k(0.315)에 정점을 찍고
# 30k(0.253)로 내려갔다 — 즉 N 마다 최적 스텝이 다르다. 마지막 체크포인트끼리 비교하면
# 데이터가 적은 쪽이 과적합 구간에서 평가되어 부당하게 불리해진다.
#
# 학습 예산은 N 과 무관하게 고정한다 — 보려는 것이 "같은 계산 예산에서 데이터가
# critic 품질을 얼마나 올리는가" 이기 때문. 데이터가 적을 때의 과적합은 holdout
# 지표(AUC)로 드러나므로 step 을 데이터에 비례시켜 가리지 않는다.
STEPS=${STEPS:-30000}

for N in "$@"; do
    echo "################ N=$N  $(date -Is)"

    $SIM sim/dexjoco/make_subset.py --n "$N" || exit 1

    echo "==== cogfeat (resume)"
    $PY -u -m rl.extract_cogfeat --exp $EXP --data $DATA --checkpoints $CKPT \
        --batch 64 --resume || exit 1

    echo "==== IQL n$N"
    $PY -u -m rl.offline_iql --exp $EXP --data $DATA --checkpoints $CKPT \
        --features cogfeat.npy --tag "n$N" \
        --bins 128 --expectile 0.7 \
        --steps "$STEPS" --holdout 0.1 --eval-every 5000 --video-eps 20 \
        --keep-last 6 \
        || exit 1

    echo "################ N=$N done $(date -Is)"
done
echo "SCALE ALL DONE"
