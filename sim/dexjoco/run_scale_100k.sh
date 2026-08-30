#!/bin/bash
# 데이터 양별 critic 학습: N = 100, 200, 500 을 각각 100k step (eval 1만 스텝마다).
#
#   nohup bash sim/dexjoco/run_scale_100k.sh > /tmp/scale100k.log 2>&1 &
#
# cogfeat 캐시 규칙 (extract_cogfeat 의 resume 한계 때문에 중요):
#   resume 은 cogfeat.json 의 done 만 보고 이어 쓴다. 그런데 npy 파일 크기는 키우지
#   못해서, N 이 커져 T 가 늘면 기존(작은) 파일에 쓰다 IndexError 로 죽는다.
#   따라서 N 을 올릴 때는 반드시 cogfeat.npy/json 을 지우고 새로 뽑아야 한다.
#   대신 N 별 결과를 cogfeat_n<N>.npy 로 남겨두면 되돌아갈 때 재추출이 필요 없다.
set -u

cd /rlwrld2/home/junmo_cho/ws/rd-rl
export PYTHONPATH="$PWD/third_party/RLDX-1:$PWD"
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
export HF_HOME=/workspace/junmo_cho/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=third_party/RLDX-1/.venv/bin/python
SIM=/workspace/junmo_cho/dexjoco/venv/bin/python
W=checkpoints/dexjoco_hammer_nail-critic

STEPS=100000
EVAL=10000

run_n() {
    local N=$1
    local CACHE=$W/cogfeat_n${N}.npy
    echo "################ N=$N  $(date -Is)"

    $SIM sim/dexjoco/make_subset.py --n "$N" || { echo "[FAIL] subset N=$N"; return 1; }

    # actnorm.npy 는 (T, H, A) 라 N 이 바뀌면 무효다. offline_iql 은 "파일이 있으면"
    # 다시 굽지 않고 넘어간 뒤(offline_iql.py:220) shape 불일치를 만나 재계산을 시도하는데,
    # 그 호출은 fn=None 이라 TypeError 로 죽는다 (N=200 첫 시도가 여기서 실패했다).
    # 데이터가 바뀌었으니 지우는 것이 맞다 — 굽는 데 몇 초면 된다.
    rm -f $W/actnorm.npy

    if [ -f "$CACHE" ]; then
        echo "==== cogfeat 캐시 재사용: $(basename $CACHE)"
        FEAT=$(basename $CACHE)
    else
        # 진행 중이던 추출이 있고 T 가 지금 N 과 맞으면 이어받고, 아니면 새로 시작한다.
        local T
        T=$($SIM -c "import json;print(json.load(open('rl-dataset/dexjoco-scale/meta/info.json'))['total_frames'])")
        if [ -f $W/cogfeat.json ] && [ "$($SIM -c "import json;print(json.load(open('$W/cogfeat.json'))['T'])" 2>/dev/null)" = "$T" ]; then
            echo "==== cogfeat 이어받기 (T=$T)"
        else
            echo "==== cogfeat 새로 추출 (T=$T)"
            rm -f $W/cogfeat.npy $W/cogfeat.json
        fi
        $PY -u -m rl.extract_cogfeat --exp dexjoco_hammer_nail \
            --data rl-dataset/dexjoco-scale --checkpoints checkpoints \
            --batch 64 --resume || { echo "[FAIL] cogfeat N=$N"; return 1; }
        cp $W/cogfeat.npy "$CACHE"
        FEAT=cogfeat.npy
    fi

    echo "==== IQL n${N}_100k  features=$FEAT  $(date -Is)"
    $PY -u -m rl.offline_iql \
        --exp dexjoco_hammer_nail \
        --data rl-dataset/dexjoco-scale \
        --checkpoints checkpoints \
        --features "$FEAT" \
        --tag "n${N}_100k" \
        --bins 128 --expectile 0.7 \
        --steps $STEPS --holdout 0.1 --eval-every $EVAL --video-eps 20 \
        --keep-last 10 || { echo "[FAIL] IQL N=$N"; return 1; }
    echo "################ N=$N DONE  $(date -Is)"
}

for N in ${SCALE_NS:-100 200 500}; do
    run_n "$N" || echo "################ N=$N 실패 — 다음으로 넘어감"
done

echo "######## SCALE100K ALL DONE $(date -Is)"
for t in n100_100k n200_100k n500_100k; do
    printf '%-14s 체크포인트 %s개\n' "$t" "$(ls $W/$t/*.pt 2>/dev/null | wc -l)"
    grep -a "eval\] step" /tmp/scale100k.log 2>/dev/null | tail -0
done
