#!/bin/bash
# Test-time Q guidance 평가 — 한 팔(arm)을 돌린다.
#
#   bash sim/dexjoco/run_guidance.sh <arm> [EPISODES]
#
# 팔:
#   bc      기준선. 순정 RLDX 서버 (후보 1개, critic 미사용)
#   sel     ExpoServer, base 후보 N=8 → target critic argmax (--guide-steps 0)
#   g005    sel + ∇_A Q 상승 4스텝, 차원당 이동 0.05 (통로: yaml 의 explore_groups)
#   g010    "                        0.10
#   g020    "                        0.20
#   g020a   g020 과 같되 **전 그룹**을 민다 (--guide-groups all)
#
# g020 vs g020a 가 가르는 것: guidance 의 통로 폭. 기본 explore_groups=[eef_position] 은
# 청크 625차원(25스텝 x 25관절) 중 60개뿐이라, Q 의 액션 민감도가 회전(rot6d)이나
# 손가락에 있으면 밀지를 못한다. all 이면 실행 구간 전 관절 500차원을 민다.
# (후보 **선택**은 두 팔 모두 청크 전 차원을 쓴다 — 마스크는 상승에만 걸린다.)
#
# **모든 팔이 같은 장면을 본다.** rollout_dexjoco.py --seed-per-episode 가 에피소드
# 인덱스로 전역 RNG 를 시드하므로 ep i 의 테이블 높이/망치 xy·yaw/못 xy 가 팔마다 같다.
# 따라서 두 비율 검정이 아니라 **McNemar(짝지어 비교)** 로 읽어야 하고, 그만큼 적은
# 표본으로 차이를 본다. (기존 1000개 수집은 ep500 에서 재시작하는 바람에 ep k 와
# ep 500+k 가 같은 장면이었다 — 그 문제도 이 플래그가 막는다.)
#
# critic: n1000_300k/critic_180000.pt — 홀드아웃 AUC 0.902 로 최고점.
set -u

ARM=${1:?arm: bc|sel|g005|g010|g020}
EPISODES=${2:-200}

REPO=/rlwrld2/home/junmo_cho/ws/rd-rl
RUN=/workspace/junmo_cho/dexjoco/run/guidance
OUT=/workspace/junmo_cho/dexjoco/rollout/guidance/$ARM
CRITIC=$REPO/checkpoints/dexjoco_hammer_nail-critic/n1000_300k/critic_180000.pt

# 롤아웃 실행 규약. configs/exp/dexjoco_hammer_nail.yaml 과 반드시 같아야 한다.
RTC_DELAY=5
REPLAN=20
MAX_STEPS=360

case $ARM in
    bc)   PORT=20301; GUIDE_STEPS=0; GUIDE_MOVE=0 ;;
    sel)  PORT=20302; GUIDE_STEPS=0; GUIDE_MOVE=0 ;;
    g005) PORT=20303; GUIDE_STEPS=4; GUIDE_MOVE=0.05 ;;
    g010) PORT=20304; GUIDE_STEPS=4; GUIDE_MOVE=0.10 ;;
    g020) PORT=20305; GUIDE_STEPS=4; GUIDE_MOVE=0.20 ;;
    g020a) PORT=20306; GUIDE_STEPS=4; GUIDE_MOVE=0.20; GUIDE_GROUPS=all ;;
    *) echo "[FATAL] 모르는 arm: $ARM"; exit 1 ;;
esac
GUIDE_GROUPS=${GUIDE_GROUPS:-}

export HF_HOME=/workspace/junmo_cho/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 토큰이 로테이트돼 무효 — 캐시만 쓴다
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1

CKPT=$REPO/$(cd $REPO &&  ls -d checkpoints/dexjoco/dexjoco_hammer_nail_*/ | head -1)
mkdir -p "$RUN" "$OUT"
LOG=$RUN/server_$ARM.log

echo "=== $(date -Is) host=$(hostname) arm=$ARM"
echo "    ckpt   $CKPT"
echo "    critic $CRITIC"
echo "    out    $OUT   port=$PORT  guide steps=$GUIDE_STEPS move=$GUIDE_MOVE"

# ── 정책 서버 ──────────────────────────────────────────────────────────────────
if ! (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
    echo "=== 서버 기동 ($ARM)"
    rm -f "$LOG"
    if [ "$ARM" = bc ]; then
        (
            cd "$REPO/third_party/RLDX-1" && source .venv/bin/activate
            exec python rldx/eval/run_rldx_server.py \
                --model-path "$CKPT" \
                --embodiment-tag GENERAL_EMBODIMENT --use-sim-policy-wrapper \
                --rtc-inference-mode trained --rtc-inference-delay $RTC_DELAY \
                --rtc-inference-exec-horizon $REPLAN \
                --host 127.0.0.1 --port $PORT
        ) > "$LOG" 2>&1 &
        READY="Server is ready"
    else
        (
            cd "$REPO"
            export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
            exec third_party/RLDX-1/.venv/bin/python -u -m rl.vla_rldx serve \
                --exp dexjoco_hammer_nail \
                --model-path "$CKPT" \
                --artifacts "$CRITIC" \
                --rtc-inference-mode trained \
                --rtc-exec-horizon $REPLAN \
                --guide-steps $GUIDE_STEPS --guide-move $GUIDE_MOVE \
                ${GUIDE_GROUPS:+--guide-groups $GUIDE_GROUPS} \
                --sim-wrapper --log-every 50 \
                --host 127.0.0.1 --port $PORT
        ) > "$LOG" 2>&1 &
        READY="듣는다 tcp"
    fi
    for _ in $(seq 1 180); do
        grep -qa "$READY" "$LOG" && break
        sleep 10
    done
    grep -qa "$READY" "$LOG" || { echo "[FATAL] 서버 기동 실패"; tail -40 "$LOG"; exit 1; }
    grep -a "RTC enabled\|\[선택\]\|\[guidance\]\|\[편집\]\|\[진단 기준\]" "$LOG" | tail -8
else
    echo "=== 서버 이미 떠 있음 (port $PORT)"
fi

# ── 롤아웃 ────────────────────────────────────────────────────────────────────
cd "$REPO"
PYTHONPATH=third_party/dexjoco/dexjoco MUJOCO_GL=egl \
    /workspace/junmo_cho/dexjoco/venv/bin/python -u sim/dexjoco/rollout_dexjoco.py \
    --task hammer_nail --episodes "$EPISODES" --port $PORT \
    --replan $REPLAN --rtc-delay $RTC_DELAY \
    --max-episode-steps $MAX_STEPS --log-every 25 --resume --output "$OUT"
rc=$?
echo "=== $(date -Is) arm=$ARM rollout exit $rc"
exit $rc
