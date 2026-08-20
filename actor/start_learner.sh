#!/usr/bin/env bash
# learner 상시 잡을 **새 run id** 로 띄운다.
#
#   ./actor/start_learner.sh [실험이름]     # 기본: openarm_red_block
#
# run id = <실험이름>_<날짜-시각>. 매번 새 이름이므로 이전 run 의 DONE/FAILED 와 절대
# 섞이지 않는다 — 같은 라운드 번호로 다시 테스트해도 리셋된 상태에서 시작한다.
# run id 는 $A_RUNS/CURRENT 에 적어두고 send_round.py 가 기본값으로 읽는다.

set -euo pipefail
cd "$(dirname "$0")/.."
source configs/paths.sh

NAME="${1:-openarm_red_block}"
RUN_ID="${NAME}_$(date +%Y%m%d-%H%M)"
# k8s 리소스 이름은 DNS-1123 (소문자·숫자·'-') 이라 밑줄을 '-' 로 바꾼다
JOB_NAME="junmo-cho-rdrl-$(printf '%s' "$RUN_ID" | tr 'A-Z_' 'a-z-')"

if [ ${#JOB_NAME} -gt 63 ]; then
    echo "error: Job 이름이 63자를 넘는다 (${#JOB_NAME}자): $JOB_NAME" >&2
    echo "  실험 이름을 줄일 것" >&2
    exit 1
fi

# 이미 도는 learner 가 있으면 알려준다 (같은 메일박스를 두 개가 보면 안 된다)
RUNNING=$(kubectl -n "$L_NS" get jobs -o name 2>/dev/null | grep 'junmo-cho-rdrl-' || true)
if [ -n "$RUNNING" ]; then
    echo "[warn] 이미 있는 rdrl 잡:"
    echo "$RUNNING" | sed 's/^/  /'
    echo "  run id 가 다르면 메일박스도 달라서 충돌하지 않지만, 필요 없으면 지울 것"
fi

sed -e "s|__RUN_ID__|$RUN_ID|g" -e "s|__JOB_NAME__|$JOB_NAME|g" k8s/learner.yaml \
    | kubectl apply -f -

mkdir -p "$A_RUNS"
printf '%s\n' "$RUN_ID" > "$A_RUNS/CURRENT"

cat <<EOS

run id : $RUN_ID   (→ $A_RUNS/CURRENT)
job    : $JOB_NAME

  로그   kubectl -n $L_NS logs -f job/$JOB_NAME
         kubectl -n $L_NS exec $L_POD -- tail -20 $L_RUNS/$RUN_ID/learner.log
  라운드 .venv/bin/python actor/send_round.py --round 0 --dataset <세션경로>
  중단   kubectl -n $L_NS delete job $JOB_NAME
EOS
