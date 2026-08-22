#!/usr/bin/env bash
# learner 를 **data pod 안에서 직접** 띄운다 (k8s Job 제출 없이).
#
#   ./learner/start.sh [실험이름] [기존 run id]      # 기본 실험: fuji, GPU 4장
#   RDRL_GPUS=0,1 ./learner/start.sh fuji           # 2장만 쓰기
#   RDRL_GPUS=2   ./learner/start.sh fuji           # 1장 (torchrun 없이 단일 프로세스)
#   RDRL_FG=1     ./learner/start.sh fuji           # tmux 없이 포그라운드
#
# k8s/learner.yaml 과 같은 일을 한다. 다른 점만:
#   - Job 이 아니라 tmux 세션에서 돈다. 파드가 살아있는 한 사니까 activeDeadlineSeconds
#     (48h) 도, backoffLimit 재시도도 없다. 죽으면 그냥 죽는다 — 로그를 보고 다시 띄운다.
#   - GPU 여러 장을 torchrun 으로 쓴다 (yaml 은 nvidia.com/gpu: 1 이었다). rank 하나가
#     GPU 하나를 잡고, gradient 를 rank 평균으로 맞춘다 (rl/ddp.py). expo.batch_size 는
#     **rank 당** 값이라 실효 배치가 GPU 수만큼 커진다 — 로그와 meta.json 에 남는다.
#   - WANDB_API_KEY 는 secret 이 아니라 파드 env 에 이미 있다.
#   - `kubectl logs` 가 없다. loop.py 가 쓰는 $L_RUNS/<run id>.learner.log 는 그대로고,
#     거기 안 들어가는 것(파이썬 traceback, torch/HF 경고)은 .console.log 에 받는다.
#
# 두 번째 인자로 기존 run id 를 주면 그 실험을 이어받는다 (메일박스·버퍼·init/·지난
# 라운드 산출물이 전부 run id 아래 있다 — 코드만 고쳐 다시 띄울 때 새 id 를 만들면 안 된다).

set -euo pipefail
cd "$(dirname "$0")/.."
source configs/paths.sh

# tmux 안에서 자기 자신을 다시 부른 경우 — 여기가 실제 실행부다 (yaml 의 args 블록에 해당).
if [ "${1:-}" = "__run" ]; then
    RUN_ID="$2"; NAME="$3"; NGPU="$4"

    echo "[paths] L_RUNS=$L_RUNS"
    echo "[paths] L_CKPT=$L_CKPT"
    echo "[paths] L_DS=$L_DS"
    echo "[python] $L_PY $("$L_PY" -V 2>&1)"
    echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(전부)}  프로세스 $NGPU개"

    # rldx 를 **우리가 pin 한 서브모듈**에서 import 하게 만든다. 없으면 venv 에 설치된
    # 다른 체크아웃이 잡혀서, 기록에는 우리 SHA 가 남는데 실제로는 다른 코드가 돈다.
    export PYTHONPATH="$L_PYTHONPATH"
    export NO_ALBUMENTATIONS_UPDATE=1
    echo "[pythonpath] $PYTHONPATH"

    # 백본 forward 가 rank 마다 스레드를 다 잡으면 128코어를 4개가 나눠 갖다 서로 민다.
    export OMP_NUM_THREADS=$((16 / NGPU > 0 ? 16 / NGPU : 1))

    ARGS=(learner/loop.py
        --exp "$RUN_ID"
        --exp-config "configs/exp/$NAME.yaml"
        --runs-root "$L_RUNS"
        --ckpt-root "$L_CKPT"
        --repo "$L_RL"
        --poll-seconds 5
        --heartbeat-seconds 300)

    if [ "$NGPU" -gt 1 ]; then
        # --standalone: rendezvous 를 이 노드 안에서만 한다 (단일 노드 4프로세스).
        exec "$L_PY" -m torch.distributed.run \
            --standalone --nnodes=1 --nproc_per_node="$NGPU" "${ARGS[@]}"
    fi
    exec "$L_PY" "${ARGS[@]}"
fi

NAME="${1:-fuji}"
RUN_ID="${2:-${NAME}_$(date +%Y%m%d-%H%M%S)}"
# 기본은 이 머신에 보이는 GPU 전부.
GPUS="${RDRL_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
NGPU=$(printf '%s' "$GPUS" | tr ',' '\n' | grep -c .)

if [ ! -f "configs/exp/$NAME.yaml" ]; then
    echo "error: configs/exp/$NAME.yaml 이 없다. 있는 것:" >&2
    ls configs/exp/*.yaml | sed 's/^/  /' >&2
    exit 1
fi
if [ -n "${2:-}" ]; then
    echo "[이어받기] run id $RUN_ID — 기존 메일박스/버퍼/산출물을 그대로 쓴다"
    echo "           FAILED 로 남은 라운드가 있으면 그 파일을 지워야 다시 처리한다"
fi

# 같은 메일박스를 두 프로세스가 보면 안 된다 (Job 시절 start_learner.sh 가 하던 경고).
RUNNING=$(pgrep -af "learner/loop\.py" || true)
if [ -n "$RUNNING" ]; then
    echo "[warn] 이미 도는 learner:"
    echo "$RUNNING" | sed 's/^/  /'
    echo "  run id 가 다르면 메일박스도 달라서 충돌하지 않지만, 필요 없으면 죽일 것"
fi

SESSION="rdrl-$(printf '%s' "$RUN_ID" | tr 'A-Z_' 'a-z-')"
CONSOLE="$L_RUNS/$RUN_ID.console.log"
mkdir -p "$L_RUNS"
# actor 쪽 runs/CURRENT 와 짝이 되는 학습서버 쪽 사본. 여기서 run id 를 잊지 않으려고 둔다
# (actor 는 자기 $A_RUNS/CURRENT 를 읽으므로 아래 안내대로 거기에도 적어야 한다).
printf '%s\n' "$RUN_ID" > "$L_RUNS/CURRENT"

if [ -n "${RDRL_FG:-}" ]; then
    CUDA_VISIBLE_DEVICES="$GPUS" exec "$0" __run "$RUN_ID" "$NAME" "$NGPU"
fi

CUDA_VISIBLE_DEVICES="$GPUS" \
    tmux new-session -d -s "$SESSION" \
    "\"$L_RL/learner/start.sh\" __run \"$RUN_ID\" \"$NAME\" \"$NGPU\" 2>&1 | tee -a \"$CONSOLE\""

sleep 2
tmux has-session -t "$SESSION" 2>/dev/null || {
    echo "error: tmux 세션이 바로 죽었다 — $CONSOLE 를 볼 것" >&2
    tail -30 "$CONSOLE" >&2 || true
    exit 1
}

cat <<EOS

run id  : $RUN_ID   (→ $L_RUNS/CURRENT)
tmux    : $SESSION   (GPU $GPUS — $NGPU 프로세스)

  로그    tmux attach -t $SESSION        (빠져나올 때 Ctrl-b d)
          tail -f $L_RUNS/$RUN_ID.learner.log
          tail -f $CONSOLE               (traceback·경고까지)
  중단    tmux send-keys -t $SESSION C-c  ← SIGINT, 현재 라운드까지 마치고 종료
          tmux kill-session -t $SESSION   ← 즉시

  actor 쪽에서 (run id 를 알려줘야 한다):
    echo $RUN_ID > \$A_RUNS/CURRENT
    .venv/bin/python actor/recv_round.py --round init
    .venv/bin/python actor/send_round.py --round 0 --dataset <세션경로>
EOS
