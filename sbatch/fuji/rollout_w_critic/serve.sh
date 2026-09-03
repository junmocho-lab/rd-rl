#!/bin/bash
# fuji (rby1m_rh56f1) 정책 서버 — BC 에 critic 을 붙여 청크를 고르거나 밀어준다.
# rrc 의 ZMQ 클라이언트(rrc/inference/zmq_client.py, RldxCodec)가 여기에 붙는다.
#
# dexjoco 판(sbatch/dexjoco/rollout_w_critic/eval.sbatch)과 갈라지는 지점:
#   dexjoco 는 같은 잡 안에서 시뮬레이터 롤아웃까지 돌린다. **fuji 는 실기라
#   롤아웃 하네스가 없다** (chain.sh 주석 참조) — 이 스크립트는 서버만 띄우고,
#   에피소드 진행과 성공/실패 라벨은 rrc 쪽에서 사람이 한다. 그래서 sbatch 가
#   아니라 실행 스크립트다. 클러스터에서 띄우려면 serve.sbatch 를 볼 것.
#
#   METHOD : bc | sel32 | parl | parl_sample
#   CTAG   : critic 학습 태그 (success | all)
#   STEP   : critic 스텝 (20000 / 50000 / 100000 / 200000). 0 이면 critic_latest.pt
#
#   bash sbatch/fuji/rollout_w_critic/serve.sh                       # 기본값
#   METHOD=parl CTAG=all STEP=100000 bash sbatch/fuji/rollout_w_critic/serve.sh
#   METHOD=parl_sample CTAG=success STEP=20000 PARL_TEMP=0.001 bash .../serve.sh
#
# 여러 critic 을 쓸어보려면 (한 번에 하나씩 — 로봇이 한 대다):
#   for c in success all; do for s in 20000 50000 100000 200000; do
#       METHOD=parl CTAG=$c STEP=$s bash sbatch/fuji/rollout_w_critic/serve.sh
#   done; done

set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
[ -d "$REPO/configs/exp" ] || { echo "REPO 가 레포 루트가 아니다: $REPO"; exit 2; }
cd "$REPO"

EXP=${EXP:-fuji_d3r8}
METHOD=${METHOD:-parl}
CTAG=${CTAG:-all}
STEP=${STEP:-200000}
HOST=${HOST:-127.0.0.1}          # rrc 가 다른 머신이면 0.0.0.0 으로 열 것
PORT=${PORT:-5555}               # rrc zmq_client 기본 포트
DEVICE=${DEVICE:-cuda}
LOG_EVERY=${LOG_EVERY:-25}

YAML=configs/exp/$EXP.yaml
[ -f "$YAML" ] || { echo "exp yaml 이 없다: $YAML"; exit 2; }
REPLAN=$(awk '/^replan_steps:/{print $2; exit}' "$YAML")
LATENCY=$(awk '/^inference_latency:/{print $2; exit}' "$YAML")

# ── 방법 프리셋 ──────────────────────────────────────────────────────────────
# parl = 후보 N -> Q 로 top-M -> 그 M 개를 ∇_A Q 로 상승 -> 다시 채점 -> 선택.
# dexjoco 실측(고정씬 100ep): BC 61.5% / sel32 79.0% / parl 89.0%.
# 그 값은 dexjoco 것이고 fuji 로 전이된다는 근거는 없다 — 실기로 재야 한다.
case $METHOD in
  bc)          N_CAND=1  ; PARL_KEEP=0  ; GUIDE_STEPS=0 ; GUIDE_MOVE=0    ; TEMP=0 ;;
  sel32)       N_CAND=32 ; PARL_KEEP=0  ; GUIDE_STEPS=0 ; GUIDE_MOVE=0    ; TEMP=0 ;;
  parl)        N_CAND=32 ; PARL_KEEP=10 ; GUIDE_STEPS=4 ; GUIDE_MOVE=0.05 ; TEMP=0 ;;
  # 최종 선택을 argmax 대신 Categorical(Q/temp) 로 뽑는다. 우리 Q 는 support [0,1]
  # 이라 후보 간 산포가 0.001 수준이다 — 온도를 안 나누면 사실상 균등분포가 된다.
  # 로그의 '샘플엔트로피' 를 보고 조절할 것 (0=결정적, ln M=균등).
  parl_sample) N_CAND=32 ; PARL_KEEP=10 ; GUIDE_STEPS=4 ; GUIDE_MOVE=0.05 ; TEMP=0.001 ;;
  *) echo "METHOD 는 bc | sel32 | parl | parl_sample"; exit 2 ;;
esac
N_CAND=${N_CAND_OVERRIDE:-$N_CAND}
PARL_KEEP=${PARL_KEEP_OVERRIDE:-$PARL_KEEP}
GUIDE_STEPS=${GUIDE_STEPS_OVERRIDE:-$GUIDE_STEPS}
GUIDE_MOVE=${GUIDE_MOVE_OVERRIDE:-$GUIDE_MOVE}
# raw-gradient guidance (relabel_parl 의 새 방식). >0 이면 GUIDE_MOVE 대신 이걸 쓴다.
GUIDE_STEP_SIZE=${GUIDE_STEP_SIZE:-0}
PARL_TEMP=${PARL_TEMP:-$TEMP}
OOD_GATE=${OOD_GATE:-0}          # 2 정도가 합리적 출발점 (serve --ood-gate 도움말)

CKPT=${MODEL_PATH:-$REPO/checkpoints/$(awk '/^base_policy:/{print $2; exit}' "$YAML")}
[ -d "$CKPT" ] || { echo "정책 체크포인트 없음: $CKPT"; exit 3; }

if [ "$METHOD" = bc ]; then
  CRITIC=""
  NAME=${NAME:-bc}
elif [ -n "${CRITIC_PATH:-}" ]; then
  # 임의 위치의 critic .pt 를 직접 지정 (CTAG/STEP 무시). 단 critic 은 여전히
  # EXP yaml 의 latency/replan 과 맞아야 한다 — 로드 시 교차검증에 걸린다.
  CRITIC=$CRITIC_PATH
  [ -f "$CRITIC" ] || { echo "critic 없음: $CRITIC"; exit 3; }
  NAME=${NAME:-${METHOD}__$(basename "$(dirname "$CRITIC")")_$(basename "$CRITIC" .pt)}
else
  if [ "$STEP" -eq 0 ]; then CF=critic_latest.pt; else CF=$(printf 'critic_%06d.pt' "$STEP"); fi
  CRITIC=$REPO/checkpoints/${EXP}-critic/${CTAG}/${CF}
  [ -f "$CRITIC" ] || { echo "critic 없음: $CRITIC"; echo "있는 것:"; \
                        ls "$REPO/checkpoints/${EXP}-critic/${CTAG}" 2>/dev/null | grep '^critic_'; exit 3; }
  NAME=${NAME:-${METHOD}__${CTAG}@$((STEP/1000))k}
fi

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
# HF 캐시: 클러스터(fsx)가 있으면 그것, 없으면(로컬 워크스테이션) ~/.cache/huggingface.
# 안 맞으면 백본(RLWRLD/RLDX-1-VLM) config 로드에서 "couldn't connect to hf.co" 로 죽는다.
if [ -z "${HF_HOME:-}" ]; then
  HF_HOME=/fsx/rlwrld/junmo_cho/hf_cache
  [ -d "$HF_HOME" ] || HF_HOME=$HOME/.cache/huggingface
fi
export HF_HOME
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
# 파이썬: pixi 환경이 있으면 우선 (RTX 5090 은 .venv 의 torch cu126 으로 CUDA 가 안 돈다).
if [ -z "${PY:-}" ]; then
  PY=$REPO/third_party/RLDX-1/.pixi/envs/rldx/bin/python
  [ -x "$PY" ] || PY=$REPO/third_party/RLDX-1/.venv/bin/python
fi
[ -x "$PY" ] || { echo "python 이 없다: $PY"; exit 3; }

# 무엇으로 서빙했는지 로그에 남긴다 (나중에 라벨과 대조할 때 쓴다).
# guide 표기: raw 모드(GUIDE_STEP_SIZE>0)면 스텝수 x step_size (raw), 아니면 스텝수 x 총이동량
if awk "BEGIN{exit !($GUIDE_STEP_SIZE > 0)}"; then
  GUIDE_SHOWN="${GUIDE_STEPS}x${GUIDE_STEP_SIZE}(raw)"
else
  GUIDE_SHOWN="${GUIDE_STEPS}x${GUIDE_MOVE}(move)"
fi
export RD_SERVE_INFO="{\"exp\":\"$EXP\",\"method\":\"$METHOD\",\"critic\":\"$CRITIC\",\"ctag\":\"$CTAG\",\"step\":$STEP,\"n_cand\":$N_CAND,\"parl_keep\":$PARL_KEEP,\"parl_temp\":$PARL_TEMP,\"guide_steps\":$GUIDE_STEPS,\"guide_move\":$GUIDE_MOVE,\"guide_step_size\":$GUIDE_STEP_SIZE,\"guide_all\":${GUIDE_ALL:-0},\"ood_gate\":$OOD_GATE,\"base_policy\":\"$CKPT\"}"

echo "=== $(date -Is)  fuji serve  $EXP / $NAME"
echo "    base   $CKPT"
echo "    critic ${CRITIC:-(없음 — 순수 BC)}"
echo "    n_cand=$N_CAND keep=$PARL_KEEP temp=$PARL_TEMP guide=$GUIDE_SHOWN guide_all=${GUIDE_ALL:-0} ood_gate=$OOD_GATE"
echo "    replan=$REPLAN latency=$LATENCY  ->  http://$HOST:$PORT"
echo "    rrc 쪽 inference_latency_steps 가 $LATENCY, execution_horizon 이 $REPLAN 인지 확인할 것"

# --sim-wrapper 는 **쓰지 않는다**: 그것은 flat 키(video.<cam>/state.<name>)를 보내는
# sim 클라이언트용이다. 실기 rrc 는 중첩 dict 를 보내므로 씌우면 키가 안 맞는다.
ARGS=(--exp "$EXP" --model-path "$CKPT"
      --rtc-inference-mode trained --rtc-exec-horizon "$REPLAN"
      --host "$HOST" --port "$PORT" --device "$DEVICE" --log-every "$LOG_EVERY"
      --n-cand "$N_CAND")
if [ -n "$CRITIC" ]; then
  ARGS+=(--artifacts "$CRITIC"
         --parl-keep "$PARL_KEEP" --parl-temp "$PARL_TEMP"
         --guide-steps "$GUIDE_STEPS" --guide-move "$GUIDE_MOVE"
         --guide-step-size "$GUIDE_STEP_SIZE"
         --ood-gate "$OOD_GATE")
fi
[ -n "${DUMP_OBS:-}" ] && ARGS+=(--dump-obs "$DUMP_OBS")
[ -n "${GUIDE_GROUPS:-}" ] && ARGS+=(--guide-groups "$GUIDE_GROUPS")
# GUIDE_ALL=1: 후보 전부 상승 → argmax (relabel_parl 과 같은 순서). PARL_KEEP=0 과 함께 쓸 것.
[ "${GUIDE_ALL:-0}" != 0 ] && ARGS+=(--guide-all)

exec "$PY" -u -m rl.vla_rldx serve "${ARGS[@]}"
