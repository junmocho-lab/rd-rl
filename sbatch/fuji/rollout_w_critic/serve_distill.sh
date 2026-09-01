#!/bin/bash
# fuji distillation 결과(base BC + LoRA 어댑터) 정책 서버.
#
# serve.sh 와 다른 점은 **어떤 정책을 띄우느냐** 하나다:
#   serve.sh          base BC + critic 을 붙여 test-time 에 액션을 고르거나 민다
#   serve_distill.sh  액션 개선이 **이미 정책 안에 증류된** LoRA 어댑터를 base 에 얹는다
#
# 그래서 기본 METHOD 가 bc 다 — critic 없이 정책이 바로 좋은 액션을 낸다. 이것이
# distillation 의 요점이고, 그래서 추론 비용이 base BC 와 같다 (후보 32개 채점도,
# ∇_A Q 상승도 없다).
#
#   ARM       : distillation arm. success_m0.01 | success_m0.05 | all_m0.01 |
#               all_m0.05 | success_m0.02 | success_m0.1 | all_m0.02 | all_m0.1
#   LORA_STEP : LoRA 스텝. 비우면 그 arm 의 최신 체크포인트
#               (2000 마다 저장돼 있어 "몇 스텝이 최적인가" 를 바로 쓸어볼 수 있다)
#
#   bash sbatch/fuji/rollout_w_critic/serve_distill.sh                    # 기본 success_m0.01
#   ARM=all_m0.05 bash sbatch/fuji/rollout_w_critic/serve_distill.sh
#   ARM=success_m0.01 HOST=0.0.0.0 bash .../serve_distill.sh              # rrc 가 다른 머신
#
# 비교 실험 — 같은 로봇/씬에서 이 순서로 재면 증류가 실제로 먹혔는지 갈린다:
#   1) base BC          METHOD=bc  bash serve.sh                     (기준선)
#   2) base + critic    METHOD=parl CTAG=success STEP=10000 bash serve.sh
#   3) distilled        ARM=success_m0.01 bash serve_distill.sh      (여기)
# 3 이 1 보다 좋고 2 에 근접하면 증류가 성공한 것이다. 3 이 2 를 넘으면 test-time
# 탐색 없이도 이득이 남았다는 뜻이라 더 좋다 (추론이 훨씬 싸다).
#
# 스텝을 쓸어보려면 (로봇이 한 대라 하나씩):
#   for s in 4000 8000 12000 16000; do
#     ARM=success_m0.01 LORA_STEP=$s bash sbatch/fuji/rollout_w_critic/serve_distill.sh
#   done

set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
[ -d "$REPO/configs/exp" ] || { echo "REPO 가 레포 루트가 아니다: $REPO"; exit 2; }
cd "$REPO"

EXP=${EXP:-fuji_d4r16}           # ★ distillation 은 d4r16 에서 나왔다 (replan 16 / latency 4).
                                 #   d3r8 을 쓰면 rtc-exec-horizon 이 8 로 들어가 청크가 어긋난다.
ARM=${ARM:-success_m0.01}
METHOD=${METHOD:-bc}             # 증류된 정책이라 critic 없이 쓴다
CTAG=${CTAG:-success}
STEP=${STEP:-10000}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-5555}
DEVICE=${DEVICE:-cuda}
LOG_EVERY=${LOG_EVERY:-25}

YAML=configs/exp/$EXP.yaml
[ -f "$YAML" ] || { echo "exp yaml 이 없다: $YAML"; exit 2; }
REPLAN=$(awk '/^replan_steps:/{print $2; exit}' "$YAML")
LATENCY=$(awk '/^inference_latency:/{print $2; exit}' "$YAML")

# ── 정책 = base BC + LoRA 어댑터 ─────────────────────────────────────────────
# merge(13GB/arm) 를 만들지 않는다. ExpoServer._load (rl/vla_rldx.py:540) 가
# --artifacts 의 sd["lora"] 를 읽어 base 위에 주입하는 경로가 이미 있다 (EXPO-FT 가
# 쓰던 것). 어댑터는 10MB 라 스텝별로 전부 두고 갈아끼울 수 있다.
CKPT=${MODEL_PATH:-$REPO/checkpoints/$(awk '/^base_policy:/{print $2; exit}' "$YAML")}
[ -d "$CKPT" ] || { echo "base 정책이 없다: $CKPT"; exit 3; }

# adapter.pt 는 체크포인트 디렉토리 안에 model.safetensors 옆으로 있다
# (lora_train.sbatch 가 학습 끝에 만든다). 별도 디렉토리를 대조할 필요가 없다.
ARM_DIR=$REPO/checkpoints/fuji_distill/$ARM
if [ -n "${LORA_STEP:-}" ]; then
  LORA=$(ls "$ARM_DIR"/*/checkpoint-${LORA_STEP}/adapter.pt 2>/dev/null | head -1)
else                                   # 비우면 최신 스텝
  LORA=$(ls "$ARM_DIR"/*/checkpoint-*/adapter.pt 2>/dev/null \
         | sed 's|.*/checkpoint-\([0-9]*\)/adapter.pt|\1 &|' | sort -n | tail -1 | cut -d' ' -f2)
fi
if [ -z "${LORA:-}" ] || [ ! -f "$LORA" ]; then
  echo "LoRA 어댑터가 없다: $ARM_DIR/*/checkpoint-${LORA_STEP:-<N>}/adapter.pt"
  echo "그 arm 에 있는 스텝:"
  ls -d "$ARM_DIR"/*/checkpoint-* 2>/dev/null | sed 's|.*checkpoint-|  |' | sort -n \
    || echo "  (아직 학습 전이거나 arm 이름이 틀렸다)"
  echo "adapter.pt 만 없다면:"
  echo "  third_party/RLDX-1/.venv/bin/python utils/adapter_to_artifacts.py --arm $ARM --all-steps"
  exit 3
fi
LSTEP_SHOWN=$(echo "$LORA" | sed 's|.*/checkpoint-\([0-9]*\)/.*|\1|')

N_CAND=${N_CAND_OVERRIDE:-1}     # 증류 정책은 후보를 고르지 않는다

# ★ --artifacts 인자는 하나뿐이라 LoRA 와 critic 을 동시에 넘길 수 없다.
#   증류 정책은 액션 개선이 이미 안에 들어 있으므로 critic 없이 쓰는 것이 정상이다.
#   test-time critic 과 비교하고 싶으면 base 정책에 대해 serve.sh 를 쓸 것.
if [ "$METHOD" != bc ]; then
  echo "METHOD=$METHOD 는 지원하지 않는다 — --artifacts 하나에 LoRA 와 critic 을 같이"
  echo "넣을 수 없다. 증류 정책은 METHOD=bc 로 서빙하고, critic 비교는 serve.sh 로 할 것."
  exit 2
fi
NAME=${NAME:-distill_${ARM}@${LSTEP_SHOWN}}

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
export HF_HOME=${HF_HOME:-/fsx/rlwrld/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
PY=${PY:-$REPO/third_party/RLDX-1/.venv/bin/python}
[ -x "$PY" ] || { echo "python 이 없다: $PY"; exit 3; }

export RD_SERVE_INFO="{\"exp\":\"$EXP\",\"kind\":\"distill\",\"arm\":\"$ARM\",\"lora_step\":$LSTEP_SHOWN,\"lora\":\"$LORA\",\"base\":\"$CKPT\"}"

echo "=== $(date -Is)  fuji serve (distill)  $NAME"
echo "    base   $CKPT"
echo "    LoRA   $LORA"
echo "    스텝   $LSTEP_SHOWN"
echo "    replan=$REPLAN latency=$LATENCY  ->  $HOST:$PORT"
echo "    ★ rrc 쪽 execution_horizon=$REPLAN, inference_latency_steps=$LATENCY 로 맞출 것"
echo "      (d3r8 시절의 8/3 이 남아 있으면 청크가 어긋난다)"

# --sim-wrapper 는 쓰지 않는다: flat 키를 보내는 sim 클라이언트용이고, 실기 rrc 는
# 중첩 dict 를 보낸다.
ARGS=(--exp "$EXP" --model-path "$CKPT" --artifacts "$LORA"
      --rtc-inference-mode trained --rtc-exec-horizon "$REPLAN"
      --host "$HOST" --port "$PORT" --device "$DEVICE" --log-every "$LOG_EVERY"
      --n-cand "$N_CAND")
[ -n "${DUMP_OBS:-}" ] && ARGS+=(--dump-obs "$DUMP_OBS")
[ -n "${GUIDE_GROUPS:-}" ] && ARGS+=(--guide-groups "$GUIDE_GROUPS")

exec "$PY" -u -m rl.vla_rldx serve "${ARGS[@]}"
