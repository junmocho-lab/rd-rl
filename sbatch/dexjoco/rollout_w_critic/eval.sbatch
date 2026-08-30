#!/bin/bash
#SBATCH --job-name=dexjoco-hammer-nail-eval-rollout-with-critic-selection-and-guidance-200ep
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue   # preempt 되면 조용히 재시작하지 말고 죽어라
                       # (background 는 PreemptMode=REQUEUE 라 런이 섞인다)
#SBATCH --time=4:00:00   # 실측 51~58분. 요청 시간을 실측에 맞춰야
                       # Slurm backfill 이 빈 슬롯에 끼워 넣는다 (12h 를 걸면 대기가 길어진다)
#SBATCH --output=out/dexjoco-eval_%j.out
#SBATCH --error=out/dexjoco-eval_%j.err
#
# critic 을 붙인 롤아웃 평가. 하나로 통일했다.
#   EXP   : replan / rtc_delay / dataset / 씬 고정 여부를 전부 여기서 끌어온다
#   CTAG  : critic 학습 태그   (success | all | success_ens | all_ens)
#   STEP  : critic 스텝        (20000 등). 0 이면 critic_latest.pt
#   METHOD: bc | sel32 | parl
#
#   sbatch --export=ALL,EXP=dexjoco_hammer_nail_d2r8_s0,CTAG=success,STEP=20000,METHOD=sel32 \
#          sbatch/dexjoco/rollout_w_critic/eval.sbatch
#
# 실측 (A=d2r8_s0, 고정씬, critic success@20K, 100ep):
#   BC 61.5%  /  sel32 79.0%  /  parl 89.0%
# parl = 후보 32 -> top-10 -> ∇_A Q 상승 4스텝(move 0.05) -> argmax.
# guide_move 는 0.01~0.05 가 정점이고 0.1 에서 꺾인다. 후보를 전부 상승시키는
# --guide-all 은 Q 가 반드시 높아지는데도 실측이 낮았다 (72% vs 84%) — critic 이
# 가장 크게 과대평가한 것이 뽑히기 때문이다. 그래서 top-M 필터가 필요하다.

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
METHOD=${METHOD:-sel32}
CTAG=${CTAG:-success}
STEP=${STEP:-20000}
EPISODES=${EPISODES:-200}
MAX_STEPS=${MAX_STEPS:-360}
PORT=${PORT:-$((20000 + ${SLURM_JOB_ID:-$$} % 9000))}

# ── 세팅은 전부 yaml 과 EXP 이름에서 끌어온다 ────────────────────────────────
REPLAN=$(awk '/^replan_steps:/{print $2; exit}'      configs/exp/$EXP.yaml)
RTC_DELAY=$(awk '/^inference_latency:/{print $2; exit}' configs/exp/$EXP.yaml)
case $EXP in *_s0) SCENE=0 ;; *) SCENE=-1 ;; esac      # _s0 = 고정 장면

# ── 방법 프리셋 ──────────────────────────────────────────────────────────────
case $METHOD in
  bc)    N_CAND=1  ; PARL_KEEP=0  ; GUIDE_STEPS=0 ; GUIDE_MOVE=0    ;;
  sel32) N_CAND=32 ; PARL_KEEP=0  ; GUIDE_STEPS=0 ; GUIDE_MOVE=0    ;;
  parl)  N_CAND=32 ; PARL_KEEP=10 ; GUIDE_STEPS=4 ; GUIDE_MOVE=0.05 ;;
  *) echo "METHOD 는 bc | sel32 | parl"; exit 2 ;;
esac
PARL_TEMP=${PARL_TEMP:-0}

CKPT=$REPO/checkpoints/dexjoco/dexjoco_hammer_nail_randobj_100ep_ptimg_h32_rtc8tr_drop03_bs16_30k_1gpu_mlxp
if [ "$METHOD" = bc ]; then
  NAME=${NAME:-bc}
  CRITIC=""
else
  if [ "$STEP" -eq 0 ]; then CF=critic_latest.pt; else CF=$(printf 'critic_%06d.pt' $STEP); fi
  CRITIC=$REPO/checkpoints/${EXP}-critic/${CTAG}/${CF}
  NAME=${NAME:-${METHOD}__${CTAG}@$((STEP/1000))k}
  [ -f "$CRITIC" ] || { echo "critic 없음: $CRITIC"; exit 3; }
fi
OUT=$REPO/rl-dataset/dexjoco/eval_${EXP}/${NAME}

export HF_HOME=${HF_HOME:-/workspace/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
SIM_PY=${SIM_PY:-/workspace/junmo_cho/dexjoco/venv/bin/python}   # dexjoco 는 python<3.12 라 rd-rl/.venv 불가

# 결과가 무엇으로 나왔는지 rollout_summary.json["serve"] 에 박힌다
export RD_SERVE_INFO="{\"method\":\"$METHOD\",\"critic\":\"$CRITIC\",\"ctag\":\"$CTAG\",\"step\":$STEP,\"n_cand\":$N_CAND,\"parl_keep\":$PARL_KEEP,\"parl_temp\":$PARL_TEMP,\"guide_steps\":$GUIDE_STEPS,\"guide_move\":$GUIDE_MOVE,\"base_policy\":\"$CKPT\"}"

echo "=== $(date -Is)  $EXP / $NAME"
echo "    replan=$REPLAN rtc_delay=$RTC_DELAY scene=$SCENE ep=$EPISODES port=$PORT"
echo "    critic $CRITIC"
echo "    out    $OUT"

# ── 정책 서버 ────────────────────────────────────────────────────────────────
export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
if [ "$METHOD" = bc ]; then
  third_party/RLDX-1/.venv/bin/python -u -m rl.vla_rldx serve \
      --exp $EXP --model-path "$CKPT" \
      --rtc-inference-mode trained --rtc-exec-horizon $REPLAN \
      --sim-wrapper --log-every 25 --host 127.0.0.1 --port $PORT &
else
  third_party/RLDX-1/.venv/bin/python -u -m rl.vla_rldx serve \
      --exp $EXP --model-path "$CKPT" --artifacts "$CRITIC" \
      --rtc-inference-mode trained --rtc-exec-horizon $REPLAN \
      --n-cand $N_CAND --parl-keep $PARL_KEEP --parl-temp $PARL_TEMP \
      --guide-steps $GUIDE_STEPS --guide-move $GUIDE_MOVE \
      --sim-wrapper --log-every 25 --host 127.0.0.1 --port $PORT &
fi
trap "kill $! 2>/dev/null" EXIT

# ── 롤아웃 (끝나면 trap 이 서버를 죽이고 잡이 종료된다) ──────────────────────
PYTHONPATH="$REPO/third_party/dexjoco/dexjoco" MUJOCO_GL=egl \
  "$SIM_PY" -u sim/dexjoco/rollout_dexjoco.py \
    --task hammer_nail --episodes $EPISODES --port $PORT \
    --replan $REPLAN --rtc-delay $RTC_DELAY --max-episode-steps $MAX_STEPS \
    --fixed-scene $SCENE --resume --log-every 25 \
    --output "$OUT"
echo "=== $(date -Is)  완료  $NAME"
