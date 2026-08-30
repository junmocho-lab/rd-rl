#!/bin/bash
#SBATCH --job-name=dexjoco-hammer-nail-rollout-fix-scene-ep-200-rtc-delay-5-replan-20-max-steps-360
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/rlwrld2/home/junmo_cho/ws/rd-rl/out/dexjoco-rollout-fix-scene_%j.out
#SBATCH --error=/rlwrld2/home/junmo_cho/ws/rd-rl/out/dexjoco-rollout-fix-scene_%j.err

set -uo pipefail

REPO=/rlwrld2/home/junmo_cho/ws/rd-rl
CKPT=$REPO/checkpoints/dexjoco/dexjoco_hammer_nail_randobj_100ep_ptimg_h32_rtc8tr_drop03_bs16_30k_1gpu_mlxp
OUT=$REPO/rl-dataset/dexjoco/hammer_nail_d5r20_s0
PORT=20521
EPISODES=200
RTC_DELAY=5          # = yaml inference_latency
REPLAN=20            # = yaml replan_steps
MAX_STEPS=360
SCENE=0              # scene 고정

SIM_PY=/workspace/junmo_cho/dexjoco/venv/bin/python
RLDX_PY=$REPO/third_party/RLDX-1/.venv/bin/python

export HF_HOME=/workspace/junmo_cho/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1

mkdir -p "$OUT"

# ── 정책 서버 ────────────────────────────────────────────────────────────────
cd "$REPO/third_party/RLDX-1"
"$RLDX_PY" rldx/eval/run_rldx_server.py \
    --model-path "$CKPT" \
    --embodiment-tag GENERAL_EMBODIMENT --use-sim-policy-wrapper \
    --rtc-inference-mode trained \
    --rtc-inference-delay $RTC_DELAY --rtc-inference-exec-horizon $REPLAN \
    --host 127.0.0.1 --port $PORT &
trap "kill $! 2>/dev/null" EXIT

# ── 롤아웃 ────────────────────────────────────────────────────────────────────
cd "$REPO"
PYTHONPATH="$REPO/third_party/dexjoco/dexjoco" MUJOCO_GL=egl \
  "$SIM_PY" -u sim/dexjoco/rollout_dexjoco.py \
    --task hammer_nail --episodes $EPISODES --port $PORT \
    --replan $REPLAN --rtc-delay $RTC_DELAY --max-episode-steps $MAX_STEPS \
    --fixed-scene $SCENE --resume --log-every 25 \
    --output "$OUT"
