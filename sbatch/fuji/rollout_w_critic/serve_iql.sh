#!/bin/bash
# fuji — **IQL guidance 서빙** (rollouts_v1 critic + base 0901-0903 checkpoint-40000).
#
# serve.sh 의 래퍼다. 다른 점: EXP/모델/critic 위치가 IQL 재학습 산출물로 고정되고,
# 스윕할 손잡이(critic 스텝, guidance 스텝수/보폭)만 앞으로 뺐다.
#
# ⚠ critic(rollouts_v1)의 cogfeat 은 checkpoint-40000 산물이다 — 다른 base 로 서빙하면
#   feature 분포가 어긋난다. MODEL_PATH 를 덮어쓸 거면 critic 도 같이 바꿀 것.
#
#   STEP        : critic 체크포인트 스텝. 1000/2000/3000/5000/7000/10000 (기본 5000 —
#                 홀드아웃 Q(실패끝) 최저 + AUC 1.0 + 시작점 미분리. 3000 이 보수적 백업)
#   METHOD      : bc | sel32 | parl | parl_sample  (bc = critic 없이 순수 base)
#   GUIDE_STEPS : ∇_A Q 상승 횟수 (기본 10)
#   GUIDE_STEP_SIZE : raw gradient 보폭 (기본 0.333 — 이전 relabel 합의값).
#                 0 이면 GUIDE_MOVE(총이동량) 모드로 떨어진다 (serve.sh 기본 0.05)
#   N_CAND      : 후보 수 (기본 METHOD 프리셋: sel32/parl=32)
#   PORT/HOST   : 기본 5555 / 127.0.0.1
#
#   bash sbatch/fuji/rollout_w_critic/serve_iql.sh                      # 기본값
#   STEP=3000 bash sbatch/fuji/rollout_w_critic/serve_iql.sh            # critic 스윕
#   GUIDE_STEPS=4 GUIDE_STEP_SIZE=0.1 bash .../serve_iql.sh             # guidance 스윕
#   METHOD=sel32 bash .../serve_iql.sh                                  # guidance 끄고 선택만
#   METHOD=bc bash .../serve_iql.sh                                     # 순수 base 참조

set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

STEP=${STEP:-5000}
CRITIC_DIR=${CRITIC_DIR:-$HOME/ws/junmo_cho/checkpoints/fuji_iql-critic/rollouts_v1}
CRITIC_PATH=${CRITIC_PATH:-$CRITIC_DIR/$(printf 'critic_%06d.pt' "$STEP")}
[ "${METHOD:-parl}" = bc ] || [ -f "$CRITIC_PATH" ] || {
  echo "critic 없음: $CRITIC_PATH"; echo "있는 것:"; ls "$CRITIC_DIR" | grep '^critic_'; exit 3; }

export EXP=${EXP:-fuji_iql}
export METHOD=${METHOD:-parl}
export MODEL_PATH=${MODEL_PATH:-$HOME/ws/junmo_cho/checkpoints/rldx-img-curated/rldx_img_0901-0903-hilw1-dp5-30k-4gpu-128b/checkpoint-40000}
export CRITIC_PATH
export GUIDE_STEPS_OVERRIDE=${GUIDE_STEPS:-10}
export GUIDE_STEP_SIZE=${GUIDE_STEP_SIZE:-0.333}
export NAME=${NAME:-${METHOD}__iql_rollouts_v1@$((STEP/1000))k_g${GUIDE_STEPS_OVERRIDE}x${GUIDE_STEP_SIZE}}
[ -n "${N_CAND:-}" ] && export N_CAND_OVERRIDE=$N_CAND

exec bash "$HERE/serve.sh"
