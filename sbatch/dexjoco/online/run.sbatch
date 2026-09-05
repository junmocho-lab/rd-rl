#!/bin/bash
#SBATCH --job-name=dexjoco-hammer-nail-online-rl
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --no-requeue   # 선점되면 조용히 재시작하지 말 것 — 재제출하면 디스크 상태에서 이어간다
#SBATCH --time=36:00:00
#SBATCH --output=out/dexjoco-online_%j.out
#SBATCH --error=out/dexjoco-online_%j.err
#
# dexjoco sim 온라인 EXPO-FT RL — 한 노드에서 [GPU0 정책서버 + GPU1 learner + CPU 롤아웃].
# 설계: configs/exp/dexjoco_hammer_nail_d5r20_online.yaml 머리말 + sim/dexjoco/online_driver.py.
#
#   RUN  : run id (메일박스 runs/<RUN>/, 산출물 checkpoints/expo/<RUN>/, wandb id).
#          재시도마다 새 이름을 줄 것 (fuji_online_2 처럼: dexjoco_online_1, _2, ...)
#   EXP  : exp yaml 이름
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl   # 제출 플러그인용
#   sbatch --export=ALL,RUN=dexjoco_online_1 sbatch/dexjoco/online/run.sbatch
#
# 재시작: 같은 RUN 으로 다시 제출하면 된다 — DONE 라운드는 건너뛰고, 반쯤 구른 라운드는
# rollout --resume 이 이어받고, wandb 는 step 이어붙기(learner/loop.py)로 곡선이 이어진다.

if [ -z "${MODEL_OUTPUT_DIR:-}" ]; then
    for _p in /fsx/rlwrld-unified-checkpoints /rlwrld-unified-checkpoints; do
        if [ -d "$_p/$USER" ]; then MODEL_OUTPUT_DIR="$_p/$USER/rd-rl"; break; fi
    done
fi
export MODEL_OUTPUT_DIR

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
[ -d "$REPO/configs/exp" ] || { echo "REPO 가 레포 루트가 아니다: $REPO"; exit 2; }
cd "$REPO"

RUN=${RUN:-dexjoco_online_1}
EXP=${EXP:-dexjoco_hammer_nail_d5r20_online}
PORT=${PORT:-$((21000 + ${SLURM_JOB_ID:-$$} % 9000))}
SIM_PY=${SIM_PY:-/workspace/junmo_cho/dexjoco/venv/bin/python}
PY=${PY:-$REPO/third_party/RLDX-1/.venv/bin/python}
[ -x "$PY" ] || { echo "RLDX venv python 이 없다: $PY"; exit 3; }
[ -x "$SIM_PY" ] || { echo "dexjoco venv python 이 없다: $SIM_PY"; exit 3; }

export HF_HOME=${HF_HOME:-/workspace/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"

echo "=== $(date -Is)  dexjoco online RL  run=$RUN exp=$EXP port=$PORT"
echo "    learner GPU1 / server GPU0 / rollout CPU+EGL"

# ── learner (GPU 1, 단일 프로세스 — 실효 배치 = expo.batch_size = 64 원본 파리티) ──
CUDA_VISIBLE_DEVICES=1 "$PY" -u learner/loop.py \
    --exp "$RUN" \
    --runs-root "$REPO/runs" --ckpt-root "$REPO/checkpoints" --repo "$REPO" \
    --exp-config "configs/exp/$EXP.yaml" \
    --expected-hw 256x256 \
    --wandb-project rd-rl-expo &
LEARNER_PID=$!
trap 'kill $LEARNER_PID 2>/dev/null; pkill -P $$ 2>/dev/null' EXIT

# ── driver (θ₀ 대기 → 서버(GPU0) 기동 → 라운드 루프). 서버는 driver 의 자식이다 ──
"$PY" -u sim/dexjoco/online_driver.py \
    --exp "$RUN" --exp-config "configs/exp/$EXP.yaml" \
    --runs-root "$REPO/runs" --ckpt-root "$REPO/checkpoints" \
    --port "$PORT" --sim-py "$SIM_PY" --server-gpu 0
rc=$?

echo "=== $(date -Is)  종료 rc=$rc  (learner 정리)"
kill $LEARNER_PID 2>/dev/null
exit $rc
