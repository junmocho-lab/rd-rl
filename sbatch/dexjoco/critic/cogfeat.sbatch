#!/bin/bash
#SBATCH --job-name=dexjoco-hammer-nail-cogfeat-extract-vlm-features-and-actnorm-cache
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue   # preempt 되면 조용히 재시작하지 말고 죽어라
                       # (background 는 PreemptMode=REQUEUE 라 런이 섞인다)
#SBATCH --time=12:00:00
#SBATCH --output=/rlwrld2/home/junmo_cho/ws/rd-rl/out/dexjoco-cogfeat_%j.out
#SBATCH --error=/rlwrld2/home/junmo_cho/ws/rd-rl/out/dexjoco-cogfeat_%j.err
#
# cogfeat + actnorm 캐시만 만든다. critic 학습 잡을 여러 개 동시에 던지기 전에
# EXP 당 한 번 돌려야 한다 — 안 그러면 여러 잡이 같은 캐시를 동시에 쓰며 깨진다.
#
#   sbatch --export=ALL,EXP=dexjoco_hammer_nail_d5r20 sbatch/dexjoco/critic/cogfeat.sbatch

: "${MODEL_OUTPUT_DIR:=/rlwrld-unified-checkpoints/junmo_cho/dexjoco-rl}"
export MODEL_OUTPUT_DIR

set -uo pipefail
REPO=/rlwrld2/home/junmo_cho/ws/rd-rl
cd "$REPO"

EXP=${EXP:-dexjoco_hammer_nail_d2r8_s0}
DATA=$(awk '/^dataset:/{print $2; exit}' configs/exp/$EXP.yaml)

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
export HF_HOME=/workspace/junmo_cho/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
PY=$REPO/third_party/RLDX-1/.venv/bin/python

echo "=== $(date -Is)  cogfeat  $EXP  <- $DATA"
$PY -u -m rl.extract_cogfeat --exp $EXP --data $DATA \
    --checkpoints checkpoints --batch 64 --resume || exit 1

# actnorm 은 학습 첫 잡이 만들지만, 여기서 미리 만들어 두면 경쟁이 없다.
$PY -u -m rl.offline_iql_qvgm --exp $EXP --data $DATA \
    --checkpoints checkpoints --features cogfeat.npy \
    --action-groups eef_position,eef_rotation --no-stepwise \
    --latent 2048 --state-latent 256 --hidden 1024,512 \
    --steps 1 --eval-every 1000000 --keep-last 0 --no-wandb --tag _warmup
rm -rf "checkpoints/${EXP}-critic/_warmup"
du -sh "checkpoints/${EXP}-critic" 2>/dev/null
echo "=== $(date -Is)  완료"
