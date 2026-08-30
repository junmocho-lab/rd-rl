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
#SBATCH --output=out/dexjoco-cogfeat_%j.out
#SBATCH --error=out/dexjoco-cogfeat_%j.err
#
# cogfeat + actnorm 캐시만 만든다. critic 학습 잡을 여러 개 동시에 던지기 전에
# EXP 당 한 번 돌려야 한다 — 안 그러면 여러 잡이 같은 캐시를 동시에 쓰며 깨진다.
#
#   sbatch --export=ALL,EXP=dexjoco_hammer_nail_d5r20 sbatch/dexjoco/critic/cogfeat.sbatch

# 제출 플러그인이 MODEL_OUTPUT_DIR 를 요구하는데 **머신마다 허용 접두사가 다르다**:
#   rlwrld2 : /rlwrld-unified-checkpoints/$USER/...
#   fsx     : /fsx/rlwrld-unified-checkpoints/$USER/...
# 이 잡은 거기에 아무것도 쓰지 않는다 (산출물은 전부 $REPO 아래). 플러그인 통과용이다.
#
# ★ 플러그인은 **제출 시점의 환경**을 본다. 그래서 아래 기본값은 `bash 이 파일` 로
#   직접 돌릴 때만 효과가 있고, sbatch 로 던질 때는 셸에서 미리 export 해야 한다:
#       export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   그 뒤 --export=ALL 이 그것을 실어 나른다.
if [ -z "${MODEL_OUTPUT_DIR:-}" ]; then
    for _p in /fsx/rlwrld-unified-checkpoints /rlwrld-unified-checkpoints; do
        if [ -d "$_p/$USER" ]; then MODEL_OUTPUT_DIR="$_p/$USER/rd-rl"; break; fi
    done
fi
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
DATA=$(awk '/^dataset:/{print $2; exit}' configs/exp/$EXP.yaml)

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
export HF_HOME=${HF_HOME:-/workspace/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
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
