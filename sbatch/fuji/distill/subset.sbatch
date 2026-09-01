#!/bin/bash
#SBATCH --job-name=fuji-rby1m-rh56f1-distill-subset-success-only
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --time=1:00:00
#SBATCH --output=out/fuji-distill-subset_%j.out
#SBATCH --error=out/fuji-distill-subset_%j.err
#
# relabel 출력에서 성공 에피소드만 추린다 (distillation 입력용).
#
# 왜 필요한가: `--eps success` 로 relabel 하면 성공 에피소드만 액션이 갈리고 **실패
# 에피소드는 원본 액션 그대로 남는다**. 그대로 학습하면 회복 불가능한 상태의 원본 액션이
# BC 를 희석한다. `--eps all` arm 은 전부 갈렸으므로 이 단계가 필요 없다.
#
# ── dexjoco 판(subset_and_merge.sbatch MODE=subset)과 다른 점 ────────────────
# 그 판은 출력 이름을 **고정**한다:
#     distill_arm1_bc_success  /  distill_arm2_relabel_success
# arm 이 여러 개면 (critic 태그 x guide_move) 전부 같은 디렉토리를 덮어쓴다.
# 여기서는 --export 로 받은 SRC/OUT 을 그대로 쓴다.
#
#   sbatch --export=ALL,SRC=rl-dataset/fuji/relabel_...,OUT=rl-dataset/fuji/distill_... \
#          sbatch/fuji/distill/subset.sbatch
#
# GPU 를 요청하지 않는다: make_subset.py 는 parquet 필터 + 비디오 심링크만 한다.

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

SRC=${SRC:?SRC 를 줘야 한다 (relabel 출력 경로, 레포 상대)}
OUT=${OUT:?OUT 를 줘야 한다 (추린 결과 경로, 레포 상대)}
EPS=${EPS:-success}

[ -d "$SRC" ] || { echo "입력 없음: $SRC"; exit 3; }

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
PY=$REPO/third_party/RLDX-1/.venv/bin/python

echo "=== $(date -Is)  subset  --eps $EPS"
echo "    src $SRC"
echo "    out $OUT"
$PY -u rl/make_subset.py --data "$SRC" --out "$OUT" --eps "$EPS" || exit 1
echo "=== $(date -Is)  완료  $OUT"
