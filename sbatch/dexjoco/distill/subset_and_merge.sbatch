#!/bin/bash
#SBATCH --job-name=dexjoco-hammer-nail-distill-build-subsets-or-merge-lora-adapter-into-base
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --time=1:00:00
#SBATCH --output=out/dexjoco-distill-aux_%j.out
#SBATCH --error=out/dexjoco-distill-aux_%j.err
#
# action distillation 의 잡다한 두 단계. MODE 로 고른다.
#
#   MODE=subset  : 성공 에피소드만 뽑아 학습용 데이터셋을 만든다 (arm1 = 원본 액션,
#                  arm2 = relabel 된 액션). 비디오는 심링크라 용량을 안 먹는다.
#   MODE=merge   : LoRA 어댑터를 base 에 병합해 추론 가능한 체크포인트로 만든다.
#
#   sbatch --export=ALL,MODE=subset sbatch/dexjoco/distill/subset_and_merge.sbatch
#   sbatch --export=ALL,MODE=merge,TAG=arm2 sbatch/dexjoco/distill/subset_and_merge.sbatch

if [ -z "${MODEL_OUTPUT_DIR:-}" ]; then
    for _p in /fsx/rlwrld-unified-checkpoints /rlwrld-unified-checkpoints; do
        if [ -d "$_p/$USER" ]; then MODEL_OUTPUT_DIR="$_p/$USER/rd-rl"; break; fi
    done
fi
export MODEL_OUTPUT_DIR

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
if [ ! -d "$REPO/configs/exp" ]; then echo "REPO 가 레포 루트가 아니다: $REPO"; exit 2; fi
cd "$REPO"

EXP=${EXP:-dexjoco_hammer_nail_d2r8_s0}
MODE=${MODE:?MODE 를 줘야 한다 (subset | merge)}
V=/workspace/junmo_cho/dexjoco/venv/bin/python
export HF_HOME=${HF_HOME:-/workspace/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1

if [ "$MODE" = subset ]; then
    SRC=$(awk '/^dataset:/{print $2; exit}' configs/exp/$EXP.yaml)
    REL=${REL:-rl-dataset/dexjoco/relabel_${EXP}__success@20k_success}
    echo "=== $(date -Is)  subset"
    # arm1: 원본 액션, 성공만 — "성공 에피소드로 파인튜닝만 해도 오른다" 를 재는 대조군.
    #       이게 없으면 arm2 의 숫자를 액션 개선 덕으로 귀속시킬 수 없다.
    $V rl/make_subset.py --data "$SRC" --out rl-dataset/dexjoco/distill_arm1_bc_success --eps success || exit 1
    # arm2: relabel 된 액션, 성공만
    if [ -d "$REL" ]; then
        $V rl/make_subset.py --data "$REL" --out rl-dataset/dexjoco/distill_arm2_relabel_success --eps success || exit 1
    else
        echo "[경고] relabel 출력이 없다: $REL — arm2 는 건너뛴다"
    fi
    echo "=== $(date -Is)  완료"

elif [ "$MODE" = merge ]; then
    TAG=${TAG:?TAG 를 줘야 한다}
    BASE=$REPO/checkpoints/$(awk '/^base_policy:/{print $2; exit}' configs/exp/$EXP.yaml)
    RUN=$REPO/checkpoints/dexjoco_distill/${TAG}
    # 마지막 체크포인트를 고른다
    CK=$(ls -d $RUN/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -n "$CK" ] || { echo "체크포인트 없음: $RUN"; exit 3; }
    OUT=$REPO/checkpoints/dexjoco_distill/${TAG}_merged
    echo "=== $(date -Is)  merge  $CK → $OUT"
    cd "$REPO/third_party/RLDX-1"
    .venv/bin/python scripts/merge_lora_checkpoint.py \
        --trainable-ckpt "$CK" --base-ckpt "$BASE" --output "$OUT" || exit 1
    echo "=== $(date -Is)  완료. 평가: eval.sbatch 에 MODEL_PATH=$OUT 로 준다"
else
    echo "MODE 는 subset | merge"; exit 2
fi
