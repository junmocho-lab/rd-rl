#!/bin/bash
# fuji 전체 체인: cogfeat → critic(success/all) → relabel → LoRA distill → merge.
# **레포 루트에서** 실행한다. preflight.sh 를 먼저 통과시킬 것.
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   export HF_HOME=/fsx/rlwrld/junmo_cho/hf_cache
#   bash sbatch/fuji/chain.sh
#
# 전부 afterok 로 묶여 있어 앞이 실패하면 뒤가 자동 취소된다.
#
# ── 이 태스크에 평가 단계가 없는 이유 ───────────────────────────────────────
# fuji 는 실기라 시뮬레이터 롤아웃 하네스가 없다. 체인은 **병합된 체크포인트까지**
# 만들고 끝난다. 성능은 로봇에서 직접 재야 한다.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
[ -d configs/exp ] || { echo "레포 루트에서 실행할 것"; exit 2; }
mkdir -p out
: "${MODEL_OUTPUT_DIR:?셸에서 export 할 것 — 제출 플러그인이 제출 시점 환경을 본다}"
export MODEL_OUTPUT_DIR HF_HOME

E=fuji_d3r8
CSTEP=${CSTEP:-5000}          # distillation 에 쓸 critic 스텝
LSTEP=${LSTEP:-30000}         # LoRA 학습 스텝
GPUS=${GPUS:-4}               # LoRA 학습에 쓸 GPU 수. global-batch-size 는 전역이라
                              # 몇 장을 쓰든 유효 배치는 같고 속도만 바뀐다
WANDB_PROJECT=${WANDB_PROJECT:-rd-rl-distill}
# 파티션은 머신마다 다르다 (rlwrld2 = rlwrld, fsx = dedicated-rd). sbatch 파일의
# #SBATCH --partition 은 명령줄 -p 가 덮으므로, 실제로 존재하는 것을 골라 붙인다.
if [ -z "${PART:-}" ]; then
    for _p in rlwrld dedicated-rd batch; do
        if sinfo -h -p "$_p" >/dev/null 2>&1 && [ -n "$(sinfo -h -p "$_p" -o %P 2>/dev/null)" ]; then
            PART=$_p; break
        fi
    done
fi
[ -n "${PART:-}" ] || { echo "쓸 수 있는 파티션을 못 찾았다. PART=<이름> 으로 지정할 것"; exit 2; }
echo "파티션: $PART   LoRA GPU: ${GPUS:-4}장/arm"
P="-p $PART"
sb() { sbatch --parsable $P "$@"; }

echo "══════ 1. cogfeat (218,359프레임, images.mm ~118GB, 약 1.5h) ══════"
CG=$(sb -J fuji-rby1m-rh56f1-cogfeat-extract-0831-all-335ep-218k-frames-vlm-features-actnorm \
     --export=ALL,EXP=$E sbatch/dexjoco/critic/cogfeat.sbatch)
echo "  $CG"

echo "══════ 2. critic 2개 (각 200K, 약 25분). 격자에 5K 가 포함된다 ══════"
declare -A CJ
for T in success all; do
    CJ[$T]=$(sb --dependency=afterok:$CG \
        -J fuji-rby1m-rh56f1-critic-d3r8-train$T-doubleq-steps200000-qvgm-iql-rightarm-77dim \
        --export=ALL,EXP=$E,TRAIN_EPS=$T,STEPS=200000 \
        sbatch/dexjoco/critic/critic_hammer_nail.sbatch)
    echo "  ${CJ[$T]}  critic $T  (after $CG)"
done

echo "══════ 3. relabel — test-time Q guidance 로 액션을 갈아끼운다 ══════"
echo "  후보 32 → Q(min) top-10 → ∇_A Q 상승 4스텝(keep-best) → Categorical(Q/0.001)"
# critic 학습 데이터와 relabel 대상을 맞춘다: success critic 은 성공 에피소드만,
# all critic 은 전부. critic 이 본 분포 밖은 외삽이라 신뢰할 수 없다.
RS=$(sb --dependency=afterok:${CJ[success]} \
     -J fuji-rby1m-rh56f1-relabel-parl-critic-success-step5000-success-episodes-only \
     --export=ALL,EXP=$E,CTAG=success,STEP=$CSTEP,EPS=success \
     sbatch/dexjoco/distill/relabel.sbatch)
RA=$(sb --dependency=afterok:${CJ[all]} \
     -J fuji-rby1m-rh56f1-relabel-parl-critic-all-step5000-all-episodes-both-outcomes \
     --export=ALL,EXP=$E,CTAG=all,STEP=$CSTEP,EPS=all \
     sbatch/dexjoco/distill/relabel.sbatch)
echo "  $RS  relabel success  (after ${CJ[success]})"
echo "  $RA  relabel all      (after ${CJ[all]})"

echo "══════ 4. success arm 은 성공 에피소드만 추린다 ══════"
# relabel 출력에는 실패 에피소드도 원본 액션 그대로 남아 있으므로 걸러야 한다.
# all arm 은 전부 쓰므로 이 단계가 없다.
REL_S=rl-dataset/fuji/relabel_${E}__success@$((CSTEP/1000))k_success
SUB=$(sb --dependency=afterok:$RS \
      -J fuji-rby1m-rh56f1-distill-build-success-only-subset-from-relabeled-dataset \
      --export=ALL,EXP=$E,MODE=subset,REL=$REL_S \
      sbatch/dexjoco/distill/subset_and_merge.sbatch)
echo "  $SUB  subset (after $RS)"

echo "══════ 5. LoRA distill 2개 (각 $LSTEP 스텝, GPU ${GPUS}장, wandb=$WANDB_PROJECT) → 병합 ══════"
L_S=$(sb --dependency=afterok:$SUB --gres=gpu:$GPUS \
      -J fuji-rby1m-rh56f1-distill-lora-finetune-success-arm-action-expert-30000steps \
      --export=ALL,EXP=$E,DATA=rl-dataset/fuji/distill_arm2_relabel_success,TAG=S,STEPS=$LSTEP,WANDB=1,WANDB_PROJECT=$WANDB_PROJECT \
      sbatch/dexjoco/distill/lora_train.sbatch)
L_A=$(sb --dependency=afterok:$RA --gres=gpu:$GPUS \
      -J fuji-rby1m-rh56f1-distill-lora-finetune-all-episodes-arm-action-expert-30000steps \
      --export=ALL,EXP=$E,DATA=rl-dataset/fuji/relabel_${E}__all@$((CSTEP/1000))k_all,TAG=A,STEPS=$LSTEP,WANDB=1,WANDB_PROJECT=$WANDB_PROJECT \
      sbatch/dexjoco/distill/lora_train.sbatch)
for spec in "S:$L_S" "A:$L_A"; do
    T=${spec%%:*}; D=${spec##*:}
    M=$(sb --dependency=afterok:$D \
        -J fuji-rby1m-rh56f1-distill-merge-lora-adapter-arm-$T-into-base-checkpoint \
        --export=ALL,EXP=$E,MODE=merge,TAG=$T \
        sbatch/dexjoco/distill/subset_and_merge.sbatch)
    echo "  $D → $M   arm $T"
done

cat <<'TXT'

══════ 아침에 볼 것 ══════
  sacct -u $USER -S today -X -o JobID,State,JobName%70 | grep -v COMPLETED    끊긴 곳
  grep "로그된 액션이 이긴 비율" out/dexjoco-relabel_*.out                    ← 70% 미만이어야 의미가 있다
  grep "AUC" out/dexjoco-critic_*.out | tail -20                              critic 이 성패를 구분하나

  산출물: checkpoints/fuji_distill/{S,A}_merged
          → 추론 코드 변경 없이 AutoModel.from_pretrained 로 읽힌다. 로봇에서 평가할 것.

  주의: critic 스텝 5K 는 dexjoco 근거로 고른 값이 아니다 (A 는 20K, B 는 1K 가 최적이었고
        서로 전이되지 않았다). fuji 의 최적 스텝은 실기 평가로만 알 수 있다.
        격자에 1K/5K/20K/25K/50K/100K/200K 가 전부 남아 있으니 나중에 바꿔 쓸 수 있다.
TXT
