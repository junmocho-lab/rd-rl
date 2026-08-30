#!/bin/bash
# 밤새 돌릴 체인. **레포 루트에서** 실행한다.
#
#   export MODEL_OUTPUT_DIR=<제출 플러그인이 허용하는 경로>
#   bash sbatch/dexjoco/chain_tonight.sh
#
# 두 갈래를 건다:
#   (A) C(랜덤씬) eval 11개 — 이미 던진 C critic 4개에 --dependency 로 매단다
#   (B) action distillation — relabel dry-run 을 관문으로 두고 그 뒤를 전부 연결
#
# 의존은 afterok 이라 앞이 실패하면 뒤는 자동으로 취소된다. 아침에 sacct 로
# 어디서 끊겼는지 보면 된다.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
[ -d configs/exp ] || { echo "레포 루트에서 실행할 것"; exit 2; }
mkdir -p out

: "${MODEL_OUTPUT_DIR:?셸에서 export 해야 한다 — 제출 플러그인이 제출 시점 환경을 본다}"
export MODEL_OUTPUT_DIR

# 이미 큐에 있는 잡들 (네가 던진 것)
C_SUCC=${C_SUCC:-145022}      # C critic success (double-Q)
C_ALL=${C_ALL:-145024}        # C critic all     (double-Q)
DRY=${DRY:-144856}            # relabel dry-run  (distillation 관문)

E=dexjoco_hammer_nail_d2r8_s0
CE=dexjoco_hammer_nail_d5r20
sb() { sbatch --parsable "$@"; }

echo "══════ (A) C 랜덤씬 eval ══════"
# C 는 A(20K 봉우리)와 B(1K 봉우리)가 서로 달랐으므로 격자를 넓게 잡는다.
for spec in "success:$C_SUCC" "all:$C_ALL"; do
    CT=${spec%%:*}; DEP=${spec##*:}
    for S in 1000 5000 25000 100000 500000; do
        J=$(sb --dependency=afterok:$DEP \
            -J dexjoco-hammer-nail-eval-d5r20-randomscene-sel32-critic-$CT-step$S-200ep \
            --export=ALL,EXP=$CE,METHOD=sel32,CTAG=$CT,STEP=$S,EPISODES=200 \
            sbatch/dexjoco/rollout_w_critic/eval.sbatch)
        echo "  $J  sel32 $CT@$S  (after $DEP)"
    done
done
# C 의 BC 기준선 — 랜덤씬이라 같은 시드로 다시 재야 페어 비교가 된다.
# 수집 데이터 앞 200ep 의 51.5% 를 재현하면 시드 정렬이 맞는 것이다.
J=$(sb -J dexjoco-hammer-nail-eval-d5r20-randomscene-bc-basepolicy-nocritic-200ep \
     --export=ALL,EXP=$CE,METHOD=bc,EPISODES=200 \
     sbatch/dexjoco/rollout_w_critic/eval.sbatch)
echo "  $J  BC 기준선"

echo
echo "══════ (B) action distillation ══════"
echo "  관문: dry-run $DRY 가 성공해야 아래가 돈다"
echo "        (로그 액션이 이긴 비율 70% 이상이면 critic 이 base policy 를 못 이긴다는 뜻 —"
echo "         잡은 성공해도 아침에 그 숫자를 반드시 확인할 것)"

# 1. relabel 본편 — critic success@20K, 성공 에피소드만 (critic 학습 데이터와 맞춘다)
D1=$(sb --dependency=afterok:$DRY \
     -J dexjoco-hammer-nail-relabel-parl-full-critic-success-step20000-success-episodes-only \
     --export=ALL,EXP=$E,CTAG=success,STEP=20000,EPS=success \
     sbatch/dexjoco/distill/relabel.sbatch)
echo "  $D1  relabel 본편 (after $DRY)"

# 2. 서브셋 — arm1(원본 액션) / arm2(개선된 액션). 둘 다 같은 123 성공 에피소드다.
D2=$(sb --dependency=afterok:$D1 --export=ALL,EXP=$E,MODE=subset \
     sbatch/dexjoco/distill/subset_and_merge.sbatch)
echo "  $D2  subset arm1+arm2 (after $D1)"

# 3~5. arm 별로 LoRA 학습 → 병합 → 평가
for A in arm1_bc arm2_relabel; do
    case $A in
        arm1_bc)      DATA=rl-dataset/dexjoco/distill_arm1_bc_success ; T=arm1 ;;
        arm2_relabel) DATA=rl-dataset/dexjoco/distill_arm2_relabel_success ; T=arm2 ;;
    esac
    L=$(sb --dependency=afterok:$D2 \
        -J dexjoco-hammer-nail-distill-lora-finetune-$T-action-expert-2000steps-batch64 \
        --export=ALL,EXP=$E,DATA=$DATA,TAG=$T,STEPS=2000 \
        sbatch/dexjoco/distill/lora_train.sbatch)
    M=$(sb --dependency=afterok:$L \
        -J dexjoco-hammer-nail-distill-merge-lora-adapter-$T-into-base-checkpoint \
        --export=ALL,EXP=$E,MODE=merge,TAG=$T \
        sbatch/dexjoco/distill/subset_and_merge.sbatch)
    # 평가는 critic 없이 (METHOD=bc). distillation 의 목적이 test-time 계산 제거이므로
    # 후보 32개를 뽑지 않는 순정 롤아웃으로 재야 한다.
    V=$(sb --dependency=afterok:$M \
        -J dexjoco-hammer-nail-eval-d2r8s0-fixedscene-distilled-$T-nocritic-basepolicy-200ep \
        --export=ALL,EXP=$E,METHOD=bc,EPISODES=200,NAME=distill_$T,MODEL_PATH=$PWD/checkpoints/dexjoco_distill/${T}_merged \
        sbatch/dexjoco/rollout_w_critic/eval.sbatch)
    echo "  $L → $M → $V   $T"
done

echo
echo "══════ 아침에 볼 것 ══════"
cat <<'TXT'
  squeue -u $USER                                    남은 것
  sacct -u $USER -S today -X -o JobID,State,JobName%60 | grep -v COMPLETED   끊긴 곳
  python3 sim/dexjoco/summarize_evals.py             전체 결과표

  distillation 판정:
    grep "로그된 액션이 이긴 비율" out/dexjoco-relabel_*.out    ← 70% 미만이어야 의미가 있다
    기대치는 parl_argmax 89% 다. arm2 가 그 근처면 성공,
    arm1(원본 액션 대조군)과의 차이가 액션 개선의 순효과다.
TXT
