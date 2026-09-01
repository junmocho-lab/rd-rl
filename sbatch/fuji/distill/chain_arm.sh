#!/bin/bash
# 한 arm(critic 태그 x guide_move)의 distillation 을 afterok 로 엮어 던진다.
#
#   arm = <CTAG>_m<GMOVE>          예: success_m0.01
#   단계 = [relabel] -> subset(success arm 만) -> LoRA -> merge
#
# relabel 은 이미 던져둔 잡을 RELJOB 으로 물려받는다 (없으면 여기서 던진다).
# 산출물 이름에 arm 이 전부 들어가 조합끼리 덮어쓰지 않는다 — dexjoco 판이 relabel
# 출력과 subset 출력을 고정 이름으로 써서 guide_move 만 다른 런이 서로를 지웠다.
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   CTAG=success GMOVE=0.01 RELJOB=774 LSTEP=30000 bash sbatch/fuji/distill/chain_arm.sh
#   CTAG=all     GMOVE=0.05           LSTEP=30000 bash sbatch/fuji/distill/chain_arm.sh
#
# LSTEP 은 스모크 테스트로 잰 step 시간에서 정한다. SAVE 마다 어댑터를 남기므로
# 나중에 "몇 스텝이 최적인가" 를 재학습 없이 고를 수 있다 (lora_train.sbatch:48-52).

set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "$REPO"

EXP=${EXP:-fuji_d4r16}
CTAG=${CTAG:?CTAG 를 줘야 한다 (success | all)}
GMOVE=${GMOVE:?GMOVE 를 줘야 한다 (0.01 | 0.05 | ...)}
STEP=${STEP:-10000}          # critic 스텝
EPS=${EPS:-$CTAG}            # relabel 대상 (critic 이 본 분포와 맞춘다)
LSTEP=${LSTEP:-30000}        # LoRA 스텝
SAVE=${SAVE:-2000}           # 어댑터 저장 간격
LGPU=${LGPU:-2}              # LoRA GPU 수
LTIME=${LTIME:-24:00:00}     # LoRA 시간 한도
RELJOB=${RELJOB:-}           # 이미 던진 relabel 잡 ID (비우면 새로 던진다)

: "${MODEL_OUTPUT_DIR:?셸에서 export 할 것 — 제출 플러그인이 제출 시점 환경을 본다}"
export MODEL_OUTPUT_DIR

ARM=${CTAG}_m${GMOVE}
DSDIR=$(dirname "$(awk '/^dataset:/{print $2; exit}' configs/exp/$EXP.yaml)")
REL=$DSDIR/relabel_${EXP}__${CTAG}@$((STEP/1000))k_${EPS}_m${GMOVE}
HF=/fsx/rlwrld/junmo_cho/hf_cache

sb() { sbatch --parsable "$@"; }
echo "══════ arm $ARM  (critic $CTAG@$STEP, eps $EPS, move $GMOVE)"

# ── 1. relabel ──────────────────────────────────────────────────────────────
if [ -z "$RELJOB" ]; then
    RELJOB=$(sb --export=ALL,EXP=$EXP,CTAG=$CTAG,STEP=$STEP,EPS=$EPS,GMOVE=$GMOVE \
             sbatch/fuji/distill/relabel.sbatch)
    echo "  relabel  $RELJOB  (새로 던짐)"
else
    echo "  relabel  $RELJOB  (기존)"
fi

# ── 2. subset — success arm 만 ───────────────────────────────────────────────
# --eps success 로 relabel 하면 실패 에피소드는 원본 액션 그대로 남는다. 회복 불가능한
# 상태의 원본 액션이 BC 를 희석하므로 걸러야 한다. --eps all arm 은 전부 갈렸으니 불필요.
if [ "$EPS" = success ]; then
    SUB=$DSDIR/distill_${EXP}__${ARM}_success
    JSUB=$(sb --dependency=afterok:$RELJOB \
           --export=ALL,SRC=$REL,OUT=$SUB,EPS=success \
           sbatch/fuji/distill/subset.sbatch)
    echo "  subset   $JSUB  -> $SUB"
    LDATA=$SUB; DEP=$JSUB
else
    LDATA=$REL;  DEP=$RELJOB
    echo "  subset   건너뜀 (eps=all 이라 전부 relabel 됨)"
fi

# ── 3. LoRA distill ─────────────────────────────────────────────────────────
# wandb 는 이 단계만 켠다 (critic/relabel 은 로깅 없음). 프로젝트 rd-rl-distill.
JL=$(sb --dependency=afterok:$DEP --gres=gpu:$LGPU --time=$LTIME \
     --export=ALL,EXP=$EXP,DATA=$LDATA,TAG=$ARM,STEPS=$LSTEP,SAVE=$SAVE,WANDB=1,HF_HOME=$HF \
     sbatch/dexjoco/distill/lora_train.sbatch)
echo "  LoRA     $JL  (gpu:$LGPU, ${LSTEP}스텝, ${SAVE}마다 저장, wandb rd-rl-distill/${EXP%%_*}_distill_${ARM})"

# ── 4. merge — 추론 코드 변경 없이 읽히는 체크포인트를 만든다 ────────────────
JM=$(sb --dependency=afterok:$JL --export=ALL,EXP=$EXP,MODE=merge,TAG=$ARM,HF_HOME=$HF \
     sbatch/dexjoco/distill/subset_and_merge.sbatch)
echo "  merge    $JM  -> checkpoints/${EXP%%_*}_distill/${ARM}_merged"
echo
