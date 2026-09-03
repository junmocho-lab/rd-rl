#!/bin/bash
#SBATCH --job-name=fuji-rby1m-rh56f1-relabel-parl-action-optimization
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --time=3:00:00   # 실측 3.6 결정/s. eps=all 이면 179050/16 = 11190 결정 -> ~52분,
                         # eps=success 는 207/290 이라 ~40분. 요청 시간을 실측에 맞춰야
                         # backfill 이 빈 슬롯에 끼워 넣는다.
#SBATCH --output=out/fuji-relabel_%j.out
#SBATCH --error=out/fuji-relabel_%j.err
#
# PA-RL 액션 최적화로 데이터셋의 action 컬럼을 갈아끼운다 (action distillation 1단계).
#   후보 32 -> Q(mean) 상위 10 + 로그액션 1 -> ∇_A Q 상승 10스텝(raw, keep-best) -> argmax
#
# ── dexjoco 판(sbatch/dexjoco/distill/relabel.sbatch)과 다른 점 ──────────────
# **출력 경로에 GMOVE 를 넣는다.** dexjoco 판은
#     relabel_${EXP}__${CTAG}@${STEP}k_${EPS}
# 로 끝나서 guide_move 만 다른 두 런이 **같은 디렉토리를 덮어쓴다**. npy 도
# --npy-out 기본값(parl_actions.npy)이 고정이라 한 파일을 공유한다.
# step size 를 쓸어보려면 둘 다 갈라야 한다.
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   sbatch --export=ALL,EXP=fuji_d4r16_0902_fuji_rc_success,CTAG=success_ens_full,\
#            STEP=25000,SSIZE=3 sbatch/fuji/distill/relabel.sbatch
#
# EPS 를 비우면 CTAG 에 맞춘다 (success critic -> success 에피소드만). critic 이 본
# 분포 밖은 외삽이라 맞춰 주는 게 안전하다 (relabel_parl --eps 도움말).
#
# 판정 기준: **로그된 액션이 이긴 비율 < 70%**. 넘으면 critic 이 base policy 를 못 이겨
# distillation 이 no-op 이 된다. 2026-08-31 dry-run 실측은 10~12% 로 넉넉히 통과했다.
#
# guide_move 근거 (2026-08-31 측정):
#   fuji 의 1프레임 자연 변화량 = 0.0051 (정규화 액션 단위, right_arm_joints).
#   openarm 은 0.022 이므로 fuji 가 4.3배 작다 — dexjoco 에서 89% 를 낸 0.05 는
#   openarm 기준 2.3 프레임치였고, fuji 에서 같은 크기는 0.012 다.
#     0.01 = 2.0 프레임치   (실기 서빙에서 안정)
#     0.05 = 9.8 프레임치   (실기 서빙에서 0.02 부터 흔들렸다)
#   dry-run 지표(ΔQ / A>0 / 개선비율)는 0.005~0.2 전 구간에서 **단조 증가**해 최적점을
#   못 준다 — ‖g‖/√d 가 1.38~1.41e-4 로 불변이어서 Q 가 액션에 선형이기 때문이다.
#   그래서 값은 실기 관측과 스케일 정합으로 정하고, 이 실험으로 relabel 쪽 답을 본다.

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

EXP=${EXP:-fuji_d4r16}
CTAG=${CTAG:-success}
STEP=${STEP:-10000}
EPS=${EPS:-}                   # 비우면 CTAG 에 맞춘다
# ── 상승 파라미터 (2026-09-02 재설계) ──────────────────────────────────────
# 예전에는 GMOVE(=--guide-move) 로 g/‖g‖ 정규화 후 **총 이동거리를 고정**했다.
# 그것은 critic 이 평평한 곳에서도 같은 거리를 밀어붙여 PA-RL 원본보다 공격적이다.
# 지금은 원본과 같은 raw gradient 를 쓰고 --step-size 로 조절한다.
#
# SSIZE 근거 (utils/probe_rtc_actions.py, critic success_ens_full@25k, 6ep/250결정):
#   |∂Q/∂a| = 4.5e-4.  자연변화 1프레임 = 0.00624.  BC 자체 샘플 산포 = 0.00437.
#     ss    상승이동    nat배   raw/관절   ΔQ->프레임   판정
#     0.3   0.00143     0.23    0.19°      +0.75f     BC 잡음보다 작다 -> 학습 불가
#     1     0.00469     0.75    0.60°      +1.17f     BC 잡음과 같은 크기
#     3     0.01370     2.20    1.75°      +2.24f   <- 채택 (잡음의 3.1배)
#     10    0.04347     7.0     3.0°       +5.15f     너무 큼
#     100   0.39792     64      22°        +81.9f     OOD 폭주 (critic 헛소리)
#   PA-RL 원본 3e-4 가 우리에게서 아무 일도 안 하는 이유는 순수하게 Q 스케일이다
#   (원본 [-100,0] vs 우리 [0,1] -> 4627배).
SSIZE=${SSIZE:-3}
GMOVE=${GMOVE:-0}              # >0 이면 옛 정규화 방식으로 돌아간다 (비권장)
GSTEPS=${GSTEPS:-10}           # raw gradient 라 스텝마다 |g| 를 다시 재므로 의미가 있다
NCAND=${NCAND:-32}
KEEP=${KEEP:-10}
TEMP=${TEMP:-0}                # 0 = argmax (PA-RL distill_argmax=True 와 동일).
                               # >0 이면 Categorical(Q/temp) — 우리 Q 산포가 1e-4
                               # 수준이라 온도를 잘못 주면 균등추첨이 된다
LIMIT=${LIMIT:-0}
DRY=${DRY:-0}
[ -n "$EPS" ] || EPS=$CTAG

DATA=$(awk '/^dataset:/{print $2; exit}' configs/exp/$EXP.yaml)
DSDIR=$(dirname "$DATA")
CRITIC=$REPO/checkpoints/${EXP}-critic/${CTAG}/$(printf 'critic_%06d.pt' "$STEP")
[ -f "$CRITIC" ] || { echo "critic 없음: $CRITIC"; exit 3; }

# ★ 상승 파라미터를 경로와 npy 이름에 넣는다 (dexjoco 판과 갈라지는 지점).
#   안 넣으면 step_size 만 다른 두 런이 같은 디렉토리와 같은 npy 를 덮어쓴다.
SUF="${CTAG}@$((STEP/1000))k_${EPS}_ss${SSIZE}n${GSTEPS}"
OUT=$REPO/$DSDIR/relabel_${EXP}__${SUF}
NPY=parl_actions__${SUF}.npy

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
export HF_HOME=${HF_HOME:-/fsx/rlwrld/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
PY=$REPO/third_party/RLDX-1/.venv/bin/python

echo "=== $(date -Is)  relabel  $EXP  critic=$CTAG@$STEP  eps=$EPS  ss=$SSIZE"
echo "    data   $DATA"
echo "    후보 $NCAND -> keep $KEEP -> 상승 ${GSTEPS}스텝 (raw, step_size=$SSIZE) -> temp $TEMP"
echo "    out    $OUT"
echo "    npy    checkpoints/${EXP}-critic/$NPY"

$PY -u -m rl.relabel_parl \
    --exp "$EXP" --data "$DATA" --checkpoints checkpoints \
    --critic "$CRITIC" --features cogfeat.npy \
    --eps "$EPS" \
    --num-samples "$NCAND" --num-keep "$KEEP" \
    --num-steps "$GSTEPS" --step-size "$SSIZE" --guide-move "$GMOVE" --temp "$TEMP" \
    --out "$OUT" --npy-out "$NPY" \
    $([ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT") \
    $([ "$DRY" = 1 ] && echo "--dry-run") || exit 1
echo "=== $(date -Is)  완료  $OUT"
