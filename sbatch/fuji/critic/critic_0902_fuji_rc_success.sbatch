#!/bin/bash
#SBATCH --job-name=fuji-critic-0902_fuji_rc_success
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --time=6:00:00   # cogfeat ~1.5h + critic 25k x 2 (각 ~5분). 실측에 맞춰야
                         # backfill 이 빈 슬롯에 끼워 넣는다.
#SBATCH --output=out/fuji-critic-0902_fuji_rc_success_%j.out
#SBATCH --error=out/fuji-critic-0902_fuji_rc_success_%j.err
#
# 0902_fuji_rc_success 데이터셋으로 critic 을 학습한다.
#
# ★ --train-eps success 만 돌린다. 이 데이터셋은 **전부 성공 에피소드**라
#   all 을 돌려도 같은 데이터로 같은 critic 이 나온다 (중복 낭비).
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   sbatch sbatch/fuji/critic/critic_0902_fuji_rc_success.sbatch
#
# ── 성공만 있는 데이터의 함의 ────────────────────────────────────────────────
# 홀드아웃에 실패 에피소드가 없으므로 **AUC 를 계산할 수 없다**
# (offline_iql_qvgm.py:445 가 성공/실패 쌍을 세는데 실패 쪽이 비어 nan 이 된다).
# 즉 "몇 스텝이 최적인가" 를 AUC 로 못 고른다. 대신 이것들을 본다:
#   · Q 곡선이 참값 γ^(T-t) 를 따라가는가        → utils/critic_grid.py 비디오
#   · 앙상블 std (액션 민감도)                    → critic_grid 의 표
#   · relabel dry-run 의 "로그된 액션이 이긴 비율" → 70% 미만이어야 의미가 있다
# 0831 데이터는 fail_cases 세션이 있어 AUC 가 나왔지만 여기는 다르다.
#
# 파라미터는 전부 configs/exp/fuji_d4r16_0902_fuji_rc_success.yaml 이 정한다
# (dataset / explore_groups / expo.discount / replan / latency).
# 덮어쓰려면 --export 로 STEPS / KEEP_LAST / NUM_QS / HOLDOUT 등을 준다.

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
[ -d "$REPO/configs/exp" ] || { echo "REPO 가 레포 루트가 아니다: $REPO"; exit 2; }
cd "$REPO"

export EXP=fuji_d4r16_0902_fuji_rc_success
export STEPS=${STEPS:-25000}
export KEEP_LAST=${KEEP_LAST:-0}      # 0 이면 정리 안 함 → 2000 마다 전부 보존
# ── 앙상블: PA-RL 원본과 동일한 REDQ ────────────────────────────────────────
#   critic_ensemble_size=10, critic_subsample_size=2  (configs/base_config.py:123-124)
#   타깃을 앙상블 전체가 아니라 매 스텝 무작위 2개에서 min 으로 잡는다.
# 이전 clipped double Q (2/0) 는 불일치를 잴 표본이 2개뿐이라 앙상블 std 가 작았고
# (실측 0.0002~0.009), 그것이 relabel 에서 후보를 줄 세울 변별력의 병목이었다.
export NUM_QS=${NUM_QS:-10}
export NUM_MIN_QS=${NUM_MIN_QS:-2}
# ── 홀드아웃: 없음 (전 데이터 학습) ─────────────────────────────────────────
# 이 데이터셋은 **전부 성공**이라 홀드아웃이 사는 게 거의 없다: AUC 가 성공/실패 쌍을
# 세는데 실패 쪽이 비어 nan 이다 (잡 1067 로그 실측). 그런데 대가는 실제로 든다 —
# 0.1 이면 350ep 중 35ep 이 빠지고 그중 13ep 이 오늘 모은 HiL 이었다.
# 그래서 학습에는 전부 쓰고, 진단(Q 곡선 / 앙상블 std / 발산 여부)은 학습 에피소드
# 중 --eval-frac 만큼을 골라 **in-sample** 로 본다.
# ★ in-sample 수치는 홀드아웃보다 반드시 좋게 나온다 — 과적합 탐지에는 쓸 수 없다.
#   로그와 플롯에 'in-sample' 이 붙어 나오니 그렇게 읽을 것.
# 정직한 홀드아웃을 원하면 HiL 을 건드리지 않는 teleop 한 세션을 쓰면 된다:
#   sbatch --export=ALL,HOLDOUT=teleop_dataset_B_dark ...   (20ep, 5.7%, HiL 0ep)
export HOLDOUT=${HOLDOUT:-0}
export HF_HOME=${HF_HOME:-/fsx/rlwrld/junmo_cho/hf_cache}

[ -f "configs/exp/$EXP.yaml" ] || { echo "exp yaml 이 없다: configs/exp/$EXP.yaml"; exit 2; }
DATA=$(awk '/^dataset:/{print $2; exit}' "configs/exp/$EXP.yaml")
if [ ! -d "$DATA" ] || [ -z "$(ls -A "$DATA" 2>/dev/null)" ]; then
    echo "데이터셋이 비었거나 없다: $DATA"
    echo "  (다운로드가 끝났는지 확인할 것)"
    exit 3
fi

echo "########## $(date -Is)  critic 학습 시작"
echo "  EXP  $EXP"
echo "  DATA $DATA"
echo "  캐시 checkpoints/$EXP-critic/"
echo "  steps=$STEPS keep_last=$KEEP_LAST (2000 마다 저장)"
echo "  train-eps=success 만 (데이터가 전부 성공이라 all 은 중복)"
echo "  앙상블 REDQ  num_qs=$NUM_QS  num_min_qs=$NUM_MIN_QS  (PA-RL 원본과 동일)"
echo "  홀드아웃 $HOLDOUT  ->  $([ "$HOLDOUT" = 0 ] && echo '전 데이터 학습, 평가는 in-sample' || echo "$HOLDOUT")"

TRAIN_EPS=success bash sbatch/fuji/critic/critic.sbatch || exit 1
echo
echo "########## $(date -Is)  완료 — checkpoints/$EXP-critic/success/"
