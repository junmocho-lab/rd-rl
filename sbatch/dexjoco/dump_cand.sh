#!/bin/bash
# srun 세션에서 돌리는 용도. GPU 1개를 잡고 순서대로 4개 조건을 돌린 뒤 바로 분석한다.
#
#   srun --gpus=1 --nodes=1 --wckey=project-short-name:rd --pty bash
#   bash sbatch/dexjoco/dump_cand.sh
#
# 하는 일: B(d5r20_s0) / critic success@20k / parl 로 롤아웃하면서, 결정 프레임마다
# BC 후보 32개 + 서버가 실제로 내보낸 액션을 npz 로 남긴다. guide_move 만 바꿔 4번.
# 성공률을 재는 게 아니다 (그건 100ep 로 이미 있다) — 후보 구름과 상승 액션을
# **같은 상태에서** 나란히 놓고 재기 위한 표본 수집이다.
#
# 조건 (괄호는 기존 100ep 실측 성공률):
#   gm=0    (69%)  대조군. chosen 이 후보 중 하나 -> z_u 가 ~2 로 나와야 측정이 맞다
#   gm=0.02 (79%)  완만한 구간
#   gm=0.1  (96%)  최고점
#   gm=0.2  (28%)  붕괴
#
# 환경변수: GMS(조건 목록) EPISODES DUMP_N
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO"

GMS=${GMS:-"0 0.02 0.1 0.2"}
EPISODES=${EPISODES:-40}      # 실측 결정 약 12회/에피소드. 실패 에피소드 표본을
                              # 확보하려면 25 로는 얇다 (gm=0.1 에서 3개뿐이었다)
DUMP_N=${DUMP_N:-450}         # 40ep = 결정 약 480회. 임계값은 그보다 낮게 둔다
                              # (넘겨도 서버가 dump_n/4 마다 덮어써서 파일은 남는다)
EXP=${EXP:-dexjoco_hammer_nail_d5r20_s0}
CTAG=${CTAG:-success}; STEP=${STEP:-20000}

D=$REPO/rl-dataset/dexjoco/eval_$EXP
start=$(date +%s)
for gm in $GMS; do
    tag="gm$(echo "$gm" | tr -d '.')"
    npz=$D/dump_${tag}__dump.npz
    if [ -f "$npz" ]; then echo "[건너뜀] $tag — npz 이미 있다"; continue; fi
    echo; echo "########## $tag  ($(( ($(date +%s)-start)/60 ))분 경과) ##########"
    EXP=$EXP CTAG=$CTAG STEP=$STEP METHOD=parl GUIDE_MOVE=$gm \
    EPISODES=$EPISODES DUMP_N=$DUMP_N NAME=dump_$tag \
        bash sbatch/dexjoco/rollout_w_critic/eval.sbatch 2>&1 | tail -20
done

echo; echo "########## 분석 ##########"
shopt -s nullglob
NPZ=($D/dump_gm*__dump.npz)
if [ ${#NPZ[@]} -eq 0 ]; then
    echo "[오류] npz 가 하나도 없다. 서버 로그에서 '[덤프]' 줄을 확인할 것:"
    echo "       grep -h '\[덤프\]' out/*.out 2>/dev/null | tail"
    exit 1
fi
ls -la "${NPZ[@]}"
third_party/RLDX-1/.venv/bin/python sim/dexjoco/cand_vs_guided.py "${NPZ[@]}"
