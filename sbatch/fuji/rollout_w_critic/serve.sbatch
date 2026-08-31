#!/bin/bash
#SBATCH --job-name=fuji-rby1m-rh56f1-serve-policy-with-critic
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --time=4:00:00
#SBATCH --output=out/fuji-serve_%j.out
#SBATCH --error=out/fuji-serve_%j.err
#
# serve.sh 를 클러스터에서 띄우는 껍데기.
#
# ★ 먼저 읽을 것 — 이게 쓸모 있으려면 **로봇이 이 계산 노드에 TCP 로 닿아야 한다**.
#   fuji 는 실기라 에피소드를 돌리는 주체가 rrc(로봇 워크스테이션)이고, 이 잡은
#   정책 서버만 띄운다. dexjoco 판은 같은 잡 안에 시뮬레이터가 있어서 127.0.0.1 로
#   충분했지만 여기는 아니다.
#   계산 노드는 사설 대역(172.31.x.x)이라, 로봇이 같은 VPC/VPN 안에 없으면 못 붙는다.
#   그 경우 클러스터가 아니라 **로봇 워크스테이션에서 serve.sh 를 직접 돌리는 것이
#   정답이다** (README_cp.md 7번이 그렇게 하고 있다 — --host 127.0.0.1, rrc 와 같은 머신).
#   아래에서 노드 IP 를 찍어주니, rrc 쪽에서 그 주소로 닿는지 먼저 확인할 것.
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   sbatch --export=ALL,EXP=fuji_d3r8,METHOD=parl,CTAG=all,STEP=100000 \
#          sbatch/fuji/rollout_w_critic/serve.sbatch
#
# 서버는 --time 까지 떠 있다가 죽는다. 롤아웃이 끝나면 scancel 로 내릴 것.

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

# 로봇이 붙어야 하므로 루프백이 아니라 모든 인터페이스에 연다.
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-5555}

echo "=== 노드 $(hostname)"
echo "    이 주소로 rrc 가 붙어야 한다:"
hostname -I | tr ' ' '\n' | grep -v '^$' | sed 's/^/      /'
echo "    포트 $PORT"
echo "    (닿지 않으면 로봇 워크스테이션에서 serve.sh 를 직접 돌릴 것)"

exec bash sbatch/fuji/rollout_w_critic/serve.sh
