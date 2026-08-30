#!/bin/bash
#SBATCH --job-name=fuji-rby1m-rh56f1-submit-full-pipeline-chain-cogfeat-critic-distill
#SBATCH --partition=rlwrld
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
#SBATCH --time=0:10:00
#SBATCH --output=out/fuji-chain-submit_%j.out
#SBATCH --error=out/fuji-chain-submit_%j.err
#
# chain.sh 는 잡을 "제출만" 하는 스크립트다 (GPU 를 안 쓴다). 로그인 노드에서
#   bash sbatch/fuji/chain.sh
# 로 직접 돌리는 것이 정상이고, 그게 막혀 있을 때만 이 래퍼를 쓴다.
#
#   sbatch --export=ALL sbatch/fuji/chain.sbatch
#
# 주의: 이 잡 안에서 sbatch 를 다시 부르므로, 클러스터가 잡 안에서의 제출을 막아
# 두었다면 실패한다. 그 경우 chain.sh 가 출력하는 명령을 손으로 던질 것.

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
bash sbatch/fuji/chain.sh
