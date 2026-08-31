#!/bin/bash
#SBATCH --job-name=fuji-rby1m-rh56f1-preflight-end-to-end-pipeline-smoke-test-8-episodes
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --time=2:00:00
#SBATCH --output=out/fuji-preflight_%j.out
#SBATCH --error=out/fuji-preflight_%j.err
#
# fuji 파이프라인 8단계를 8 에피소드 부분집합으로 관통시킨다 (30~40분).
# 본편(chain.sh)을 걸기 전에 이것부터 통과시킬 것.
#
#   cd <레포 루트>
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   export HF_HOME=/fsx/rlwrld/junmo_cho/hf_cache
#   sbatch --export=ALL sbatch/fuji/preflight.sbatch
#
# 로직은 sbatch/fuji/preflight.sh 한 곳에만 둔다 (srun/로그인 노드에서도 같은 것을 쓴다).

if [ -z "${MODEL_OUTPUT_DIR:-}" ]; then
    for _p in /fsx/rlwrld-unified-checkpoints /rlwrld-unified-checkpoints; do
        if [ -d "$_p/$USER" ]; then MODEL_OUTPUT_DIR="$_p/$USER/rd-rl"; break; fi
    done
fi
export MODEL_OUTPUT_DIR

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
if [ ! -d "$REPO/configs/exp" ]; then
    echo "REPO 가 레포 루트가 아니다: $REPO"
    echo "  레포 루트에서 sbatch 를 던지거나 --export=ALL,REPO=<루트> 를 줄 것"
    exit 2
fi
cd "$REPO"

echo "=== $(date -Is)  host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "    REPO=$REPO"
echo "    HF_HOME=${HF_HOME:-<미설정 → ~/.cache/huggingface>}"
bash sbatch/fuji/preflight.sh
rc=$?
echo "=== $(date -Is)  preflight 종료 코드 $rc"
exit $rc
