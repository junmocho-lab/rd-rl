#!/bin/bash
#SBATCH --job-name="2/4 iqlcritic_edit — iql critic + EXPO residual(edit) policy. PA-RL 이 아니다"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=rlwrld
#SBATCH --time=24:00:00
#SBATCH --wckey=project-short-name:rd
#SBATCH --output=/rlwrld2/home/junmo_cho/ws/rd-rl/out/policy_2_iqlcritic_edit_%j.out
#SBATCH --error=/rlwrld2/home/junmo_cho/ws/rd-rl/out/policy_2_iqlcritic_edit_%j.err
# MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/junmo_cho/ sbatch ./sbatch/offline_rl/policy_2_iqlcritic_edit.sbatch

cd /rlwrld2/home/junmo_cho/ws/rd-rl/
source configs/kakao_path.sh
set -euo pipefail

# ** 이 잡은 velocity field 를 가르치지 않는다.** DiT 는 영구 동결이고 그 출력에
# 유계 보정(edit_scale)을 얹는 EXPO-FT 방식이다. Q-VGM 논문이 자기 baseline 으로 둔
# 'Test-time Q guidance' 계열(88.7 vs 92.5). 여기서 재는 것은 **critic 설계가 더 좋은
# grad_A Q 를 주는가** 이고, 추출은 값싼 방법으로 읽어내는 것이다.

DATA=rl-dataset/0825_openarm_f1_inference
export PYTHONPATH="$PWD/third_party/RLDX-1:$PWD"
export NO_ALBUMENTATIONS_UPDATE=1
PY=third_party/RLDX-1/.venv/bin/python

# base 청크 캐시(75MB)가 없으면 먼저 만든다 (VLA 로 8후보 샘플, ~11분, CPU 병목). 그 뒤 학습 루프에는
# VLA 가 들어가지 않는다 — 0.5M MLP + 캐시 텐서뿐이라 초당 수천 스텝이다.
# 캐시는 4/4 와 공유된다 (base policy 가 같으므로). 원자적 교체라 동시 실행에도 안전.
$PY -u -m rl.train_policy --method edit \
  --exp openarm_rim --data $DATA --checkpoints checkpoints \
  --critic iql-cog-dist128-t07-g0995-q10all-s0-h01/critic_latest.pt \
  --base-chunks 8 --cache-batch 128 --steps 30000 --batch 256 \
  --holdout 0.1 --eval-every 3000 --save-every 3000 --wandb-run iqlcritic-edit

echo "[done 2/4] checkpoints/openarm_rim-policy/edit-iql-*/edit_latest.pt"
