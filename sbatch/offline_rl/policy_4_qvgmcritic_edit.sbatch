#!/bin/bash
#SBATCH --job-name="4/4 qvgmcritic_edit — stepwise IQL critic + EXPO residual(edit). Q-VGM 이 아니다"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=rlwrld
#SBATCH --time=24:00:00
#SBATCH --wckey=project-short-name:rd
#SBATCH --output=/rlwrld2/home/junmo_cho/ws/rd-rl/out/policy_4_qvgmcritic_edit_%j.out
#SBATCH --error=/rlwrld2/home/junmo_cho/ws/rd-rl/out/policy_4_qvgmcritic_edit_%j.err
# MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/junmo_cho/ sbatch ./sbatch/offline_rl/policy_4_qvgmcritic_edit.sbatch

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

CRITIC=checkpoints/openarm_rim-critic/qvgm-cog-t07-g0995-q2-s0-h01/critic_latest.pt

# 3/4 잡이 만드는 critic 을 기다린다 (실측 12분). 최대 90분.
# 3/4 를 먼저 던지거나, critic 을 직접 먼저 돌려 두면 즉시 진행한다.
for i in $(seq 1 180); do
  [ -f "$CRITIC" ] && break
  [ $i -eq 1 ] && echo "[대기] 3/4 잡의 qvgm critic 을 기다린다: $CRITIC"
  sleep 30
done
if [ ! -f "$CRITIC" ]; then
  echo "[실패] 90분을 기다렸는데 critic 이 없다. 3/4 잡이 돌고 있는지 확인할 것." >&2
  exit 1
fi
echo "[대기 완료] critic 발견"

# base 청크 캐시(74MB)를 공유한다. 2/4 가 이미 만들었으면 재사용, 없으면 여기서 만든다
# (~11분). 이 단계는 CPU 병목이라 --cache-batch 를 키워도 안 빨라진다
# (rldx 가 샘플별 파이썬 루프로 이미지를 전처리한다 — GPU 가동률 0%).
# tmp 파일명에 pid 가 들어가므로 두 잡이 동시에 만들어도 안전하다.
# 학습 루프에는 VLA 가 들어가지 않는다 — 0.5M MLP + 캐시 텐서.
$PY -u -m rl.train_policy --method edit \
  --exp openarm_rim --data $DATA --checkpoints checkpoints \
  --critic qvgm-cog-t07-g0995-q2-s0-h01/critic_latest.pt \
  --base-chunks 8 --cache-batch 128 --steps 30000 --batch 256 \
  --holdout 0.1 --eval-every 3000 --save-every 3000 --wandb-run qvgmcritic-edit

echo "[done 4/4] checkpoints/openarm_rim-policy/edit-qvgm-*/edit_latest.pt"
