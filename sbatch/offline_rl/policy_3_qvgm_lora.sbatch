#!/bin/bash
#SBATCH --job-name="3/4 qvgm_lora — stepwise IQL critic + value-gradient matching (LoRA)"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=rlwrld
#SBATCH --time=24:00:00
#SBATCH --wckey=project-short-name:rd
#SBATCH --output=/rlwrld2/home/junmo_cho/ws/rd-rl/out/policy_3_qvgm_lora_%j.out
#SBATCH --error=/rlwrld2/home/junmo_cho/ws/rd-rl/out/policy_3_qvgm_lora_%j.err
# MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/junmo_cho/ sbatch ./sbatch/offline_rl/policy_3_qvgm_lora.sbatch

cd /rlwrld2/home/junmo_cho/ws/rd-rl/
source configs/kakao_path.sh
set -euo pipefail

DATA=rl-dataset/0825_openarm_f1_inference
export PYTHONPATH="$PWD/third_party/RLDX-1:$PWD"
export NO_ALBUMENTATIONS_UPDATE=1
PY=third_party/RLDX-1/.venv/bin/python

# (1) Q-VGM 식 critic. offline_iql 과 다른 파일이다:
#       stepwise Q^(i)/V^(i) (청크 안에서 다음 위치로 부트스트랩) / 층마다 액션 재주입 /
#       Q 헤드 2개 min / 스칼라(분포형 아님). 실측 7ms/step → 100k 약 12분.
#     ** 4/4 잡이 이 critic 을 기다린다. 이 잡을 먼저 던질 것.
$PY -u -m rl.offline_iql_qvgm \
  --exp openarm_rim --data $DATA --checkpoints checkpoints \
  --features cogfeat.npy \
  --discount 0.995 --expectile 0.7 --num-qs 2 \
  --steps 100000 --batch 128 --holdout 0.1 --eval-every 20000 --wandb-run qvgm-critic

# (2) value-gradient matching — velocity field 를 직접 가르친다 (Q-VGM Eq. 5-9).
#     v_base 는 LoRA adapter 를 끈 같은 가중치. 디노이징 체인 미분 없음.
#     실측 1.30s/step (batch 16) → 30,000 스텝 약 11시간.
#     타깃 ĥ_Q 가 매 스텝 다시 만들어지므로 parl 보다 고정라벨 과적합 위험이 낮다.
$PY -u -m rl.train_policy --method qvgm \
  --exp openarm_rim --data $DATA --checkpoints checkpoints \
  --critic qvgm-cog-t07-g0995-q2-s0-h01/critic_latest.pt \
  --flow-steps 10 --late-steps 5 --ascent-steps 4 --auto-step 0.05 --clip-grad 1.0 \
  --steps 30000 --batch 16 --lr 1e-4 --lora-rank 16 --lora-alpha 32 \
  --holdout 0.1 --eval-every 3000 --save-every 3000 --wandb-run qvgm-lora

echo "[done 3/4] checkpoints/openarm_rim-policy/qvgm-lora16-lr0.0001-K10M5J4-s0/lora_latest.pt"
