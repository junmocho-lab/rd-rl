#!/bin/bash
# fuji 파이프라인 전체를 **8 에피소드 부분집합**으로 관통시킨다. 30~40분.
#
# 밤새 도는 긴 잡을 던지기 전에 이걸 먼저 돌린다. 잡히는 것들:
#   · 경로/venv/HF 캐시 문제        · modality·관절 순서 불일치
#   · 액션 차원 계산 오류            · qvgm critic 로딩
#   · relabel 의 관절 순열 왕복      · LoRA 인자 (modality-config / horizon / rtc-delay)
#   · LoRA 병합
#
# GPU 1장 있는 곳에서 (srun 안이든 로그인 노드든) 그냥 실행한다:
#   bash sbatch/fuji/preflight.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
[ -d configs/exp ] || { echo "레포 루트에서 실행할 것"; exit 2; }
ROOT=$(pwd)                     # 아래 서브셸에서 상대경로를 쓰지 않기 위해 한 번만 잡는다

EXP=fuji_preflight
SRC=rl-dataset/fuji/0831_fuji_all/rby1m_rh56f1_inference_s180_20260829_234913
MINI=rl-dataset/fuji/_preflight
PY=third_party/RLDX-1/.venv/bin/python

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/third_party/RLDX-1:$PWD"
[ -f "$HOME/.rldx_secrets.sh" ] && source "$HOME/.rldx_secrets.sh"

# 0~6 단계는 8 에피소드라도 모델을 네 번 올려 20분쯤 걸린다. 앞 단계가 통과했으면
# 산출물을 그대로 두고 건너뛴다:
#     FROM=7 bash sbatch/fuji/preflight.sh      # LoRA 학습/병합만 다시
# 건너뛴 단계의 산출물이 없으면 그 자리에서 멈춘다.
FROM=${FROM:-0}

step() { echo; echo "════════ $* ════════"; }
skip() { echo; echo "──────── $* — 건너뜀 (FROM=$FROM) ────────"; }
die()  { echo; echo "✗ 실패: $*"; echo "  여기서 멈춘다. 위 로그를 보고 고친 뒤 다시 실행."; exit 1; }

# 7·8 단계가 함께 쓰는 값. 게이트 밖에 둬야 FROM=8 로 병합만 돌릴 때도 정의된다.
BASE=$ROOT/checkpoints/$(awk '/^base_policy:/{print $2; exit}' configs/exp/$EXP.yaml)
MODCFG=$(awk '/^rldx_data_config:/{print $2; exit}' configs/exp/$EXP.yaml)
AH=$(awk '/^action_horizon:/{print $2; exit}' configs/exp/$EXP.yaml)
RTCD=$($PY -c "import json;print(json.load(open('$BASE/config.json'))['rtc_training_max_delay'])")
# embodiment tag 는 BC 학습이 쓴 값과 같아야 한다 (dexjoco=general, fuji=new).
# conf.yaml 은 소문자 값을, CLI 는 enum 이름(대문자)을 쓴다.
EMB=$(grep -m1 -oP '^\s*embodiment_tag:\s*\K\S+' "$BASE/experiment_cfg/conf.yaml" 2>/dev/null | tr 'a-z' 'A-Z')
EMB=${EMB:-GENERAL_EMBODIMENT}

# 건너뛴 단계의 산출물이 실제로 있는지 먼저 확인한다. 없는 채로 진행하면
# 엉뚱한 곳에서 죽어 원인 찾기가 어렵다.
if [ "$FROM" -gt 0 ]; then
    echo "[FROM=$FROM] 앞 단계를 건너뛴다. 선행 산출물 확인:"
    chk() { if [ -e "$1" ]; then echo "  ✓ $1"; else
        echo "  ✗ $1 이 없다 — FROM 을 낮춰 다시 돌릴 것"; exit 1; fi; }
    [ "$FROM" -gt 1 ] && chk "$MINI/meta" 
    [ "$FROM" -gt 2 ] && chk "configs/exp/$EXP.yaml"
    [ "$FROM" -gt 3 ] && chk "checkpoints/$EXP-critic/cogfeat.npy"
    [ "$FROM" -gt 4 ] && chk "checkpoints/$EXP-critic/smoke/critic_latest.pt"
    [ "$FROM" -gt 6 ] && chk "${MINI}_relabel"
    [ "$FROM" -gt 7 ] && chk "checkpoints/fuji_distill/_preflight"
    true
fi

if [ "$FROM" -le 0 ]; then
step "0. 백본이 오프라인으로 풀리나"
$PY -c "
from transformers import AutoConfig, AutoProcessor
AutoConfig.from_pretrained('RLWRLD/RLDX-1-VLM'); AutoProcessor.from_pretrained('RLWRLD/RLDX-1-VLM')
print('  ✓ RLDX-1-VLM config/processor 해결됨')" || die "HF 캐시에 RLDX-1-VLM 이 없다.
  HF_TOKEN 을 넣고 HF_HUB_OFFLINE=0 으로 한 번 받거나, HF_HOME 을 캐시 있는 곳으로 지정할 것"

else
skip "0. 백본이 오프라인으로 풀리나"
fi

if [ "$FROM" -le 1 ]; then
step "1. 8 에피소드 미니 데이터셋 (성공/실패 섞어서)"
rm -rf "$MINI"
EPS=$($PY - <<PYEOF
import pandas as pd, glob, json
ok, ng = [], []
for f in sorted(glob.glob("$SRC/data/*/*.parquet")):
    d = pd.read_parquet(f, columns=["next.success","episode_index"])
    (ok if d["next.success"].to_numpy().astype(bool).any() else ng).append(int(d["episode_index"].iloc[0]))
print(",".join(str(x) for x in ok[:4] + ng[:4]))
PYEOF
) || die "에피소드 고르기"
echo "  고른 에피소드: $EPS"
$PY rl/make_subset.py --data "$SRC" --out "$MINI" --episodes "$EPS" || die "make_subset"

else
skip "1. 8 에피소드 미니 데이터셋 (성공/실패 섞어서)"
fi

if [ "$FROM" -le 2 ]; then
step "2. 임시 exp yaml"
sed -e "s|^name: .*|name: $EXP|" -e "s|^dataset: .*|dataset: $MINI|" \
    configs/exp/fuji_d3r8.yaml > configs/exp/$EXP.yaml || die "yaml 생성"
$PY -c "
import yaml; d=yaml.safe_load(open('configs/exp/$EXP.yaml'))
print(f\"  name={d['name']} dataset={d['dataset']} groups={d['explore_groups']}\")"

else
skip "2. 임시 exp yaml"
fi

if [ "$FROM" -le 3 ]; then
step "3. cogfeat 추출 (8 에피소드)"
$PY -u -m rl.extract_cogfeat --exp $EXP --data "$MINI" --checkpoints checkpoints --batch 16 --resume \
    || die "extract_cogfeat — 모달리티/카메라/모델 로딩 문제"

else
skip "3. cogfeat 추출 (8 에피소드)"
fi

if [ "$FROM" -le 4 ]; then
step "4. critic 학습 200 스텝 — 액션 차원이 77 로 찍혀야 한다"
$PY -u -m rl.offline_iql_qvgm --exp $EXP --data "$MINI" --checkpoints checkpoints \
    --features cogfeat.npy --action-groups right_arm_joints \
    --train-eps all --discount 0.998 --expectile 0.8 --num-qs 2 --bins 128 --q-range 0,1 \
    --latent 2048 --state-latent 256 --hidden 1024,512 --no-stepwise \
    --steps 200 --holdout 0.25 --eval-every 100 --keep-last 2 --keep-steps 100 --no-wandb \
    --tag smoke || die "critic 학습"

else
skip "4. critic 학습 200 스텝 — 액션 차원이 77 로 찍혀야 한다"
fi

if [ "$FROM" -le 5 ]; then
step "5. relabel dry-run 8 결정 — 관절 순열 왕복까지 확인된다"
$PY -u -m rl.relabel_parl --exp $EXP --data "$MINI" --checkpoints checkpoints \
    --critic checkpoints/${EXP}-critic/smoke/critic_latest.pt --features cogfeat.npy \
    --num-samples 4 --num-keep 2 --num-steps 2 --guide-move 0.05 --temp 0.001 \
    --dry-run --limit 8 || die "relabel_parl — critic 로딩 / 액션 인덱스 / 후보 생성"

else
skip "5. relabel dry-run 8 결정 — 관절 순열 왕복까지 확인된다"
fi

if [ "$FROM" -le 6 ]; then
step "6. relabel 본편 (8 에피소드) — parquet 재작성 + 순열 검증"
$PY -u -m rl.relabel_parl --exp $EXP --data "$MINI" --checkpoints checkpoints \
    --critic checkpoints/${EXP}-critic/smoke/critic_latest.pt --features cogfeat.npy \
    --num-samples 4 --num-keep 2 --num-steps 2 --guide-move 0.05 --temp 0.001 \
    --out "${MINI}_relabel" || die "relabel 본편 — parquet 쓰기 / 관절 순열"

else
skip "6. relabel 본편 (8 에피소드) — parquet 재작성 + 순열 검증"
fi

if [ "$FROM" -le 7 ]; then
step "7. LoRA 학습 5 스텝 — 인자가 fuji 에 맞는지"

# LeRobot 데이터셋 경로 목록을 만든다.
#   dexjoco : 루트 자체가 하나의 데이터셋      → 1개
#   fuji    : 루트 아래에 세션 디렉토리 여러 개 → N개
# launch_train 은 --dataset-paths (복수) 가 --dataset-path 보다 우선한다.
collect_datasets() {
    local root=$1
    if [ -f "$root/meta/info.json" ]; then
        echo "$root"
    else
        local d
        for d in "$root"/*/; do
            [ -f "$d/meta/info.json" ] && echo "${d%/}"
        done
    fi
}
mapfile -t DPATHS < <(collect_datasets "$ROOT/${MINI}_relabel")
echo "  학습 데이터셋 ${#DPATHS[@]}개: ${DPATHS[*]}"
echo "  modality=$MODCFG horizon=$AH rtc_max_delay=$RTCD embodiment=$EMB"
( cd third_party/RLDX-1 && .venv/bin/python -m rldx.experiment.launch_train \
    --base-model-path "$BASE" --dataset-paths "${DPATHS[@]}" \
    --embodiment-tag "$EMB" --modality-config-path "$MODCFG" \
    --video-length 1 --n-cog-tokens 64 --action-horizon "$AH" --rtc-training-max-delay "$RTCD" \
    --action-model-use-lora --save-trainable-only \
    --save-only-model \
    --no-tune-projector --no-tune-llm --no-tune-visual \
    --global-batch-size 4 --learning-rate 1e-4 --max-steps 5 --save-steps 5 \
    --num-gpus 1 --dataloader-num-workers 2 \
    --output-dir "$ROOT/checkpoints/fuji_distill/_preflight" \
    --experiment-name fuji_preflight ) || die "LoRA 학습 — launch_train 인자"

else
skip "7. LoRA 학습 5 스텝 — 인자가 fuji 에 맞는지"
fi

if [ "$FROM" -le 8 ]; then
step "8. LoRA 병합"
# launch_train 은 산출물을 <output-dir>/<experiment-name>/checkpoint-N 으로 한 단계
# 더 중첩한다. 그래서 output-dir 바로 아래를 보면 안 된다. 스텝 번호로 정렬해 마지막을 고른다.
CK=$(find "checkpoints/fuji_distill/_preflight" -maxdepth 2 -type d -name 'checkpoint-*' -printf '%f\t%p\n' 2>/dev/null \
     | sed 's/^checkpoint-//' | sort -n | tail -1 | cut -f2)
[ -n "$CK" ] || die "LoRA 체크포인트가 안 나왔다 (checkpoints/fuji_distill/_preflight 아래)"
echo "  체크포인트 $CK  ($(du -sh "$CK" | cut -f1))"
( cd third_party/RLDX-1 && .venv/bin/python scripts/merge_lora_checkpoint.py \
    --trainable-ckpt "$ROOT/$CK" --base-ckpt "$BASE" \
    --output "$ROOT/checkpoints/fuji_distill/_preflight_merged" ) || die "merge_lora_checkpoint"

else
skip "8. LoRA 병합"
fi

echo
echo "════════ 전부 통과 ════════"
echo "파이프라인 8단계가 끝까지 돌았다. 이제 본편을 걸어도 된다:"
echo "    bash sbatch/fuji/chain.sh"
echo
echo "정리 (본편 전에 지울 것 — 미니 캐시가 본편 이름과 겹치지는 않지만 용량을 먹는다):"
echo "    rm -rf $MINI ${MINI}_relabel checkpoints/${EXP}-critic checkpoints/fuji_distill/_preflight*"
echo "    rm -f configs/exp/$EXP.yaml"
