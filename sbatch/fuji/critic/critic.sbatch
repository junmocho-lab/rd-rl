#!/bin/bash
#SBATCH --job-name=fuji-rby1m-rh56f1-critic-qvgm-iql-rightarm-step-grid
#SBATCH --partition=dedicated-rd
#SBATCH --wckey=project-short-name:rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue   # preempt 되면 조용히 재시작하지 말고 죽어라
#SBATCH --time=2:00:00 # 200K 는 ~25분. 요청 시간을 실측에 맞춰야 backfill 이 잘 끼워 넣는다
#SBATCH --output=out/fuji-critic_%j.out
#SBATCH --error=out/fuji-critic_%j.err
#
# fuji (rby1m_rh56f1) 전용 critic 학습. dexjoco 판(sbatch/dexjoco/critic/
# critic_hammer_nail.sbatch)에서 갈라져 나왔다 — 그쪽 하드코딩이 fuji 에서 두 번
# 사고를 냈기 때문이다. 갈라진 지점은 아래 "dexjoco 와 다른 점" 참조.
#
#   export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
#   sbatch --export=ALL,EXP=fuji_d3r8,TRAIN_EPS=success sbatch/fuji/critic/critic.sbatch
#   sbatch --export=ALL,EXP=fuji_d3r8,TRAIN_EPS=all     sbatch/fuji/critic/critic.sbatch
#
# 두 잡은 서로 의존이 없어 동시에 던져도 된다. 단 cogfeat 캐시를 공유하므로
# sbatch/dexjoco/critic/cogfeat.sbatch 를 EXP 당 한 번 먼저 완주시킬 것.
#
# ── dexjoco 와 다른 점 (전부 실패로 확인된 것들) ────────────────────────────
# 1. --action-groups: dexjoco 는 eef_position,eef_rotation 을 하드코딩했는데 fuji 는
#    관절 공간이라 그런 그룹이 없다. 잡 700/701 이 여기서 즉사했다:
#      "--action-groups eef_position,eef_rotation 에 맞는 그룹이 없다
#       (가능: torso_joints, right_arm_joints, ... , neck_joints)"
#    여기서는 exp yaml 의 explore_groups 에서 읽는다. relabel_parl.py:108 /
#    train_policy.py:128 이 쓰는 값과 같아야 한다 — critic 이 본 액션 부분공간과
#    relabel 이 편집하는 부분공간이 어긋나면 Q 상승이 엉뚱한 축을 민다.
#    fuji: right_arm_joints → 창 (latency+replan)=11 스텝 x 7관절 = 77차원.
#
# 2. --discount: dexjoco 는 0.995 를 하드코딩했다. fuji 는 에피소드가 평균 652 프레임
#    이라 0.995 면 γ^T = 0.038 로 초반 V 가 사실상 0 이 된다. exp yaml 의
#    expo.discount(0.998 → 0.271)가 그 동적 범위를 fuji 길이에 맞춘 값이다.
#    offline_iql_qvgm.py 는 yaml 의 expo.discount 를 읽지 않으므로(읽는 것은
#    replan_steps / inference_latency / action_horizon / base_policy /
#    rldx_data_config 뿐) 여기서 명시적으로 넘겨야 한다.
#
# 3. cogfeat 완료 판정: dexjoco 는 "$DATA/meta/info.json" 을 읽었는데 fuji 의
#    $DATA 는 세션들을 담은 **루트 디렉토리**라 그 파일이 없다. 그래서 판정이 늘
#    실패하고 매번 extract_cogfeat 를 불러 VLA 백본을 헛되이 올렸다 (잡당 ~1분,
#    게다가 동시 제출 시 같은 캐시를 함께 연다). 여기서는 세션들의 total_frames 를
#    합산하고 cogfeat.json 의 세션 목록까지 대조한다.
#
# 4. 파티션 rlwrld -> dedicated-rd, 로그 out/dexjoco-critic_* -> out/fuji-critic_*.

# 제출 플러그인이 MODEL_OUTPUT_DIR 를 요구하는데 머신마다 허용 접두사가 다르다.
# 이 잡은 거기에 아무것도 쓰지 않는다 (산출물은 전부 $REPO 아래). 플러그인 통과용이다.
# 플러그인은 **제출 시점의 환경**을 보므로 sbatch 로 던질 때는 셸에서 미리 export 할 것.
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

EXP=${EXP:-fuji_d3r8}
YAML=configs/exp/$EXP.yaml
[ -f "$YAML" ] || { echo "exp yaml 이 없다: $YAML"; exit 2; }

DATA=$(awk '/^dataset:/{print $2; exit}' "$YAML")        # 데이터셋은 yaml 이 정한다
TRAIN_EPS=${TRAIN_EPS:-success}                          # success | all | fail
STEPS=${STEPS:-200000}
# 격자를 하나로 합쳐 둔다 — STEPS 를 넘는 값은 그냥 안 생긴다.
# (sbatch --export 는 쉼표로 항목을 자르므로 쉼표를 쓰는 값은 이 파일 안에 둔다)
KEEP_STEPS=1000,5000,20000,25000,50000,100000,200000,500000
KEEP_LAST=${KEEP_LAST:-3}
NUM_QS=${NUM_QS:-2}                                      # 2 = clipped double Q (Q-VGM)
NUM_MIN_QS=${NUM_MIN_QS:-0}                              # >0 이면 REDQ (예: 10 / 2)
HOLDOUT=${HOLDOUT:-0.1}
# TAG 가 저장 디렉토리(<exp>-critic/<TAG>/)를 정하므로, **설정이 다르면 TAG 도 달라야**
# 한다. 안 그러면 이전 체크포인트를 조용히 덮어써서 비교가 불가능해진다.
#   _ens   REDQ (num_min_qs > 0)
#   _full  홀드아웃 없음 = 전 데이터 학습 (평가는 in-sample)
case "$HOLDOUT" in 0|0.0|none|off|"") _HO_SFX=_full ;; *) _HO_SFX="" ;; esac
TAG=${TAG:-${TRAIN_EPS}$([ "$NUM_MIN_QS" -gt 0 ] && echo _ens)${_HO_SFX}}

# yaml 이 정본이다. 덮어쓰려면 --export 로 ACTION_GROUPS / DISCOUNT 를 준다.
ACTION_GROUPS=${ACTION_GROUPS:-$(python3 -c "
import yaml,sys
e=yaml.safe_load(open('$YAML'))
g=e.get('explore_groups') or []
print(','.join(g))")}
DISCOUNT=${DISCOUNT:-$(python3 -c "
import yaml
e=yaml.safe_load(open('$YAML'))
print((e.get('expo') or {}).get('discount', 0.995))")}
[ -n "$ACTION_GROUPS" ] || { echo "explore_groups 가 $YAML 에 없다. ACTION_GROUPS 를 직접 줄 것"; exit 2; }

export PYTHONPATH="$REPO/third_party/RLDX-1:$REPO"
# 기본값이 이 클러스터 경로다. dexjoco 판은 /workspace/... 를 기본으로 두는데
# fsx 머신엔 그 경로가 없고 HF_HUB_OFFLINE=1 이라 모델 로딩에서 죽는다.
export HF_HOME=${HF_HOME:-/fsx/rlwrld/junmo_cho/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
PY=$REPO/third_party/RLDX-1/.venv/bin/python

echo "=== $(date -Is)  critic  $EXP / $TAG"
echo "    data=$DATA  steps=$STEPS  qs=$NUM_QS/$NUM_MIN_QS  holdout=$HOLDOUT"
echo "    action-groups=$ACTION_GROUPS  discount=$DISCOUNT  keep=$KEEP_STEPS"

# ── cogfeat 캐시 검증 ──────────────────────────────────────────────────────
# 프레임 수 / 세션 목록 / **base 정책**까지 대조한다. 손상 세션을 빼거나 base 를
# 갈아끼우면 캐시는 무효다 — 특히 base 는 프레임 수가 그대로라 안 보면 못 잡는다.
COG=checkpoints/${EXP}-critic/cogfeat.json
BASE_POLICY=checkpoints/$(awk '/^base_policy:/{print $2; exit}' "$YAML")
VERDICT=$($PY - "$DATA" "$COG" "$BASE_POLICY" <<'PY'
import json, os, sys
root, cog, base = sys.argv[1], sys.argv[2], sys.argv[3]
if os.path.exists(os.path.join(root, "meta/info.json")):          # 단일 데이터셋
    sess = [os.path.basename(root.rstrip("/"))]
    frames = json.load(open(os.path.join(root, "meta/info.json")))["total_frames"]
else:                                                              # 세션 루트
    sess, frames = [], 0
    for d in sorted(os.listdir(root)):
        ij = os.path.join(root, d, "meta/info.json")
        if os.path.exists(ij):
            sess.append(d)
            frames += json.load(open(ij))["total_frames"]
if not sess:
    print(f"NODATA {root} 아래에 LeRobot 세션이 없다"); raise SystemExit
if not os.path.exists(cog):
    print(f"MISSING cogfeat.json 이 없다 (필요 {frames} 프레임 / 세션 {len(sess)}개)"); raise SystemExit
c = json.load(open(cog))
if c.get("done") != c.get("T"):
    print(f"PARTIAL {c.get('done')}/{c.get('T')} 프레임"); raise SystemExit
if c.get("T") != frames:
    print(f"FRAMES 캐시 {c.get('T')} != 데이터 {frames}"); raise SystemExit
if list(c.get("sessions", [])) != sess:
    print(f"SESSIONS 캐시 {len(c.get('sessions', []))}개 != 데이터 {len(sess)}개"); raise SystemExit
# ★ cogfeat 은 base 정책의 백본에서 뽑은 feature 다. 정책이 바뀌면 프레임 수와 세션이
#   같아도 **완전히 다른 값**이 된다. 이 검사가 없으면 base_policy 만 갈아끼웠을 때
#   낡은 cogfeat 을 조용히 재사용해 엉뚱한 critic 이 나온다.
if c.get("model") and os.path.normpath(c["model"]) != os.path.normpath(base):
    print(f"MODEL 캐시 base={c['model']} != yaml base_policy={base}"); raise SystemExit
print(f"OK {frames} 프레임 / 세션 {len(sess)}개")
PY
)
echo "[cogfeat] $VERDICT"
case "$VERDICT" in
    OK*)     : ;;                                  # 그대로 쓴다
    NODATA*) exit 2 ;;
    *)  # extract_cogfeat 는 캐시가 꽉 차 있어도 VLA 백본을 올리느라 ~1분을 쓴다.
        # 여기 오는 것은 캐시가 실제로 모자랄 때뿐이다.
        echo "[cogfeat] 추출한다 — 여러 critic 잡을 동시에 던졌다면 지금 캐시 경쟁이 난다."
        echo "[cogfeat] cogfeat.sbatch 를 먼저 한 번 완주시키는 편이 안전하다."
        $PY -u -m rl.extract_cogfeat --exp "$EXP" --data "$DATA" \
            --checkpoints checkpoints --batch 64 --resume || exit 1 ;;
esac

# ── critic 학습 ────────────────────────────────────────────────────────────
$PY -u -m rl.offline_iql_qvgm --exp "$EXP" --data "$DATA" \
    --checkpoints checkpoints --features cogfeat.npy \
    --action-groups "$ACTION_GROUPS" \
    --train-eps "$TRAIN_EPS" \
    --discount "$DISCOUNT" --expectile 0.8 --num-qs "$NUM_QS" --num-min-qs "$NUM_MIN_QS" \
    --bins 128 --q-range 0,1 \
    --latent 2048 --state-latent 256 --hidden 1024,512 --no-stepwise \
    --steps "$STEPS" --holdout "$HOLDOUT" --eval-every 1000 \
    --keep-last "$KEEP_LAST" --keep-steps "$KEEP_STEPS" --no-wandb \
    --tag "$TAG" || exit 1

RUN=checkpoints/${EXP}-critic/${TAG}
echo "=== 남은 체크포인트"; ls "$RUN" | grep '^critic_[0-9]' | sort
# 이 스크립트는 래퍼(critic_<dataset>.sbatch)에서 bash 로 불릴 수도 있어서 로그
# 파일명이 out/fuji-critic_<잡ID>.out 이 아닐 수 있다. 없으면 조용히 넘어간다 —
# 여기서 grep 이 exit 2 를 내면 **학습이 다 끝났는데 잡이 FAILED 로 기록된다**
# (실측: 잡 1005/1006). || true 가 그것을 막는다.
echo "=== AUC (critic 이 성패를 구분하나)"
_LOG=$(ls -t "out/fuji-critic"*"_${SLURM_JOB_ID:-x}.out" 2>/dev/null | head -1)
if [ -n "$_LOG" ]; then grep -h "AUC" "$_LOG" | tail -5; else echo "  (로그 파일을 못 찾음)"; fi
true
