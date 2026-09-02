#!/bin/bash
# 3시간 srun 세션 하나에서 순차 실행한다. 세션 번호를 인자로 준다.
#
#   srun --gpus=1 --nodes=1 --wckey=project-short-name:rd --pty bash
#   cd /rlwrld2/home/junmo_cho/ws/rd-rl && bash sbatch/dexjoco/session.sh 1
#
# 각 항목은 sbatch 파일을 `bash` 로 실행한다 (#SBATCH 줄은 주석이라 무시된다).
# 예상 시간을 찍고, 남은 시간이 부족하면 시작하지 않는다 — 3시간 벽에서 잘리면
# eval 은 --resume 으로 이어지지만 critic 은 처음부터 다시라 낭비다.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
[ -d configs/exp ] || { echo "레포 루트에서 실행할 것"; exit 2; }
nvidia-smi -L >/dev/null 2>&1 || { echo "GPU 가 없다. srun 안에서 실행할 것"; exit 2; }

S=${1:?세션 번호 (1~5)}
BUDGET=${BUDGET:-170}                 # 분. 3시간에서 여유 10분
T0=$(date +%s)
left() { echo $(( BUDGET - ($(date +%s) - T0) / 60 )); }

run() {   # run <예상분> <라벨> <env...> -- <스크립트>
    local est=$1 label=$2; shift 2
    local rem; rem=$(left)
    if [ "$est" -gt "$rem" ]; then
        echo; echo "[건너뜀] $label — 예상 ${est}분 > 남은 ${rem}분"; return 0
    fi
    echo; echo "════ $(date +%H:%M) [$label]  예상 ${est}분, 남은 ${rem}분 ════"
    # 인자에 -- 가 없으면 (줄 연결이 깨진 경우 등) 여기서 명확히 죽는다.
    # set -u 아래에서 $1 unbound 로 죽으면 원인을 찾기 어렵다.
    case " $* " in *" -- "*) ;; *) echo "[오류] run 호출에 -- 가 없다: $label"; return 1 ;; esac
    local envs=() ; while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    env "${envs[@]}" bash "$@" 2>&1 | tail -25
    echo "──── $(date +%H:%M) 완료: $label"
}

A=dexjoco_hammer_nail_d2r8_s0
B=dexjoco_hammer_nail_d5r20_s0
C=dexjoco_hammer_nail_d5r20
CR=sbatch/dexjoco/critic/critic_hammer_nail.sbatch
EV=sbatch/dexjoco/rollout_w_critic/eval.sbatch

echo "세션 $S  예산 ${BUDGET}분  시작 $(date +%H:%M)"
case $S in

1)  # C critic 4개 + distillation 관문. eval 을 막고 있는 것부터.
    for T in success all; do
      run 20 "C critic $T (200K, double-Q)" EXP=$C TRAIN_EPS=$T STEPS=200000 -- $CR
      run 50 "C critic ${T}_ens (200K, REDQ 10/2)" EXP=$C TRAIN_EPS=$T STEPS=200000 NUM_QS=10 NUM_MIN_QS=2 -- $CR
    done
    run 15 "relabel dry-run (distillation 관문)" EXP=$A CTAG=success STEP=20000 EPS=success DRY=1 LIMIT=200 \
        -- sbatch/dexjoco/distill/relabel.sbatch
    run 55 "A parl@20k 200ep (89% 를 n=200 으로 확정)" EXP=$A METHOD=parl CTAG=success STEP=20000 EPISODES=200 -- $EV
    ;;

2)  # 세션 1 에서 시간 가드에 밀린 A parl 이 먼저. 논문 숫자라 우선순위가 높다.
    run 55 "A parl@20k 200ep (89% 를 n=200 으로 확정)" EXP=$A METHOD=parl CTAG=success STEP=20000 EPISODES=200 -- $EV
    run 55 "B sel32 all@100k 200ep (B 곡선 완성)" EXP=$B METHOD=sel32 CTAG=all STEP=100000 EPISODES=200 -- $EV
    run 55 "B sel32 all@200k 200ep (B 곡선 완성)" EXP=$B METHOD=sel32 CTAG=all STEP=200000 EPISODES=200 -- $EV
    ;;

2b) # 세션 2 의 나머지. 2 가 일찍 끝났을 때만.
    run 30 "B parl@1k 100ep (B 는 1K 가 최고였다)" EXP=$B METHOD=parl CTAG=all STEP=1000 EPISODES=100 -- $EV
    ;;

3)  # C 스텝 스캔 (success). 탐색이라 100ep.
    for STEP in 1000 5000 25000 100000 200000; do
      run 30 "C sel32 success@$STEP 100ep" EXP=$C METHOD=sel32 CTAG=success STEP=$STEP EPISODES=100 -- $EV
    done
    ;;

4)  # C 스텝 스캔 (all) + C 의 BC 기준선
    for STEP in 1000 5000 25000 100000 200000; do
      run 30 "C sel32 all@$STEP 100ep" EXP=$C METHOD=sel32 CTAG=all STEP=$STEP EPISODES=100 -- $EV
    done
    run 30 "C BC 기준선 100ep (수집 앞200ep 51.5% 와 대조)" EXP=$C METHOD=bc EPISODES=100 -- $EV
    ;;

5)  # A·B 곡선의 빈 구간 (20K 봉우리와 100K 붕괴 사이). 체크포인트는 이미 있다.
    for E in $A $B; do for STEP in 25000 50000; do
      run 30 "$(basename $E) sel32 success@$STEP 100ep" EXP=$E METHOD=sel32 CTAG=success STEP=$STEP EPISODES=100 -- $EV
    done; done
    ;;

6)  # 방법 축: parl(선택+상승) 이 sel32(선택만) 을 B·C 에서도 넘는가.
    # sel32@1k 는 B·C 둘 다 이미 있으므로 parl 만 채우면 직접 페어 비교가 완성된다.
    # critic 은 1K 로 고정한다 — 롤아웃 성공률로 체크포인트를 고르지 않기 위해서이고,
    # B 는 AUC 가 1K 에서 이미 1.000 으로 포화한다 (25-step return 이라 전이가 빠르다).
    # guide_move 0.05 는 차원 수에 불변이다 (총 이동 = move x sqrt(d), 차원당 RMS = move).
    for T in success all; do
      run 45 "B parl $T@1k 200ep" EXP=$B METHOD=parl CTAG=$T STEP=1000 EPISODES=200 -- $EV
    done
    for T in success all; do
      run 25 "C parl $T@1k 100ep" EXP=$C METHOD=parl CTAG=$T STEP=1000 EPISODES=100 -- $EV
    done
    ;;

7)  # C 의 guide_move 스윕. 0.05 에서 54%->29% (p=0.0008) 로 망가졌으므로
    # 안전한 폭을 찾는다. A 에서도 0.01~0.05 가 정점이고 0.1 에서 꺾였다.
    # 랜덤씬은 critic 외삽 위험이 크므로 A 보다 작은 값이 맞을 것으로 본다.
    for T in success all; do
      for GM in 0.005 0.01 0.02; do
        run 25 "C parl gm=$GM $T@1k 100ep" \
            EXP=$C METHOD=parl CTAG=$T STEP=1000 EPISODES=100 GUIDE_MOVE=$GM -- $EV
      done
    done
    ;;

7b) # 위가 3시간에 안 들어가면 success 만 먼저.
    for GM in 0.005 0.01 0.02; do
      run 25 "C parl gm=$GM success@1k 100ep" \
          EXP=$C METHOD=parl CTAG=success STEP=1000 EPISODES=100 GUIDE_MOVE=$GM -- $EV
    done
    ;;

8)  # 격자의 빈 칸: A 의 all x parl. 3세팅 x 2필터 x 2방법이 완성된다.
    run 55 "A parl all@20k 200ep" EXP=$A METHOD=parl CTAG=all STEP=20000 EPISODES=200 -- $EV
    ;;

9)  # C 를 20K critic 으로 다시. 1K 는 0.44 epoch 로 데이터를 절반도 못 봤다
    # (C 는 288K 프레임으로 A·B 의 5배라 같은 스텝이 완전히 다른 학습량이다).
    #   C 1K = 0.44 ep,  C 20K = 8.9 ep,  A 1K = 2.2 ep,  A 20K = 43.7 ep
    # 두 가설이 갈린다:
    #   학습부족이 원인 → 20K 에서 gm 0.05 의 붕괴(29%)가 고쳐진다
    #   랜덤씬 외삽이 원인 → 20K 에서도 gm 0.05 는 나쁘고 작은 gm 만 작동한다
    # sel32 도 같은 스텝에서 재서 parl vs sel32 페어 비교가 되게 한다.
    for T in success all; do
      run 15 "C sel32 $T@20k 100ep" EXP=$C METHOD=sel32 CTAG=$T STEP=20000 EPISODES=100 -- $EV
    done
    for T in success all; do
      for GM in 0.05 0.005; do
        run 30 "C parl gm=$GM $T@20k 100ep" \
            EXP=$C METHOD=parl CTAG=$T STEP=20000 EPISODES=100 GUIDE_MOVE=$GM -- $EV
      done
    done
    ;;

10) # C 를 100K critic 으로. **A 의 최적점과 학습량이 같은 지점이다:**
    #   A 20K = 43.67 epoch  (85.0%, p=1e-6)
    #   C 100K = 44.37 epoch  ← 여기
    # 학습량을 A 와 맞추면 남는 차이는 **씬 랜덤화 하나**뿐이므로,
    # "랜덤씬 자체가 critic 을 무력화하는가" 를 깨끗하게 가릴 수 있다.
    # 세션 9 의 20K(8.9 ep)는 그 중간 지점이다.
    for T in success all; do
      run 15 "C sel32 $T@100k 100ep" EXP=$C METHOD=sel32 CTAG=$T STEP=100000 EPISODES=100 -- $EV
    done
    for T in success all; do
      for GM in 0.05 0.005; do
        run 30 "C parl gm=$GM $T@100k 100ep" \
            EXP=$C METHOD=parl CTAG=$T STEP=100000 EPISODES=100 GUIDE_MOVE=$GM -- $EV
      done
    done
    ;;

11) # **과적합 critic 에서 parl 이 망가지는가** — B 로 검증한다.
    # B 를 고른 이유: parl 이 확실히 작동하는 세팅이고(1K 에서 87.5%, p=1e-4),
    # sel32 곡선이 U자라 200K 가 20K 보다 유의하게 좋았다(+14.0pp, p=0.0018).
    # 그 U자가 sel32 의 특성인지 critic 의 특성인지 parl 로 다시 훑어 가른다.
    #
    # 스텝 스캔 20개를 전부 sel32 로 한 것이 이 실험의 설계 결함이었다 — B 에서
    # sel32 는 10개 스텝 전부 무의미했는데 parl 은 p=1e-4 였으므로, sel32 곡선은
    # "critic 품질" 이 아니라 "후보 다양성이 언제 덜 부족한가" 를 잰 것일 수 있다.
    #
    # 1K 는 이미 있다 (87.5%). 20K/100K/200K 를 채우면 parl 의 스텝 곡선이 완성된다.
    for S in 200000 100000 20000; do
      run 45 "B parl success@$S 200ep" EXP=$B METHOD=parl CTAG=success STEP=$S EPISODES=200 -- $EV
    done
    run 45 "B parl all@200k 200ep (필터 교차 확인)" EXP=$B METHOD=parl CTAG=all STEP=200000 EPISODES=200 -- $EV
    ;;

12) # guide_move 곡선 — B, critic **success@20K**.
    # 세션 11 이 스텝 축(20K/100K/200K)을 훑으므로 gm 축은 20K 에 고정해서 십자로 만든다.
    # 양 끝에서 무슨 일이 일어나는지 본다:
    #   gm -> 0   : 상승 폭이 0 이면 후보가 안 바뀌고, top-10 의 argmax = 전체 argmax
    #               이므로 **parl 은 수학적으로 sel32 와 같아진다**. B sel32@20K 는
    #               63.5% 였으므로 그 근처가 나와야 한다 (파이프라인 정합성 검사).
    #   gm 크게   : 액션이 분포 밖으로 밀려 critic 이 외삽하고, keep-best 는 같은
    #               critic 으로 판정하므로 자기 오차를 못 잡는다. A 에서 0.1 -> 68%,
    #               0.2 -> 33% 였고 C 에서는 0.05 만으로 29% 로 무너졌다.
    # gm=0.05 는 세션 11 의 parl__success@20k 가 만든다 — 여기서는 돌리지 않는다.
    for GM in 0 0.001 0.005 0.01 0.02 0.1 0.2; do
      run 22 "B parl gm=$GM success@20k 100ep" \
          EXP=$B METHOD=parl CTAG=success STEP=20000 EPISODES=100 GUIDE_MOVE=$GM -- $EV
    done
    ;;

13) # 3시간 종합: 진단 2개(롤아웃 불필요) + 빈칸 3종.
    # research_guidance.md 6절의 검증 순서 1~3 + A 의 gm 곡선 완성.
    mkdir -p out
    PY=third_party/RLDX-1/.venv/bin/python
    export PYTHONPATH="$PWD/third_party/RLDX-1:$PWD"

    echo; echo "════ 진단 1: 앙상블 critic 의 std 가 OOD 신호가 되나 (15분) ════"
    echo "  10헤드 REDQ 가 2헤드보다 나은가. shuffled 에서 10배 이상 올라야 쓸 수 있다."
    echo "  (2헤드 실측: A 3.2배 / B 3.2배 / C 1.9배 — 사실상 노이즈)"
    for T in success success_ens all all_ens; do
      $PY -u -m rl.probe_actsens --exp dexjoco_hammer_nail_d5r20 \
          --data rl-dataset/dexjoco/hammer_nail_d5r20 --checkpoints checkpoints \
          --critic "$T/critic_001000.pt" --features cogfeat.npy 2>&1 \
          | tee -a out/actsens_C_ens.log | grep -E "격차|shuffled|표본|critic\]" | sed "s/^/  [$T] /"
    done

    echo; echo "════ 진단 2: guide_steps 가 경로에 영향을 주나 (10분) ════"
    echo "  총 이동거리는 gm x sqrt(d) 로 스텝 수와 무관하다. 스텝 수는 경로 해상도다."
    echo "  Q 가 국소 선형이면 조기 정지(신뢰영역)의 해상도 이득이 작다."
    $PY -u sim/dexjoco/probe_steps.py --exp dexjoco_hammer_nail_d2r8_s0 \
        --critic success/critic_020000.pt --steps 1,2,4,10,20 --moves 0.02,0.05,0.1 2>&1 \
        | tee out/probe_steps_A.log | tail -25

    echo; echo "════ 빈칸 1: C 앙상블 critic 평가 — 학습만 하고 한 번도 안 재봤다 ════"
    for M in sel32 parl; do
      run 25 "C $M success_ens@1k 100ep" EXP=$C METHOD=$M CTAG=success_ens STEP=1000 EPISODES=100 -- $EV
    done

    echo; echo "════ 빈칸 2: C 의 BC 노이즈 바닥 + 시드 정렬 검증 ════"
    echo "  수집 데이터 앞 100ep 의 52.0% 를 재현하면 씬 시드가 맞는 것이다."
    run 25 "C BC 100ep" EXP=$C METHOD=bc EPISODES=100 -- $EV

    echo; echo "════ 빈칸 3: A 의 gm 곡선 — 0.05 하나만 있어 붕괴 지점을 모른다 ════"
    echo "  B 는 0.2 에서, C 는 0.05 에서 무너졌다. A 는 그 사이 어딘가일 것이다."
    for GM in 0.1 0.2; do
      run 30 "A parl gm=$GM success@20k 100ep" \
          EXP=$A METHOD=parl CTAG=success STEP=20000 EPISODES=100 GUIDE_MOVE=$GM -- $EV
    done
    ;;

14) # 오라클 — gradient ascent 의 **천장**을 잰다. critic 없이 시뮬레이터로 고른다.
    #
    # 왜: C 는 어떤 조합도 BC 를 유의하게 넘지 못했다(최고 63%, p=0.061). 그것이
    #   (a) critic 이 후보를 못 고르는 것인지  (b) 애초에 고를 것이 없는 것인지 모른다.
    # 오라클은 후보를 실제로 20스텝 굴려 보고 못 삽입량으로 고른다 — critic 오차가 0 이다.
    #
    # --sigma 가 우리 guide_move 에 대응한다 (둘 다 차원당 이동 크기다). 그래서
    #   sigma=0.01 에서 오라클 90%  →  critic 이 병목. 온도/신뢰영역 조절이 값어치 있다
    #   sigma=0.01 에서 오라클 65%  →  그 반경 안에 답이 없다. N 을 늘리거나 정책을 바꿔야 한다
    # sigma=0 은 선택이 없는 것과 같아 BC 재현이 되어야 한다 (검증용).
    #
    # 비용: 결정마다 후보 32개 x 20스텝 = 640 env.step 추가 (렌더링은 끈다).
    # C 는 에피소드당 결정 14.4회 → 약 9,200 스텝 추가로 정상 에피소드의 32배 물리다.
    # **실측이 없어 먼저 20 에피소드로 재고, 그 속도로 본격 규모를 정한다.**
    mkdir -p out
    P=${PORT:-$((20000 + ${SLURM_JOB_ID:-$$} % 9000))}
    CK=$PWD/checkpoints/$(awk '/^base_policy:/{print $2; exit}' configs/exp/$C.yaml)
    export HF_HOME=${HF_HOME:-/workspace/junmo_cho/hf_cache}
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
    export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
    export PYTHONPATH="$PWD/third_party/RLDX-1:$PWD"
    echo "  정책 서버 (critic 없이 순정 BC — 오라클이 후보를 스스로 만든다) port $P"
    third_party/RLDX-1/.venv/bin/python -u -m rl.vla_rldx serve \
        --exp $C --model-path "$CK" --rtc-inference-mode trained --rtc-exec-horizon 20 \
        --sim-wrapper --log-every 50 --host 127.0.0.1 --port $P > out/oracle_srv_$$.log 2>&1 &
    trap "kill $! 2>/dev/null" EXIT
    echo "  서버 기동 대기 — oracle_select 의 wait_ready 가 알아서 기다린다"
    command sleep 20

    PYTHONPATH="$PWD/third_party/dexjoco/dexjoco" MUJOCO_GL=egl \
      /workspace/junmo_cho/dexjoco/venv/bin/python -u sim/dexjoco/oracle_select.py \
        --task hammer_nail --port $P --episodes ${OEP:-20} --seed 0 --fixed-scene -1 \
        --replan 20 --rtc-delay 5 --max-steps 360 --n-cand 32 \
        --sigma 0,0.005,0.01,0.02,0.05 \
        --out $PWD/rl-dataset/dexjoco/oracle_d5r20 2>&1 | tee out/oracle_C.log | tail -40
    ;;

*)  echo "세션 번호는 1~14"; exit 2 ;;
esac

echo; echo "════ 세션 $S 종료  $(date +%H:%M)  남은 예산 $(left)분 ════"
python3 sim/dexjoco/summarize_evals.py 2>/dev/null | tail -30
