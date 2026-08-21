# rd-rl

지표와 튜닝은 [METRICS.md](METRICS.md) — 학습 중에는 wandb `rd-rl-expo` 프로젝트를 본다.

# 실행 순서

창을 3개띄운다 (학습서버, 로컬 데이터 전송, 로컬 모델 열기)

0. rd-rl git clone 받고 각 submodule 및 .venv 세팅 
1. paths 세팅 후 `source ./configs/paths.sh`, modality 파일 추가 
2. teleop data 세팅 (post-training에 활용) 학습서버 t0 지우고 시작. kubernetes 이슈.
   ```
   uv run ./utils/convert_data.py ../rrc-rollout-example/test/t0 -o ./rl-dataset --modality modality/rby1m_rh56f1/modality.json
   ```
3. teleop data 학습 서버에 업로드
   ```
   kubectl cp $A_DS/t0 $L_POD:$L_DS/t0
   ```
4. bc policy (bc policy는 학습서버에서 다운로드) rollout 진행 및 위처럼 2. 처럼 inference data 세팅

   먼저 **base BC 정책만** 서빙한다 (RLDX 원본 서버. edit policy·critic 이 없는 순수 BC — 7번의 EXPO 서버와 다르다). 밑에 명령어 execution length 추가해야되나?
   ```
   cd third_party/RLDX-1
   ROS_DOMAIN_ID=106 pixi run -e rldx python -u -m rldx.eval.run_rldx_server     --model-path $A_CKPT/rldx-img-curated/rldx_img_curated-0810-0818-r03/     --embodiment-tag NEW_EMBODIMENT     --host 127.0.0.1 --port 5555     --rtc-inference-mode trained     --rtc-inference-delay 3
   ```
   `python rldx/eval/run_rldx_server.py` 가 아니라 `-m` 인 이유: 이 서브모듈의 pixi 환경에는
   rldx 가 설치돼 있지 않아 스크립트로 부르면 `ModuleNotFoundError: rldx` 가 난다
   (`-m` 은 cwd 를 sys.path 에 넣는다).

   `--rtc-inference-delay` 는 rrc 의 `inference_latency_steps` 와 같아야 한다
   (openarm 2 / fuji·CJL 3 — `configs/exp/<실험>.yaml` 의 `inference_latency`).

   실측 (로컬 5090, ZMQ 왕복 20회): 평균 118ms/회. 7번의 EXPO 서버는 같은 조건에서
   121ms 였다 — 후보 8개를 뽑는 비용이 3ms 밖에 안 되는 건 백본이 시간을 다 쓰기 때문이다.

   그리고 롤아웃한 데이터를 변환한다.
   ```
   uv run ./utils/convert_data.py ../rrc-rollout-example/test/r0 -o ./rl-dataset --modality modality/rby1m_rh56f1/modality.json
   ```
5. inference data 학습 서버에 업로드
   ```
   kubectl cp $A_DS/r0 $L_POD:$L_DS/r0
   ```

6. learner 상시 잡 띄우기 (**한 번만**. 라운드마다 다시 띄우지 않는다)
   ```
   ./actor/start_learner.sh fuji
   ```
   `run id = openarm_rim_<날짜-시각>` 이 `$A_RUNS/CURRENT` 에 적히고 이후 명령들이 그걸 기본값으로
   읽는다. 잡은 폴링을 시작하기 전에 **θ₀ 를 만들어 내보낸다**:

   ```
   $L_CKPT/expo/<run id>/init/{theta.pt, meta.json, DONE}      184MB
   ```

   θ₀ = 비전 인코더 + critic 앙상블 + target critic + edit policy + temperature.
   `configs/exp/<이름>.yaml` 의 `modality` / `rldx_data_config` / `expo` 블록으로 만들고,
   base 정책 체크포인트와 순서를 교차검증한다 (어긋나면 여기서 죽는다). `meta.json` 에
   seed·sha256·torch 버전·코드 SHA 가 남는다. Job 이 재시작해도 `init/DONE` 이 있으면
   **다시 만들지 않는다** — actor 가 롤아웃 중인 θ₀ 를 바꿔치기하면 안 되니까.

   그리고 `$L_RUNS/<run id>/` 메일박스를 5초마다 폴링한다.
   잡 이름(`junmo-cho-rdrl-openarm-rim-<날짜-시각>`)과 로그·중단 명령은 스크립트가 출력해준다.
   ```
   kubectl -n $L_NS logs -f job/<잡 이름>                              # 로그
   kubectl -n $L_NS exec $L_POD -- tail -20 $L_RUNS/<run id>/learner.log
   kubectl -n $L_NS delete job <잡 이름>                               # 중단
   ```

6b. θ₀ 를 로컬로 받는다 (**round 0 롤아웃 전에 한 번**)
   ```
   uv run ./actor/recv_round.py --round init
   ```
   → `$A_CKPT/expo/<run id>/init/` (learner 와 같은 상대경로). 이걸 받아야 actor 가
   **learner 가 학습을 시작하는 것과 같은 파라미터로** round 0 을 돌 수 있다. 안 받고
   띄우면 서버가 그 자리에서 랜덤 초기화하는데, 그러면 round 0 을 무엇으로 모았는지
   기록할 수 없다 (off-policy 라 학습이 틀리는 건 아니지만 재현이 안 된다).

   ── 여기부터 7~10을 라운드마다 반복 ──

7. **EXPO** 정책 서버 띄우기 (rrc 의 ZMQ 클라이언트가 붙는다. 4번의 순수 BC 서버와 달리
   base 정책 + edit policy + critic 을 함께 서빙한다)
   ```
   cd third_party/RLDX-1
   ROS_DOMAIN_ID=106 PYTHONPATH="$PWD:$A_RL" pixi run -e rldx python -u -m rl.vla_rldx serve     --exp fuji     --model-path $A_CKPT/rldx-img-curated/rldx_img_curated-0810-0818-r03/     --artifacts $A_CKPT/expo/fuji_20260821-073310/init/theta.pt     --host 127.0.0.1 --port 5555
   ```
   `--model-path` 는 **base BC 정책**이고 라운드가 지나도 바뀌지 않는다 (13.8GB. 3번에서
   받은 그것). 라운드마다 바뀌는 것은 `--artifacts` 뿐이다:

   | 라운드 | `--artifacts` | 들어있는 것 |
   |---|---|---|
   | 0 | `init/theta.pt` | 인코더 + critic + target + edit policy + temperature |
   | 1~ | `r<NNN>/theta.pt` | 위 + **학습된 action expert LoRA** |

   round 0 에 LoRA 가 없는 이유: PEFT 의 `lora_B` 가 0 초기화라 주입 직후 델타가
   정확히 0 이다 (실측 `0.00e+00`). 즉 **round 0 의 VLA 출력은 base BC 와 같고**,
   정책을 바꾸는 건 critic argmax + edit 뿐이다. 13.8GB 를 올려 0 을 저장할 이유가 없어
   learner 가 θ₀ 에서 빼고, 서버 로더는 "있는 키만 채운다" 라서 이 차이가 코드 경로를
   나누지 않는다. round 0 이 랜덤 critic·랜덤 residual 로 도는 것은 EXPO-FT 의 warmup 과
   같은 조건이고, edit 이 액션을 흔들어주는 덕에 탐색 데이터가 생긴다 (성공률은 base BC 보다 낮다).

   `action_horizon` / `replan_steps` / `inference_latency` / `explore_groups` / EXPO 값은
   전부 `--exp` 의 yaml 에서 읽는다. RTC 는 `--rtc-inference-mode trained` (기본) 이고 지연은
   yaml 의 `inference_latency` 를 쓴다 — RLDX 원본 서버의
   `--rtc-inference-delay` 를 따로 줄 필요가 없다.

   RLDX 원본(`rldx/eval/run_rldx_server.py`) 과 다른 점은 청크를 **고르는** 단계 하나뿐이다:
   ```
   unbatch → 프로세서 → RTC prefix 주입 → [추론] → RTC 캐시 → 디코드
                                          ↑ 여기서 백본 1회 + 디노이저 N회 → 후보 N개
                                            + edit 후보 → target critic argmax
   ```
   실측 (로컬 5090, openarm base 정책, N=8+edit 8): **첫 회 639ms, 이후 121ms/회** (예산 400ms).
   `[EXPO] #12 88ms 후보 8+8 → 11 (edit) Q=+0.017 후보간 Q std=0.001` 형태로 라운드마다
   선택 상태가 로그에 남는다. **후보간 Q std** 가 이 루프의 핵심 지표다 (아래 참고).

8. 롤아웃 + 사람이 에피소드별 성공/실패 라벨 (rrc-release) → 2번처럼 변환
   ```
   uv run ./utils/convert_data.py ../rrc-rollout-example/test/r<N> -o ./rl-dataset --modality modality/rby1m_rh56f1/modality.json
   ```

9. 라운드 전송 (세션별로 올린 뒤 **맨 마지막에** READY)
   ```
   uv run ./actor/send_round.py --round <N> \
       --dataset ./rl-dataset/r<N>/<데이터셋>/<세션>_320x192 \
       --collected-by base          # 이후 라운드는 r000, r001, ... (어느 정책이 모았는지)
   ```
   `--dataset` 에 부모 디렉토리를 주면 그 안의 세션들로 펼쳐진다. 같은 라운드를 다시 보내면
   원격을 비우고 새로 올린다(멱등) — 단 learner 가 그 라운드를 **처리 중**일 때는 금지.

10. 학습 끝날 때까지 기다렸다 산출물 회수 (DONE 이 뜨면 완성된 것)
    ```
    uv run ./actor/recv_round.py --round <N> --timeout 3600
    ```
    → `$A_CKPT/expo/<run id>/r<NNN>/` (learner 와 같은 상대경로). 그리고 7로 돌아간다.

## 라운드 번호

send / learner / recv 가 **같은 번호**를 쓴다.

```
actor    $A_RUNS/<run id>/r000/                          보낸 기록
learner  $L_RUNS/<run id>/r000/{dataset/,READY}          메일박스
learner  $L_CKPT/expo/<run id>/r000/{payload/,meta.json,DONE}
actor    $A_CKPT/expo/<run id>/r000/                     회수
```

learner 는 **READY 가 있고 DONE/FAILED 가 없는 가장 작은 번호**를 집는다 — 잡이 죽어도 이어받고,
같은 번호를 다시 보내면 다시 처리한다 (READY 의 SHA 로 구분). READY 를 맨 마지막에 따로 올리는
이유는 `kubectl cp` 가 원자적이지 않아서다 — learner 가 READY 의 숫자를 디스크와 대조해 절반만
도착한 라운드를 걸러낸다.

## 후보간 Q std — 이 루프가 되는지 보는 지표

EXPO 는 후보 액션들을 critic 으로 줄 세워 고른다. 그래서 **critic 이 액션을 구분하지 못하면
argmax 가 무작위**가 되고 루프 전체가 base BC 와 같아진다. 서버 로그의 `후보간 Q std` 가 그것이다.

지금 상태 (Phase D critic 20k step, openarm):

```
후보간 Q std        0.0012      ← 사실상 0. argmax 가 무작위다
base 후보 다양성    std 0.0275  ← 같은 관측에서 8개를 뽑아도 이만큼밖에 안 벌어진다
edit_scale         0.2         ← edit 이 흔드는 크기 (base 다양성의 7배)
```

원인은 학습 데이터에 **"같은 상황에서 다르게 행동해본" 기록이 없다**는 것이다. BC 정책이 거의
결정론적이라 Q 가 사실상 V(s) 로 수렴한다. 이걸 푸는 것이 라운드를 도는 이유다 — edit 이 실제로
실행되고 그 결과가 라벨링되면서 critic 이 처음으로 액션의 좋고 나쁨을 볼 재료를 얻는다.
라운드가 지나며 이 숫자가 커지는지 보면 된다.

## 아직 stub 인 것

- **learner 의 학습** — 지금 `learner/loop.py` 는 왕복(감지 → 검증 → 산출물 → DONE)만 검증하는
  `export_stub` 이다. 실제 `update()` 로 교체해야 한다: 라운드 누적 리플레이 버퍼 →
  `rl/data.py` × `rl/expo.py` × `rl/vla_rldx.py`, 산출물(LoRA + critic + encoder + residual +
  temperature ≈ 120MB) export, 주기적 체크포인트
- **정책 서버 핫리로드** — 지금은 라운드마다 서버를 다시 띄워야 한다 (백본 13.8GB 로드에
  약 40초). `--artifacts` 만 다시 읽으면 되므로 나중에 붙인다
- **GPU 잡 yaml** — `k8s/learner.yaml` 은 CPU 전용이고 learner 환경도 py3.10(RLDX-1) 로 맞춰야 한다

# BC Recipe
## Openarm
```
uv run ./utils/convert_data.py ~/Code/rrc-release/data/user/0819/openarm_rh56f1_teleop/ -o ~/ws/junmo_cho/dataset/0819_openarm_rh56f1_teleop --modality modality/openarm_lefthand/modality.json
```
```
kubectl cp ~/ws/junmo_cho/dataset/0819_openarm_rh56f1_teleop junmo-cho-data-pod:/data/junmo_cho/workspace/datasets/0819_openarm_rh56f1_teleop
```
## CJL
```
uv run ./utils/convert_data.py ~/ws/rrc-release/data/CJL/ -o ~/ws/junmo_cho/datasets/cjl_0819_v9 --modality modality/rby1m_wuji2/modality.json
```



actor(로컬 워크스테이션 + 로봇) ↔ learner(학습 클러스터)로 분리된 **동기(synchronous) RL 루프**.
정책은 **RLDX-1**, RL 알고리즘은 **EXPO-FT**. 리얼월드 실험이라 라운드를 사람이 끊어가며 돈다.

```
                    ┌─────────── round N ───────────┐
actor   정책 서빙 → 롤아웃(사람이 성패 라벨) → 변환 → 업로드
                                                      │
learner                              RL 학습 ←────────┘
                                        │
actor                       체크포인트 회수 ←── round N+1 정책
```

## 경계

| | 무엇 | rd-rl 이 하는 일 |
|---|---|---|
| `~/Code/rrc-release` | 로봇 조작·teleop·롤아웃 (RLWRLD 배포판) | **건드리지 않는다.** 데이터 디렉토리를 읽고, ZMQ 정책 서버를 띄워준다 |
| `third_party/RLDX-1` | VLA 정책 (submodule, `junmo/cjl-v6`) | 학습·서빙 백본. RL 쪽 수정은 여기 브랜치에 |
| `third_party/expo-ft` | EXPO-FT 원본 (submodule, JAX/pi0.5) | **참조용 pin.** 실행하지 않는다 (아래 참고) |

rrc-release 와의 접점은 두 개뿐:
- **데이터**: rrc 가 LeRobot 포맷으로 저장한 디렉토리를 읽는다 (경로는 명령어 인자로 받는다)
- **정책**: rrc 의 ZMQ 클라이언트(`rrc/inference/zmq_client.py`, `RldxCodec`, 기본 포트 5555)가
  rd-rl 이 띄운 정책 서버에 붙는다 — rrc 쪽은 host/port 만 보면 된다

## 설치

의존성은 [uv](https://docs.astral.sh/uv/) 로 관리한다 (`pyproject.toml` + `uv.lock`).
**다른 머신에서 같은 세팅을 재현하려면**:

```bash
git clone --recurse-submodules git@github.com:junmocho-lab/rd-rl.git
cd rd-rl
uv sync          # .python-version(3.12) 의 파이썬까지 uv 가 받아온다
```

`uv.lock` 이 커밋돼 있어서 패키지 버전이 정확히 같은 `.venv/` 가 만들어진다. 실행은:

```bash
uv run python utils/convert_data.py ...      # 또는 .venv/bin/python
```

그 머신에 맞게 **`configs/paths.sh` 의 경로는 고쳐야 한다.**

시스템 의존성이 하나 있다 — **`ffmpeg`** (비디오 재인코딩). 파이썬 패키지가 아니라 따로 깔려
있어야 한다 (학습 서버 k8s Job 은 시작 시 `apt-get install -y ffmpeg` 를 한다).

## 라운드 한 바퀴 순서

| # | 단계 | 어디서 | 상태 |
|---|---|---|---|
| 1 | 정책 서버 띄우기 (round N-1 정책, round 0 은 base BC) | actor | ⬜ |
| 2 | 롤아웃 + 사람이 에피소드별 성공/실패 라벨 (rrc-release) | actor | ✅ rrc 가 함 |
| 3 | **비디오 320x192 변환** | actor | ✅ `utils/convert_data.py` |
| 4 | 학습 서버로 업로드 (`kubectl cp`) | actor → learner | ⬜ |
| 5 | RL 학습 잡 제출 + 로컬에서 로그 모니터링 | learner | ⬜ |
| 6 | 체크포인트 회수 (`kubectl cp`) | learner → actor | ⬜ |
| 7 | → 1 로 (round N+1) | | |

**3번을 업로드 전에 하는 이유**: 720x1280 원본은 dataloader 가 GPU 를 굶긴다
(RLDX-1 실측: util 9.5% / 3.06 s-per-step → 320x192 에서 1.74 it/s). 모델 입력은 어느 쪽이든
(3,192,320) 이라 결과가 같고, 전송량도 크게 줄어든다 (262 에피소드 356MB → 12MB 급).

## 데이터

rrc 가 쓰는 LeRobot v2.1 이 **이미 RL transition 형태**라 포맷 변환은 필요 없다 (해상도만 줄인다).

```
observation.images.camera_ego_{left,right}   [3,720,1280]
observation.joint_{position,effort,velocity} [28],  action [28]
next.success    성공 에피소드의 마지막 1프레임만 True  → sparse terminal reward
next.done       에피소드 종료
next.truncated  시간 제한 종료 (bootstrapping 구분용)
```

두 종류를 쓴다:
- **teleop** (`rrc-release/data/user/<날짜>/..._teleop/`) — 사람 시연. base BC(SFT) 학습 데이터이자
  EXPO-FT 의 오프라인 demo 버퍼
- **rollout** (`rrc-release/data/junmo_cho/<날짜>.../`) — 정책 롤아웃. 라운드마다 새로 생기는 RL 데이터

## 지금 있는 것

```
rd-rl/
├── configs/paths.sh             # 경로 설정 (source 해서 쓴다)
├── modality/                    # RLDX modality.json (embodiment 별)
├── utils/convert_data.py        # LeRobot 비디오 다운스케일 (증분·병렬)
├── rl-dataset/                  # 변환된 데이터 (git 에는 디렉토리만)
├── pyproject.toml  uv.lock      # 의존성 (재현용)
└── third_party/{RLDX-1,expo-ft} # submodule (pin)
```

### configs/paths.sh
경로를 셸 변수로 둔다. 파서가 필요 없고 `kubectl` 명령에 그대로 꽂힌다.

```bash
source configs/paths.sh

# base 정책 받아오기 (learner → local)
kubectl -n $RDRL_NS cp $RDRL_POD:$RDRL_LEARNER_CKPT/$RDRL_BASE_POLICY \
                       $RDRL_LOCAL_CKPT/$RDRL_BASE_POLICY

# 데이터 올리기 (local → learner). kubectl cp 는 없는 원격 경로에 못 쓰므로 부모를 먼저 만든다
kubectl -n $RDRL_NS exec $RDRL_POD -- mkdir -p $RDRL_LEARNER_DS/r0
kubectl -n $RDRL_NS cp $RDRL_LOCAL_DS/r0/openarm_rh56f1_teleop \
                       $RDRL_POD:$RDRL_LEARNER_DS/r0/openarm_rh56f1_teleop
```

한 번 쓰고 안 바뀌는 것만 둔다. 롤아웃/teleop 원본 경로와 실험 설정(task, 하이퍼파라미터)은
**여기 없다** — 라운드마다 바뀌므로 명령어 인자로 받는다.

### utils/convert_data.py
```bash
python utils/convert_data.py ~/Code/rrc-release/data/junmo_cho/0815_openarm_rh56f1_inference/ -o ../dataset
```
SRC 하위의 LeRobot 데이터셋들을 `OUT/<SRC 이름>/<데이터셋 이름>_320x192/` 로 미러링.
`data/`·`meta/` 는 그대로 복사하고 `meta/info.json` 의 해상도만 패치한다. 재실행은 증분
(이미 변환된 것은 건너뛰고, 에피소드가 늘어난 데이터셋은 없는 비디오만 변환).
`--modality <파일>` 을 주면 각 데이터셋의 `meta/modality.json` 으로 설치한다 (RLDX 가 요구하는
파일이고 rrc 는 만들지 않는다). 이미 변환된 데이터셋에도 넣으므로 나중에 따로 추가할 수 있다.
주요 옵션: `-s 320x192` `-j <병렬>` `--dry-run` `--force` `--keep-info`

## 남은 것

큰 순서로:

1. **업로드/회수** (`kubectl cp`, 파드 `junmo-cho-data-pod`) — 라운드를 실제로 돌리려면 먼저 필요
2. **학습 잡 제출** — RLDX-1 `k8s/*.yaml` 을 템플릿으로. 라운드마다 데이터 경로 /
   `MODEL_OUTPUT_DIR` / base 정책이 바뀐다
3. **EXPO-FT ↔ RLDX-1 연동** ← 제일 큰 블록. expo-ft 원본은 JAX/Flax + pi0.5(openpi) 전용이고
   Python `>=3.11` 인데 RLDX-1 은 PyTorch + `==3.10.*` 이라 같은 환경에 올라가지 않는다.
   VLA 자체가 학습 대상(`actor_train_state`, `ema_params`)이라 "RLDX 를 추론 서버로 분리"하는
   우회도 안 된다. → **알고리즘을 torch 로 포팅**해야 한다
   (`expo_ft/agents/alg/expo_ft.py` 931줄 + `networks/` + `distributions/` + `data/replay_buffer.py`)
4. **정책 서버** — base RLDX 정책 + EXPO 의 edit policy/critic 을 묶어 ZMQ 로 서빙
5. **라운드 기록** — 어떤 정책으로 모은 데이터로 무엇을 학습했는지. RECAP round 1 에서 checkout
   어긋남(`RLDX_ROOT` 가 recap 없는 트리를 가리켜 조용히 BC 가 돌던)으로 한 번 물렸으므로,
   라운드마다 rd-rl/RLDX-1 SHA 를 남기는 장치가 필요하다
