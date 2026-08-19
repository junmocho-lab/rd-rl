# rd-rl

# 실행 순서

0. rd-rl git clone 받고 각 submodule 및 .venv 세팅 
1. config 세팅 
2. teleop data 세팅 (post-training에 활용)
   ```
   asd
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
├── configs/config.yaml          # 사이트 설정 (양쪽 rd-rl 경로, 체크포인트 루트, k8s)
├── utils/convert_data.py        # LeRobot 비디오 다운스케일 (증분·병렬)
└── third_party/{RLDX-1,expo-ft} # submodule (pin)
```

### configs/config.yaml
한 번 쓰고 안 바뀌는 것만 넣는다. 롤아웃/teleop 경로와 실험 설정(task, base policy,
하이퍼파라미터)은 **여기 없다** — 라운드마다 바뀌므로 명령어 인자 / 별도 설정으로 받는다.

### utils/convert_data.py
```bash
python utils/convert_data.py ~/Code/rrc-release/data/junmo_cho/0815_openarm_rh56f1_inference/ -o ../dataset
```
SRC 하위의 LeRobot 데이터셋들을 `OUT/<SRC 이름>/<데이터셋 이름>_320x192/` 로 미러링.
`data/`·`meta/` 는 그대로 복사하고 `meta/info.json` 의 해상도만 패치한다. 재실행은 증분
(이미 변환된 것은 건너뛰고, 에피소드가 늘어난 데이터셋은 없는 비디오만 변환).
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
