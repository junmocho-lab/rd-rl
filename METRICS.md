# 지표와 튜닝

EXPO 라운드 학습이 잘 되고 있는지 보는 값들과, 문제가 생겼을 때 만질 값들.

로깅은 세 군데 남는다:

```
wandb                                       train/* (update 마다), round/* (라운드 마다)
$L_RUNS/<run id>.learner.log                update 5회마다 한 줄
$L_CKPT/expo/<run id>/rNNN/meta.json        라운드 끝의 스냅샷 + 코드 SHA + expo_deviations
```

wandb 는 `WANDB_API_KEY` 가 있으면 자동으로 붙는다 (`k8s/learner.yaml` 이 RLDX-1 학습 잡과
같은 secret `junmo-cho-train-creds/wandb-api-key` 를 쓴다). 프로젝트 기본값 `rd-rl-expo`,
run 이름은 run id. Job 이 재시작해도 `id=<run id>` + `resume="allow"` 로 같은 run 에 이어 붙는다.

---

## 1. `train/candidate_q_std` — 이 루프의 핵심 지표

EXPO 는 후보 액션을 critic 으로 줄 세워 고른다. **이 값이 0 이면 argmax 가 주사위**이고
루프 전체가 "BC + 랜덤 노이즈" 로 퇴화한다.

```
현재(초기)  0.03      사실상 0
목표        라운드가 지나며 커진다
```

0 인 이유는 데이터에 "같은 상황에서 다르게 행동해본" 기록이 없어서다 — BC 정책이 거의
결정론적이라 Q 가 사실상 V(s) 로 수렴한다. 측정값:

```
base 후보 8개의 다양성   std 0.0275   (같은 관측에서 뽑아도 이만큼밖에 안 벌어진다)
edit_scale               0.2          (base 다양성의 7배)
```

즉 **edit 이 실제로 실행되고 그 결과가 라벨링되는 것**만이 critic 에게 액션의 좋고 나쁨을
가르칠 재료다. 라운드를 도는 이유가 이것이고, 이 숫자가 오르는지가 그 검증이다.

## 2. `train/q`, `train/q_max` — 범위가 정해져 있다

보상이 sparse terminal (성공 에피소드의 마지막 1프레임만 1.0) 이라 **리턴 상한이 1** 이다.

| | 뜻 |
|---|---|
| `q` 가 계속 음수 | 보상이 아직 전파되지 않았다. 랜덤 초기화 직후엔 정상 (측정: -0.62) |
| `q` 가 [0, 1] 로 올라옴 | 정상. 성공에 가까운 프레임이 1 에 가까워야 한다 |
| **`q_max > 1.2`** | **발산.** 상한이 1 인데 넘었다는 뜻이므로 즉시 멈출 신호 |

값 전파는 느리다. Bellman backup 한 번이 결정 하나를 옮기고 `tau=0.005` 의 시간상수가
약 200스텝이므로, openarm(에피소드당 약 38결정) 은 최소 38×200 ≈ **7,600 스텝**이 필요하다.
fuji 는 에피소드가 더 길어(약 150결정) 더 오래 걸린다. "안 되는 것"과 "느린 것"을
구분할 때 이 숫자를 쓴다.

## 3. `train/select_ratio_with_residual` — edit 후보가 이긴 비율

```
0.96 (현재)  Q std 가 0 이라 의미 없다 — 8개 섭동 중 최댓값이 base 를 우연히 이기는 것
높음 + Q std > 0   edit 이 실제로 더 나은 액션을 찾고 있다 (EXPO 의 전제가 맞는 상태)
0                  residual 이 쓸모없다
```

## 4. `train/entropy` vs target — SAC 온도

온도가 자동으로 조절해 target 을 맞춘다. `target = -활성차원/2`:

| 실험 | 활성 차원 | target |
|---|---|---|
| openarm_rim | left_arm 7 + left_hand 6 = 13, × replan 8 = 104 | **-52** |
| fuji | right_arm 7, × replan 8 = 56 | **-28** |

entropy 가 target 보다 낮으면 `train/temperature` 가 올라가고, 높으면 내려간다.
**temperature 가 발산하면(수십~수백) 뭔가 잘못된 것.**

## 5. 보조 지표

| 값 | 정상 | 벗어나면 |
|---|---|---|
| `train/next_q_nan_ratio` | 0 | 타깃 계산에 NaN. 즉시 조사 |
| `train/critic_grad_norm` | 튀지 않음 | 스파이크는 발산 전조 |
| `train/actor_loss` | 낮고 안정 | 오르면 BC 타깃(고른 후보)이 VLA 가 낼 수 있는 범위를 벗어나는 중 |
| `train/mean_edit_norm` | 1.31 (상한 `edit_scale·√dim` ≈ 2.04) | 상한에 붙으면 tanh 포화 = residual 이 한계 |
| `train/critic_loss` | 내려감 | — |

## 6. `round/success_rate` — 진짜 목표

`send_round.py --success N` 을 준 라운드만 남는다. **이게 실제로 좋아지는지가 전부**이고,
위 지표들은 그게 왜 좋아지는지/안 좋아지는지를 설명하는 값이다. 라운드마다 꼭 주는 게 좋다.

`round/buffer_*` 는 버퍼가 실제로 커지고 있는지 (프레임/에피소드/성공 에피소드).

---

## 튜닝

### 우리가 정한 값 — 자유롭게 만진다

| 값 | 어디 | 언제 |
|---|---|---|
| `round.updates_per_episode` (3) | exp yaml | 데이터당 학습량. 라운드 = 이것 × 에피소드 수. 적으면 안 배우고 많으면 적은 데이터에 과적합 |
| `round.episodes_per_round` (5) | exp yaml | 라운드 주기. 사람이 감당할 만큼 |
| `explore_groups` | exp yaml | 측정된 프레임 간 \|Δ\| 로 정한다. 안 움직이는 관절을 넣으면 엔트로피 예산만 낭비 |
| `inference_latency`, `replan_steps` | exp yaml | rrc 설정과 **반드시** 일치. 다르면 critic 이 실행되지 않은 액션을 평가한다 |
| `round.buffer_capacity` | exp yaml | 디스크. fuji 는 프레임당 553KB(카메라 3개) |

### EXPO-FT 원본 값 — 근거가 생겼을 때만

바꾸면 `ExpoConfig.deviations()` 가 라운드 manifest 에 남긴다. 의심스러운 순서로:

**1. `discount` (0.99)** — 실효 할인은 `discount**replan_steps` = 0.9227/결정.
fuji 는 에피소드당 약 150결정이라 `0.9227^150 ≈ 6e-6` — **보상이 에피소드 초반까지 사실상
안 닿는다.** Q 가 후반만 살아있고 초반이 평평하면 이게 원인이고, 올리는 것이 원칙적인 대응.

**2. `tau` (0.005)** — 값 전파 속도. 2번 항목의 7,600스텝 계산이 여기서 나온다.
"맞는데 느린" 것처럼 보이면 여기.

**3. `edit_scale` (0.2)** — 정규화 액션 스케일(std 0.5)의 40%. **유일하게 하드웨어 위험과
직결된다.** 실기에서 너무 거칠면 낮추는 것이 안전하지만, 낮추면 critic 이 배울 액션
다양성도 줄어든다.

**4. `utd_ratio` (20)** — 계산량 그 자체. update() 1회 85~89초(로컬 5090, openarm)의
대부분이 이것 × N 이다. 라운드가 너무 오래 걸리면 여기.

### 아직 코드가 읽지 않는 값

- `round.seed_teleop_episodes` — 성공 시연 시드. 지금은 `send_round.py --dataset` 에
  teleop 세션을 같이 주는 것으로 대신한다.
- `round.warmup_n_edit_samples` — 초기에 edit 을 줄이는 값. 정책 서버는 항상
  `expo.n_edit_samples` 를 쓴다.
- `round.offline_ratio`, `round.min_online_episodes` — min_online_episodes 만 learner 가 읽는다.

---

## 실측 기준값 (로컬 5090, openarm, 버퍼 25,746 프레임)

```
update() 1회        85~89초       batch 64 × utd 20 → 디노이저 배치 512
라운드 (3×5회)      약 21분
theta.pt            175 MB        enc + critic + target + residual + temp + lora(64텐서)
정책 서버 추론      121 ms/회     N=8 + edit 8 (순수 BC 는 118ms — 후보 8개 비용이 3ms)
```
