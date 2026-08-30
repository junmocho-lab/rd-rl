# fuji (rby1m_rh56f1) — offline RL 루프 세팅

태스크: *"Pick up the plastic tray by its handle, pull it out of the shelf, turn the torso,
and place it into the other shelf."*

dexjoco hammer_nail 에서 관통시킨 루프(BC → 롤아웃 → critic → test-time 선택/guidance →
distillation)를 실기로 옮긴다. 실행 명령과 dexjoco 결과는 `RUNS.md`, distillation 설계는
`dexjoco_distill.md` / `PARL-DISTILL.md`.

---

## 1. 설정

```
exp          configs/exp/fuji_d3r8.yaml
dataset      rl-dataset/fuji/0831_fuji_all       335ep / 218,359프레임 / 30fps
base_policy  checkpoints/fuji/rldx_img_0828-40k-2gpu-128b-A-only
chunk        action_horizon 40 / replan 8 / latency 3
critic 액션  right_arm_joints → 11스텝 x 7관절 = 77차원
discount     0.998
```

### 데이터 구성

| 세션 | ep | 성공 | 비고 |
|---|---|---|---|
| teleop A_init | 98 | 100% | 전문가 시연 |
| teleop B | 45 | 100% | 전문가 시연 |
| hil_anchor x2 | 39 | 100% | **`is_intervention` / `is_stop` 컬럼 있음** (개입 프레임 16~21%) |
| inference rollout_101146 | 32 | 93.8% | |
| inference rollout_dark | 39 | 76.9% | 조명 조건이 다르다 |
| inference s180_20260829 | 51 | 19.6% | |
| inference fail_cases | 31 | 0.0% | **큐레이션된 실패 모음 — 무작위 표본이 아니다** |
| **합계** | **335** | **75.2%** | |

**75.2% 를 BC 기준선으로 읽으면 안 된다.** teleop(전부 성공)과 큐레이션된 실패가 섞인 값이다.
정책의 실제 성공률을 알려면 별도의 무작위 롤아웃이 필요하다.

`next.success` 는 진짜 태스크 라벨이다 (에피소드 마지막 프레임에 True). 다만 `fail_cases` 를
뺀 `0831_fuji` 만 보면 전부 성공이라 라벨이 없는 것처럼 보이니 주의.

### exp 를 하나만 두는 이유

`0831_fuji_only_success` 는 `0831_fuji_all` 의 부분집합임을 확인했다 (7개 세션의 성공
에피소드가 개수도 길이도 정확히 일치). 트레이너의 `--train-eps success|all` 로 두 critic 을
만들면 되고, 그편이 **더 낫다** — `only_success` 로 학습하면 홀드아웃에 실패가 없어 AUC 를
못 재지만, `all` 을 쓰면 `--train-eps success` 여도 홀드아웃에 실패가 남아
(`offline_iql_qvgm.py:226` 은 train 만 필터한다) 학습 내내 AUC 를 볼 수 있다.

---

## 2. dexjoco 에서 가져오는 것

### fuji 는 A 세팅과 같다

| | latency+replan | 관절 | critic 액션 | 에피소드당 결정 | 결과 |
|---|---|---|---|---|---|
| **A** `d2r8_s0` | 10 | 9 | **90차원** | 36.7 | **76.0%** vs BC 61.5% (p=0.0025) |
| **fuji** | 11 | 7 | **77차원** | ~81 | — |
| B `d5r20_s0` | 25 | 9 | 225차원 | 12.4 | 63.5% vs BC 72.5% (유의한 arm 없음) |

**A 가 fuji 의 참조다.** 액션 차원 77 은 A 의 90 과 같은 구간이고, B(225)에서는 critic 이
BC 보다 나빴다. 액션 공간을 작게 유지하는 것이 이 실험의 제1 원칙이다:

```
dexjoco 실측 — 액션 차원이 유일한 결정 변수였다
  전 관절 625차원        test-time 선택 -16pp (해로움)
  eef pos+rot 90차원     +17.5pp (이로움)
  (20,5) 의 225차원      BC 보다 나빠짐
```

fuji 에서 전 관절을 쓰면 11 x 34 = **374차원**이다. 쓰면 안 된다.

### 학습 스텝은 세팅마다 다시 찾아야 한다

**step\* 는 (r,d) 를 넘어 전이되지 않는다.** A 는 20K 가 봉우리인데 (76.0%, p=0.0025)
같은 20K 가 B 에서는 BC 보다 9pp 낮았다 (63.5%). B 는 격자 최소값인 1K 가 최고였다.

`--keep-steps` 로 한 번의 학습이 격자 전체를 남기므로 (`1000 5000 20000 25000 50000 100000
200000`, 200K 가 22분) 스텝 스캔 자체는 공짜다. **비용은 전부 롤아웃 평가에 있다.**

### test-time 방법 (A@20K, n=100)

```
BC                                                61.5%
sel32       후보 32 → Q(min) argmax                79.0%
parl        후보 32 → top-10 → ∇Q 상승 4x0.05 → argmax   89.0%
guide_all   후보 전부 상승 → argmax                 72.0%   ← Q 는 높은데 성능은 낮다
```

`guide_all` 이 무너지는 것이 winner's curse 다 — 후보를 전부 분포 밖으로 밀면 critic 이 가장
크게 과대평가한 것이 뽑힌다. top-M 필터와 keep-best 가 그것을 막는다.

---

## 3. 실행

sbatch 는 `sbatch/dexjoco/` 아래 있지만 **전부 EXP 이름만 받는 범용**이다. yaml 에서
dataset/replan/latency 를 읽으므로 fuji 도 그대로 쓴다.

```bash
cd <repo>
MOD=MODEL_OUTPUT_DIR=<쓰기 가능한 경로>          # 제출 플러그인이 요구한다
```

### (1) cogfeat 추출 — 임계경로

`images.mm` 118GB (3카메라 x 192x320 x 218,359프레임), `cogfeat.npy` 3.3GB.
dexjoco 288K프레임이 2시간이었으니 약 1.5시간.

```bash
sbatch -J fuji-rby1m-rh56f1-cogfeat-extract-0831-all-335ep-218k-frames-vlm-features-actnorm \
  --export=ALL,$MOD,EXP=fuji_d3r8 sbatch/dexjoco/critic/cogfeat.sbatch
```

srun 안이면: `EXP=fuji_d3r8 bash sbatch/dexjoco/critic/cogfeat.sbatch`

### (2) critic 학습 2개 — 각 ~25분

```bash
for T in success all; do
sbatch -J fuji-rby1m-rh56f1-critic-d3r8-train$T-doubleq-steps200000-qvgm-iql-rightarm77d \
  --export=ALL,$MOD,EXP=fuji_d3r8,TRAIN_EPS=$T,STEPS=200000 \
  sbatch/dexjoco/critic/critic_hammer_nail.sbatch
done
```

### (3) 평가

**여기서 dexjoco 와 갈린다.** dexjoco 는 시뮬레이터가 있어 `eval.sbatch` 로 200 에피소드를
자동으로 돌렸다. fuji 는 실기라 롤아웃이 비싸고, `sim/dexjoco/rollout_dexjoco.py` 를 쓸 수 없다.

정책 서버는 그대로 쓸 수 있다 (`rl.vla_rldx serve` 가 exp 이름만 받는다):

```bash
python -m rl.vla_rldx serve \
  --exp fuji_d3r8 --model-path <base_policy 절대경로> \
  --artifacts checkpoints/fuji_d3r8-critic/success/critic_020000.pt \
  --rtc-inference-mode trained --rtc-exec-horizon 8 \
  --n-cand 32 --parl-keep 10 --parl-temp 0.001 --guide-steps 4 --guide-move 0.05 \
  --host 0.0.0.0 --port <port>
```

**★ `--guide-move 0.05` 는 dexjoco 에서 튜닝한 값이다.** 정규화 액션 공간의 이동 폭이라
로봇이 다르면 물리적 의미가 다르다. dexjoco 에서 replay 로 허용오차를 재서 (액션을 0.005
흔들면 성공률 100→40%) 눈금을 잡았는데, fuji 에서 같은 측정을 할 수 없으면 **`--guide-steps 0`
(선택만) 으로 시작하는 것이 안전하다.** dexjoco 에서도 선택만으로 61.5 → 79.0 이 나왔다.

### (4) 롤아웃 없이 먼저 볼 수 있는 것

실기 롤아웃 전에 critic 이 쓸 만한지 오프라인으로 판정할 수 있다:

```bash
# 홀드아웃 성공/실패 Q 곡선 비디오
python sim/dexjoco/critic_grid.py --exp fuji_d3r8 --data <데이터 절대경로> \
  --critic success/critic_latest.pt --holdout 0.1 --set holdout --max-rows 8 \
  --out <출력>.mp4

# PA-RL 액션 최적화 통계 (relabel 없이 --dry-run)
python -m rl.relabel_parl --exp fuji_d3r8 --data <데이터> --checkpoints checkpoints \
  --critic <critic.pt> --features cogfeat.npy \
  --num-samples 32 --num-keep 10 --num-steps 4 --guide-move 0.05 --temp 0.001 \
  --dry-run --limit 200
```

두 번째가 특히 중요하다. **로그된 액션이 이긴 비율 < 70%** 여야 critic 이 base policy 를
개선한다는 뜻이고, 그렇지 않으면 선택/guidance/distillation 이 전부 no-op 이 된다.
학습 로그의 **홀드아웃 AUC** 도 같이 본다 (A 는 0.94 수준이었다).

---

## 4. 아직 안 쓰고 있는 신호 — `is_intervention`

`hil_anchor` 세션 39개에만 있는 컬럼이다.

```
hil_anchor_114705   20ep   개입 2,907프레임 (16.5%)   개입 있는 ep  8/20
hil_anchor_135441   19ep   개입 3,960프레임 (21.3%)   개입 있는 ep 12/19
```

**사람이 개입했다 = 그 시점에 정책이 실패하고 있었다.** 지금 critic 은 이걸 무시하고 종단
성공만 본다. 종단 보상은 에피소드당 1프레임뿐이라 (652프레임 중 1개) 크레딧 할당이 매우
희박한데, 개입 라벨은 **실패 순간을 직접 가리키는 조밀한 신호**다. HIL-SERL 계열이 정확히
이 신호로 보상을 만든다.

종단 보상 파이프라인이 도는 것을 확인한 뒤에 별도로 검토한다. 지금 손대면 두 변화가
교란된다.

---

## 5. 열린 것

- 실기 롤아웃 하네스 — `eval.sbatch` 대응물이 없다. 정책 서버는 그대로 쓰되 클라이언트가 필요하다
- `guide_move` 눈금 — fuji 의 물리적 허용오차를 모른다. 선택(`--guide-steps 0`)부터 시작
- 정책의 진짜 성공률 — 무작위 롤아웃이 없어 BC 기준선이 없다. 개선폭을 잴 기준이 필요하다
- `is_intervention` 보상
