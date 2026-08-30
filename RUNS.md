# DexJoCo hammer_nail — offline RL ablation 실행 기록

| | 세팅 | (r,d) | 씬 | ep | BC 기준선 | 성공길이 |
|---|---|---|---|---|---|---|
| **A** | `d2r8_s0` | (8,2) | 고정 | 200 | **61.5%** | 251.7 |
| **B** | `d5r20_s0` | (20,5) | 고정 | 200 | **72.5%** | 206.2 |
| **C** | `d5r20` | (20,5) | 랜덤 | 1000 | 앞200ep **51.5%** | 214.5 |

핵심 질문 — **Q1 몇 스텝** / **Q2 성공만 vs 전부** / **Q3 (r,d)·랜덤씬 일반화 + 앙상블**

Stage 1 은 방법을 `sel32` 로 고정해 critic 품질만 잰다. 방법 비교는 승자 step\* 에서만.

---

## 실행 규약

제출 정책 두 가지가 있다. 아래 명령에 이미 반영돼 있다.

- `MODEL_OUTPUT_DIR` 이 제출 환경에 있어야 한다
- `sbatch` 잡 이름은 **50자 이상**이어야 한다 (`srun` 은 해당 없음)

**`background` 파티션을 쓰지 말 것.** `PreemptMode=REQUEUE` 라 선점되면 조용히 재시작하는데,
학습은 스텝 0부터 다시 돌면서 `--keep-steps` 로 보호된 체크포인트는 남아 **서로 다른 런이
한 디렉토리에 섞인다**. 실제로 A/all 학습이 이렇게 3번 죽었다 (105000 → 50000 → 18400 스텝에서
각각 선점). `rlwrld` / `rlwrld_premium` / `batch` 는 `PreemptMode=OFF` 라 파일 기본값(`rlwrld`)을
그대로 쓰면 된다. 방어를 두 겹 넣어 뒀다 — sbatch 의 `--no-requeue`, 그리고 학습 시작 시
run 디렉토리의 이전 런 체크포인트 삭제.

```bash
cd /rlwrld2/home/junmo_cho/ws/rd-rl
MOD=MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/junmo_cho/dexjoco-rl   # 아래에서 계속 쓴다
```

---

## ✅ 완료

### 캐시 (cogfeat + actnorm)

A 는 이전부터 있었고, B 는 srun 으로 만들었다. `srun` 이 이미 잡혀 있으면 그 안에서
`bash` 로 실행하면 된다 (`#SBATCH` 줄은 주석이라 무시된다).

```bash
EXP=dexjoco_hammer_nail_d5r20_s0 bash sbatch/dexjoco/critic/cogfeat.sbatch
```

### critic 학습 4개 — 각 ~19분

`--keep-steps` 로 **한 잡이 스텝 격자를 전부 남긴다** (`1000 5000 20000 25000 50000 100000 200000`).
스텝마다 따로 돌리지 않는다.

```bash
for E in dexjoco_hammer_nail_d2r8_s0 dexjoco_hammer_nail_d5r20_s0; do
for T in success all; do
sbatch -J dexjoco-hammer-nail-critic-${E#dexjoco_hammer_nail_}-train$T-doubleq-steps200000-qvgm-iql-eefpose \
  --export=ALL,$MOD,EXP=$E,TRAIN_EPS=$T,STEPS=200000 \
  sbatch/dexjoco/critic/critic_hammer_nail.sbatch
done; done
```

### Stage 1 — A 스텝 × 필터 곡선 10개 (각 ~55분)

```bash
for C in success all; do
for S in 1000 5000 20000 100000 200000; do
sbatch -J dexjoco-hammer-nail-eval-d2r8s0-fixedscene-sel32-critic-$C-step$S-200ep \
  --export=ALL,$MOD,EXP=dexjoco_hammer_nail_d2r8_s0,METHOD=sel32,CTAG=$C,STEP=$S,EPISODES=200 \
  sbatch/dexjoco/rollout_w_critic/eval.sbatch
done; done
```

**결과** (200ep, 같은 씬 페어, McNemar) — BC 61.5% 대비:

| critic | 성공률 | 성공길이 | 승/패 | p |
|---|---|---|---|---|
| success@1k | 70.5% | 201.9 | 45/27 | 0.044 * |
| success@5k | 65.5% | 236.5 | 51/43 | 0.47 |
| **success@20k** | **76.0%** | 229.2 | 58/29 | **0.0025** ** |
| success@100k | 58.5% | 200.4 | 47/53 | 0.62 |
| success@200k | 69.0% | 203.2 | 59/44 | 0.17 |
| all@1k | 62.5% | 217.3 | 53/51 | 0.92 |
| all@5k | 60.4% | 216.5 | 43/44 | 1.00 |
| **all@20k** | **74.5%** | 224.1 | 58/32 | **0.008** ** |
| all@100k | 65.0% | 216.9 | 44/37 | 0.51 |
| all@200k | 65.0% | 211.4 | 51/44 | 0.54 |

**Q1 답: 20K.** 두 필터 모두 20K 가 봉우리이고, 각 곡선에서 **유일하게 p<0.01 인 지점**이다.
독립적인 두 곡선이 같은 답을 낸 것이 근거다. 100K 에서 critic 이 무용해진다
(success@100k 는 BC 보다 낮고 승/패 47/53).

**다중비교 보정.** arm 10 개를 검정했으므로 α=0.05 면 우연한 유의가 0.5개 기대된다.
Bonferroni (0.05/10 = 0.005) 를 적용하면 `success@20k` (0.0025) 만 통과하고 `all@20k` (0.008) 는
경계, **`success@1k` (0.044) 는 탈락한다** — 1K 가 BC 보다 낫다는 것은 약한 증거다.

**Q2 답: A 에서는 무승부. 단 일반화하면 안 된다.** 20K 에서 76.0 vs 74.5 이고, 둘을 직접
페어 비교하면 45승 42패 p=0.830 이다. 그런데 **불일치 쌍이 87/200 = 43.5%** 다 — 두 critic 은
장면의 43% 에서 서로 다른 답을 내는데 이기는 횟수만 비슷하다. "필터가 critic 에 영향을 주지
않는다" 가 아니라 "A 에서는 둘 다 우연히 비슷하게 좋다" 는 뜻이다.

A 가 이 축의 나쁜 시험대인 이유: 씬이 하나뿐이라 실패 77개가 성공 123개와 **같은 장면**이다.
성공 궤적이 이미 그 장면을 조밀하게 덮으므로 실패가 커버리지를 거의 안 늘린다. C 는 실패
514개가 서로 다른 장면이고 프레임이 2.79배(A 는 1.90배)라 상황이 다르다.

sparse terminal reward 에서 실패 에피소드는 **종단 값 0 인 유일한 데이터**다. 없으면 critic 은
"나쁜 액션" 을 외삽으로만 안다. IQL 의 expectile 0.8 은 낮은 리턴의 영향을 눌러주므로 실패를
넣어도 V 가 끌려내려가지 않으면서 커버리지만 얻는다 — expectile 을 쓰는 이유 자체다.
**→ C 에서 필터 축을 반드시 유지한다.**

곡선이 단조롭지 않다 (70.5 → 65.5 → 76.0 → 58.5 → 69.0). 5K/200K 의 출렁임은 노이즈일 수
있지만 **20K 봉우리와 100K 골의 18pp 차이는 노이즈로 설명되지 않는다.**

---

## ⏳ 진행 중

### Stage 1 — B 스텝 × 필터 곡선 10개

```bash
for C in success all; do
for S in 1000 5000 20000 100000 200000; do
sbatch -J dexjoco-hammer-nail-eval-d5r20s0-fixedscene-sel32-critic-$C-step$S-200ep \
  --export=ALL,$MOD,EXP=dexjoco_hammer_nail_d5r20_s0,METHOD=sel32,CTAG=$C,STEP=$S,EPISODES=200 \
  sbatch/dexjoco/rollout_w_critic/eval.sbatch
done; done
```

**볼 것: step\* 가 (8,2) → (20,5) 로 옮겨가도 20K 인가.** A 와 격자를 맞춰 놓아서 곡선 대
곡선으로 비교된다. BC 기준선 72.5%.

### C cogfeat 추출 (288,559 프레임, `images.mm` ~113GB)

```bash
sbatch -J dexjoco-hammer-nail-cogfeat-extract-d5r20-randomscene-1000ep-288k-frames-vlm-features-actnorm \
  --export=ALL,$MOD,EXP=dexjoco_hammer_nail_d5r20 \
  sbatch/dexjoco/critic/cogfeat.sbatch
```

### distillation relabel — dry-run 검증

```bash
sbatch --export=ALL,$MOD,EXP=dexjoco_hammer_nail_d2r8_s0,CTAG=success,STEP=20000,EPS=success,DRY=1,LIMIT=200 \
  sbatch/dexjoco/distill/relabel.sbatch
```

---

## ⬜ 남은 것

### C critic 학습 4개 (cogfeat 끝난 뒤)

1000ep 이라 500K 까지. 격자에 25K/100K/500K 가 자동 포함된다.

```bash
E=dexjoco_hammer_nail_d5r20
for T in success all; do
sbatch -J dexjoco-hammer-nail-critic-d5r20-randomscene-train$T-doubleq-steps500000-qvgm-iql-eefpose \
  --export=ALL,$MOD,EXP=$E,TRAIN_EPS=$T,STEPS=500000 \
  sbatch/dexjoco/critic/critic_hammer_nail.sbatch
sbatch -J dexjoco-hammer-nail-critic-d5r20-randomscene-train$T-redq-ens10min2-steps500000-qvgm-iql-eefpose \
  --export=ALL,$MOD,EXP=$E,TRAIN_EPS=$T,STEPS=500000,NUM_QS=10,NUM_MIN_QS=2 \
  sbatch/dexjoco/critic/critic_hammer_nail.sbatch
done
```

### Stage 1 — C (랜덤씬) 11개

**필터 축을 유지한다** (위 Q2 참고 — A 의 무승부는 C 로 전이될 이유가 없다).
앙상블은 스텝을 다 스캔하지 않고 step\* 에서만 붙인다.

confound 하나: 같은 스텝이어도 `all` 은 데이터가 2.79배라 epoch 수가 다르다. A 에서
batch 128 기준 20K 스텝이면 success 83 epoch / all 44 epoch 인데 **둘 다 20K 에서 정점**이었다.
epoch 가 아니라 절대 스텝 수가 지배한다는 뜻이므로 같은 격자로 비교하는 것이 맞다.
다만 C 는 데이터가 5배라 최적점이 위로 밀릴 수 있어 500K 까지 잡았다.

```bash
E=dexjoco_hammer_nail_d5r20

# 필터 x 스텝 (double-Q) — 8개
for C in success all; do
for S in 5000 25000 100000 500000; do
sbatch -J dexjoco-hammer-nail-eval-d5r20-randomscene-sel32-critic-$C-step$S-200ep \
  --export=ALL,$MOD,EXP=$E,METHOD=sel32,CTAG=$C,STEP=$S,EPISODES=200 \
  sbatch/dexjoco/rollout_w_critic/eval.sbatch
done; done

# 아키텍처 (REDQ 10/2) — step* 확정 후 2개. <step*> 를 바꿔 넣는다
for C in success_ens all_ens; do
sbatch -J dexjoco-hammer-nail-eval-d5r20-randomscene-sel32-critic-$C-step-STAR-200ep \
  --export=ALL,$MOD,EXP=$E,METHOD=sel32,CTAG=$C,STEP=<step*>,EPISODES=200 \
  sbatch/dexjoco/rollout_w_critic/eval.sbatch
done

# C 의 BC 기준선 — 랜덤씬이라 같은 시드로 다시 재야 페어 비교가 된다.
# 51.5% (수집 데이터 앞 200ep) 를 재현하면 시드 정렬이 맞는 것이다.
sbatch -J dexjoco-hammer-nail-eval-d5r20-randomscene-bc-basepolicy-nocritic-200ep \
  --export=ALL,$MOD,EXP=$E,METHOD=bc,EPISODES=200 \
  sbatch/dexjoco/rollout_w_critic/eval.sbatch
```

### Stage 2 — 방법 비교 (step\* 확정 후)

`parl` = 후보 32 → top-10 → ∇_A Q 상승 4스텝(move 0.05) → argmax.
A@20K 에서 sel32 76% → parl **89%** (n=100).

```bash
for E in dexjoco_hammer_nail_d2r8_s0 dexjoco_hammer_nail_d5r20_s0; do
sbatch -J dexjoco-hammer-nail-eval-${E#dexjoco_hammer_nail_}-parl-critic-success-step20000-200ep \
  --export=ALL,$MOD,EXP=$E,METHOD=parl,CTAG=success,STEP=20000,EPISODES=200 \
  sbatch/dexjoco/rollout_w_critic/eval.sbatch
done
```

---

## Action distillation

설계는 `dexjoco_distill.md`, 알고리즘 근거는 `PARL-DISTILL.md`.
**목표는 89% 를 재현하는 것**이다 — Q-VGM 논문 Table 1 도 guidance 88.7 vs distillation 88.8 로
둘이 같다. 넘어야 할 벽이 아니라 맞춰야 할 눈금이다.

arm 3(전체 200ep, 원본 액션)은 생략한다 — 실패를 가르치는 것이라 결과가 뻔하다.

| arm | 에피소드 | 액션 | 역할 |
|---|---|---|---|
| 0 | — | — | BC base (61.5%, 있음) |
| 1 | 성공 123 | 원본 | **filtered BC — 필터링 효과 대조군** |
| 2 | 성공 123 | relabel (critic=success@20k) | distill |
| 4 | 전체 200 | relabel (critic=all@20k) | distill, 실패 궤적 포함 |

`2 − 1` 이 액션 개선의 순효과다. **arm 1 이 없으면 arm 2 를 귀속시킬 수 없다** — 같은 정책의
성공 롤아웃으로 파인튜닝만 해도 오를 수 있기 때문이다 (self-imitation).

```bash
# (1) relabel — 서빙이 89% 를 낸 설정이 기본값 (N=32, keep=10, 상승 4x0.05, temp 0.001,
#     선택/채택은 앙상블 min, 상승 방향만 mean, keep-best 켜짐)
sbatch --export=ALL,$MOD,EXP=dexjoco_hammer_nail_d2r8_s0,CTAG=success,STEP=20000,EPS=success \
  sbatch/dexjoco/distill/relabel.sbatch
sbatch --export=ALL,$MOD,EXP=dexjoco_hammer_nail_d2r8_s0,CTAG=all,STEP=20000,EPS=all \
  sbatch/dexjoco/distill/relabel.sbatch

# (2) 서브셋 — arm 1 은 원본에서, arm 2 는 relabel 출력에서. 비디오는 심링크된다.
V=/workspace/junmo_cho/dexjoco/venv/bin/python
$V rl/make_subset.py --data rl-dataset/dexjoco/hammer_nail_d2r8_s0 \
     --out rl-dataset/dexjoco/distill_arm1_bc_success --eps success
$V rl/make_subset.py --data rl-dataset/dexjoco/relabel_dexjoco_hammer_nail_d2r8_s0__success@20k_success \
     --out rl-dataset/dexjoco/distill_arm2_relabel_success --eps success
```

(3) LoRA 파인튜닝과 (4) 병합·평가는 `PARL-DISTILL.md` 6절 참고.

**진행 판정** — relabel dry-run 의 **로그 액션이 이긴 비율 < 70%**. 넘으면 critic 이 base
policy 를 못 이겨 distillation 이 no-op 이 된다. relabel 대상이 critic 이 학습에 쓴 바로 그
에피소드들이라(holdout 0.1) 로그 액션의 Q 가 부풀려져 있을 수 있다.

---

## 결과 집계

```bash
python3 sim/dexjoco/summarize_evals.py            # 전체
python3 sim/dexjoco/summarize_evals.py --exp d2r8_s0
```

씬이 고정이라 eval 의 k번째와 수집 데이터의 k번째가 같은 장면이다. 그래서 McNemar 정확검정을
쓸 수 있고, 진행 중인 런도 앞 n개만 잘라 비교하므로 중간 집계가 유효하다.

`rollout_summary.json` 의 `serve` 필드에 어떤 critic 체크포인트로 낸 결과인지 박힌다
(전에 `sel32__critic_unknown` 이 생긴 원인을 막는다). `mean_length` 는 **전체** 평균이라
해석에 쓰면 안 된다 — 실패는 대부분 360 타임아웃이라 성공률의 함수가 된다.
성공만 본 것은 `success_mean_length` 다.
