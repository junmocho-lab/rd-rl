# DexJoCo hammer_nail — offline-RL ablation 설계안

## 0. 실측 비용 (이 계획의 근거)

| 항목 | 실측 | 출처 |
|---|---|---|
| critic 학습 | **~150 step/s** → 20K=2분, **200K=22분** | success_200k ckpt mtime (176K→200K, 159초) |
| BC 롤아웃 수집 | 6~14 초/ep | 세 데이터셋 parquet mtime |
| **critic 롤아웃 평가** | **~17 초/ep (비혼잡) → 100ep ≈ 30분**, 혼잡 시 최대 79초/ep | eval 8종 파일 mtime |
| cogfeat 추출 | 미측정 (A는 완료) | — |
| images.mm | **384 KB/frame** → C(288K프레임) ≈ **113 GB** | 23.07GB / 58673 |
| 디스크 여유 | 653 GB | df |
| 클러스터 | background 21노드, 대기 57건 → 동시 5~6잡 현실적 | squeue |

### 여기서 나오는 결론 3가지

1. **학습 스텝 ablation은 공짜다.** 200K를 22분에 돌리고 중간 ckpt를 남기면
   1K/5K/10K/20K/50K/100K/200K가 *한 번의 학습*으로 전부 나온다.
   따로 6번 돌릴 필요 없다.
2. **비용은 전부 "롤아웃 평가"에 있다.** 화폐 단위를 `1 eval = 100ep ≈ 30~45분`으로 잡고
   이 단위를 어디에 쓸지가 설계의 전부다.
3. **cogfeat 추출이 임계경로다.** B(49.7K프레임), C(288.6K프레임) 미추출.
   C는 113GB + 수 시간. **이걸 제일 먼저 걸어야 한다.**

## 1. 세팅 이름

| 태그 | (r,d) | scene | ep | BC 성공률 | 프레임 |
|---|---|---|---|---|---|
| **A** `d2r8_s0` | (8,2) | fixed | 200 | 61.5% | 58,673 |
| **B** `d5r20_s0` | (20,5) | fixed | 200 | 72.5% | 49,700 |
| **C** `d5r20` | (20,5) | random | 1000 | 48.6% | 288,559 |

## 2. 설계 원칙 — 왜 이 순서인가

지금까지 측정된 효과 크기를 보면:

| 축 | 효과 |
|---|---|
| **학습 스텝 (20K vs 200K)** | **−12pp** (79% → 67%)* |
| 방법 (sel32 → parl_argmax) | +10pp (79% → 89%) |
| 액션 공간 (625d → 90d) | +33pp (선택이 해로움 → 이로움) |

\* 단, 67% 런은 critic 경로가 기록 안 돼 있어 `success_200k` 라는 게 **추정**이다.

즉 **오버피팅이 지금까지 잰 것 중 제일 큰 단일 효과**다. 스텝을 잘못 고르면
다른 모든 비교가 노이즈에 묻힌다. 그러니 스텝은 "곁다리 ablation"이 아니라
**먼저 고정해야 할 1순위 축**이다.

동시에, 학습 중 **holdout AUC를 1K마다 이미 공짜로 재고 있다.**
(train_eps=success 여도 holdout은 실패 에피소드를 남기므로 AUC는 유효 — `offline_iql_qvgm.py:226,418`)

> **핵심 가설: holdout AUC가 롤아웃 성공률을 예측하는가?**
> 예측한다면 A에서 한 번만 비싸게 검증하고, B/C의 ckpt 선택은 **평가 없이 AUC로** 하면 된다.
> → eval 20개 이상 절약. 이 검증이 전체 계획에서 가장 값어치 있는 실험이다.

## 3. 실행 계획

### T0 — 언블록 (eval 0개)

- **T0a. cogfeat 추출: C 먼저, 그 다음 B.** 임계경로. 지금 바로.
- **T0b. 코드 3건**
  - `--num-min-qs`: `StepwiseEnsemble`에 REDQ 부분집합 min. (`nets.py`의 `subsample`은
    구 `CriticEnsemble`에만 있음 → qvgm 경로에 없음. ~5줄)
  - `build_images` 스킵: cogfeat 완성 시 건너뛰기 (`offline_iql_qvgm.py:153`이 무조건 호출).
    → images.mm 삭제 가능해져 **~155GB 확보**
  - `rollout_summary.json`에 `fixed_scene`/`rtc_delay`/`critic_path` 기록
    (지금 없어서 `sel32__critic_unknown` 같은 사태가 났다)
- **T0c. critic 8개 학습** (각 200K, ckpt {1,5,10,20,50,100,200}K 보존) — 총 3 GPU-시간

| critic | 세팅 | 아키텍처 | 필터 |
|---|---|---|---|
| `A/succ_dq`, `A/all_dq` | A | double-Q | success / all |
| `B/succ_dq`, `B/all_dq` | B | double-Q | success / all |
| `C/succ_dq`, `C/all_dq` | C | double-Q | success / all |
| `C/succ_ens`, `C/all_ens` | C | ensemble 10, min 2 | success / all |

### T1 — 스텝 × 필터 곡선 (세팅 A) — **eval 10개**

`sel32` 로 고정하고 (방법이 아니라 critic 품질을 재는 것이므로):

    sel32 × {succ_dq, all_dq} × {1K, 5K, 20K, 100K, 200K}

산출물: ① under/overfit 곡선 ② **AUC↔성공률 상관** ③ 필터링 승자 ④ 각 필터의 step\*

### T2 — 방법 비교 (A, B) — **eval 8개**

step\*는 T1(A) 또는 AUC(B)로 결정.

- A: `parl_argmax × {succ,all}` (2) — succ@20K는 이미 있음(89%)
- B: `sel32 × {succ,all}`, `parl_argmax × {succ,all}` (4)
- B: `sel32__succ_dq@{1K,200K}` (2) — **AUC 프록시가 B로 전이되는지 확인용**
- B의 BC 베이스라인은 수집 런(72.5%, n=200) 재사용

### T3 — 랜덤 씬 (C) — **eval 7개**

- BC 동일시드 재실행 (1) — 랜덤 씬이라 페어링하려면 필요
- `{sel32, parl_argmax} × {succ_dq, succ_ens}` (4) ← **아키텍처 축**
- 승자 방법 × `all_dq` (1), 승자 방법 × `all_ens` (1)

### 합계

eval 25개 × ~40분 ≈ **17 GPU-시간**, 동시 5~6잡 → 대기 포함 **8~10시간**. 하루에 들어간다.

## 4. 계획에서 뺀 것과 이유

- **Q-guided action distillation**: 아직 구현 자체가 없다. 설계+구현+디버그가 필요하고,
  정책 파인튜닝 루프라 critic ablation과 성격이 다르다 → **이번 배치에서 제외**, 다음 단계.
- **B/C의 조밀한 스텝 곡선**: T2의 확인용 2개로 프록시 전이만 보고, 나머지는 AUC로 대체.
- **A/B의 ensemble**: 아키텍처 축은 C(어려운 세팅)에서만. 쉬운 고정씬에서 앙상블 이득이
  나올 여지가 작다.
- expectile / discount / bins 스윕: 효과 크기가 위 3축보다 작을 것으로 보고 제외.

## 5. 결정이 필요한 것

1. eval당 에피소드 수: 100(±4pp) vs 200(±3.5pp)
2. holdout 비율: 0.1(학습데이터 많음) vs 0.2(AUC 신뢰도 높음 — A 성공123ep 기준 홀드아웃 12→25ep)
3. distillation을 이번 배치에 포함할지
4. guidance 변종(`parl_samp`, `guide_move`)을 계속 들고 갈지, `sel32`+`parl_argmax` 2개로 줄일지
