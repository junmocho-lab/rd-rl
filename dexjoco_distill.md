# Q-guided Action Distillation — 설계안

## 0. 핵심 관찰 — 새로 짤 코드가 거의 없다

`parl_argmax` 롤아웃의 출력이 **이미 학습 가능한 LeRobot v2.1 데이터셋**이다.

| 확인 항목 | 결과 |
|---|---|
| `codebase_version` / `fps` | `v2.1` / 50 — BC 학습 데이터셋과 동일 |
| `features` 키 | 동일 |
| `modality.json` | state·action `[eef_position, eef_rotation, hand_joints]`, video `[camera_front, camera_wrist]` — 동일 |
| `action` 컬럼의 내용 | **critic 이 고른 명령 액션** (`rollout_dexjoco.py:568` `executed25`) |
| rot6d manifold | `action_25_to_env` 로 SO(3) 재직교화됨 → **BC 데모와 같은 manifold** |

즉 relabel 도, 포맷 변환도 필요 없다. 테스트 타임 절차가 실행한 액션이 곧 distillation 타깃이다.

`launch_train.py` 쪽도 이미 필요한 걸 다 갖고 있다:
- `dataset_paths` + `dataset_mix_ratios` (`assembly.py:30-47`) → 원본 데모와 혼합 가능
- `base_model_path` 가 로컬 디렉토리 허용 (`train_config.py:42`) → 현 BC 체크포인트에서 이어 학습

**새로 짜야 하는 것: 성공 에피소드만 뽑는 서브셋 빌더 하나뿐.**

## 1. 무엇을 왜 distill 하는가

테스트 타임에 후보 32 → top-10 → ∇_A Q 상승 → argmax 로 **89%** 를 냈다 (BC 61.5%).
그 절차가 실제로 실행한 액션들을 정책에 supervised 로 넣으면 정책이 절차를 흉내내게 된다.

얻는 것:
1. 테스트 타임 **32배 추론 비용 제거**
2. 배포 시 **critic 불필요**
3. **반복 가능** — 다음 라운드의 후보 32개가 더 좋은 분포에서 나오므로 선택의 출발점이 올라간다

> **1라운드 distill 로는 89% 를 넘을 수 없다. 근사할 뿐이다.**
> 89% 를 넘는 유일한 경로는 (3) 의 반복이다. 이걸 먼저 못 박아 두어야 결과 해석이 안 꼬인다.

## 2. 파이프라인

```
Round 1
  1. parl_argmax 롤아웃 N ep         eval.sbatch 그대로 (EPISODES 만 키움)
  2. 성공 에피소드만 필터            ← 새 코드 (서브셋 빌더)
  3. BC 파인튜닝                     launch_train.py, base = 현 BC ckpt
  4. critic 없이 평가                eval.sbatch METHOD=bc, 같은 200 씬
                                     BC 61.5% / parl 89% 와 3자 비교
Round 2 (선택)
  5. 새 정책으로 parl_argmax 재실행 → 2~4 반복
```

## 3. 정직하게 짚어둘 함정

**고정 씬에서의 distill 은 숫자가 부풀려진다.** A 는 씬이 하나뿐이라, 성공 궤적 400개로 BC
파인튜닝하면 정책이 그 씬을 **외운다**. 95% 가 나와도 "distillation 이 작동했다" 가 아니라
"한 장면을 외웠다" 일 수 있다. selection/guidance 는 critic 도 같은 씬으로 학습돼서 비교가
내적으로 일관됐지만, distillation 은 정책 자체가 씬을 외울 수 있어 성격이 다르다.

→ A 는 **파이프라인 검증용**으로만 쓰고, 의미 있는 숫자는 **C(랜덤 씬)** 에서 낸다.
   fuji 로 옮길 때 믿을 수 있는 건 C 쪽 결과다.

**붕괴 위험.** 자기가 만든 데이터로 계속 학습하면 분포가 좁아진다. 완화책 두 가지:
낮은 LR + 적은 스텝으로 파인튜닝 (희석 없음), 또는 원본 teleop 데모 혼합 (희석 있음).

## 4. 결정이 필요한 것

1. **어느 세팅부터** — A(고정씬, critic 준비됨, 오늘 가능) / C(랜덤씬, 정직한 숫자, cogfeat 42%)
2. **distill 롤아웃 개수** — BC base 는 teleop 100ep 로 학습됐다
3. **원본 teleop 혼합 여부** — 순수 distill vs 혼합
4. **Round 2 를 오늘 범위에 넣을지**
