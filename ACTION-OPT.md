# 액션 최적화 (PA-RL) 와 critic 진단 계획

목적: `offline_iql.py` 로 학습한 critic 이 **실제로 액션을 조정하는지** 를 본다.
방법: 성공/실패 에피소드의 각 프레임에서 (a) 로그된 액션 과 (b) critic 으로 최적화한 액션 을
비교한다. 우리 실험은 `explore_groups: [right_arm_joints]` 이므로 **그 그룹만 최적화**한다.

---

## 1. PA-RL 의 액션 최적화 — 코드 기준 정리

`jaxrl_m/agents/continuous/action_optimization.py` 의 `action_optimization_sample_actions`.
정책 파라미터를 학습하지 않고, **관측마다 액션 공간에서 최적화를 풀어** 정책을 만든다.

```
입력: 관측 o, base policy 가 뽑은 후보 M 개, 학습된 critic
──────────────────────────────────────────────────────────────────────
① 인코딩 1회        obs → critic_encoder → encoding      (:311-331)
                    후보 M 개에 대해 encoding 을 repeat  (액션만 다르므로 재사용)
② global (선별)     Q(o, a_m) 을 M 개 전부 계산
                    q_values_before = 앙상블 **mean**    (:364  .mean(axis=0))
                    argsort 로 상위 K 개만 남긴다         (:372  top_k_indices)
                    (옵션) 가장 나쁜 후보 하나를 데이터 액션으로 교체 (:434)
③ local (상승)      K 개에 대해 num_steps 번 반복:       (:96-140)
                        g = ∇_a  Q̄(o, a)                 ← jax.grad(critic_fn)
                        a ← a + step_size · g
                        a ← clip(a, action_low, action_high)
                    Q̄ = 앙상블 mean (기본) 또는 min (optimize_critic_ensemble_min)
                    ※ 스텝마다 Q 가 가장 높았던 액션을 따로 기록해 둔다
                       (keep_action_with_max_value — 상승이 발산해도 최선을 잃지 않는다)
④ 선택              상승 후 Q 를 다시 계산 → logits       (:483-497)
                    argmax=True  → 결정론적 최선 하나
                    argmax=False → Categorical(logits) 에서 샘플 (기본)
```

### 기본 하이퍼파라미터

| | 값 | 위치 |
|---|---|---|
| `num_base_policy_actions` M | 32 (OpenVLA 8) | `base_config.py:15`, `real_config.py:145` |
| `num_actions_to_keep` K | 10 (OpenVLA 4) | 같은 곳 |
| `num_steps` | 10 | `base_config.py:17` |
| `step_size` | **3e-4** | `base_config.py:18` |
| `optimize_critic_ensemble_min` | **False** (= mean 으로 상승) | `base_config.py:19` |
| `use_target_critic` | False (= online critic) | `base_config.py:20` |

### 우리 것(EXPO)과의 대응

| | EXPO (`rl/expo.py:select_from_chunks`) | PA-RL |
|---|---|---|
| 후보 | VLA N=8 | base policy M=32 → Q 로 top-K=10 |
| 개선 | 학습된 residual actor (`edit_scale` 0.2/0.01) | **∇_a Q 상승** (학습 파라미터 없음) |
| 랭킹 | target critic, 무작위 2개 **min** | online critic, 앙상블 **mean** |
| 선택 | argmax | Categorical (또는 argmax) |
| 편집 범위 | `explore_groups` 만 (`spec.index` 마스킹) | 전체 액션 (액션공간 clip 만) |

**우리는 `explore_groups: [right_arm_joints]` 로 편집 범위를 제한하므로, 액션 상승도 같은
마스크를 써야 한다** — `spec.index` 가 이미 그 위치를 담고 있다 (액션벡터 374 중 right_arm 7차원
× 실행 8스텝 = 56개 위치. prefix 3스텝은 편집 불가라 애초에 index 에서 빠져 있다).

### step_size 는 그대로 쓸 수 없다

`3e-4` 는 그들의 Q 스케일·액션 스케일에 맞춘 값이다. 우리는:
- Q ∈ [0,1] (distributional support), 액션은 정규화 공간 (std ≈ 0.5, clip ±1.2)
- 실측 `|ΔQ|` 이 액션 전체를 뒤섞을 때 0.15 → `∂Q/∂a` 가 매우 작다

따라서 **먼저 `‖∇_a Q‖` 를 재고**, `num_steps × step_size × ‖g‖` 가 EXPO 의 `edit_scale`(0.2)
규모가 되도록 step_size 를 잡는다. 진단 스크립트가 이 값을 찍어준다.

---

## 2. 진단 설계 — `rl/probe_actopt.py` (구현 예정)

### 무엇을 볼 것인가

성공/실패 에피소드의 프레임 t 에서:

```
a_log  = 로그된 액션 (actnorm 의 [0, LAT+R) 구간, 374차원)
a_opt  = a_log 에서 시작해 ∇_a Q 로 상승시킨 액션 (right_arm 56차원만 이동)
```

측정치:

| 지표 | 뜻 | 기대 |
|---|---|---|
| `‖∇_a Q‖` (그룹별) | critic 이 어느 관절에 민감한가 | right_arm 이 0 이 아니어야 한다 |
| `ΔQ = Q(a_opt) − Q(a_log)` | 최적화가 Q 를 얼마나 올렸나 | > 0 (당연). **크기**가 관심 |
| `‖a_opt − a_log‖` | 얼마나 움직였나 | `edit_scale` 0.2 규모면 적정 |
| **실패 vs 성공의 ΔQ** | **실패 프레임에서 더 크게 고칠 것이 있는가** | 실패 > 성공 이면 critic 이 "고칠 여지" 를 안다 |
| `a_opt` 의 방향 | 실패 프레임에서 성공 궤적 쪽으로 가는가 | 아래 ③ |

### 세 가지 비교

**① Q 상승량** — 실패 프레임에서 성공 프레임보다 `ΔQ` 가 크면, critic 이 "여기는 액션을
바꿀 여지가 있다" 를 인식하는 것이다. 반대면 critic 이 상태만 보고 체념한 것이다.

**② 방향 일치 (핵심)** — 앵커(라벨한 실패 시점) 프레임에서, 픽셀로 매칭한 성공 프레임의
액션을 `a_succ` 라 하자. 최적화가 그쪽으로 갔는지 코사인 유사도로 본다:

```
cos( a_opt − a_log ,  a_succ − a_log )   > 0 이면 성공 액션 쪽으로 밀고 있다
```

`probe_pairs.py` 가 만든 `probe_pairs.json` 의 쌍을 그대로 재사용한다.

**③ 관절별 변화량** — `a_opt − a_log` 를 right_arm 7관절 × 8스텝으로 펼쳐 어느 관절/시점을
움직이는지 본다. 사람이 보는 실패 모드(`down miss`, `wrong insert`)와 맞는지 눈으로 확인.

### 절차

```
1. critic_iql-*.pt 로드 (enc + critic, distributional 이면 스칼라 읽기)
2. spec = explore_spec(mod.offsets("action"), ["right_arm_joints"], A, replan, latency)
   → spec.index 가 편집 가능한 56개 위치
3. step_size 캘리브레이션: 표본 256개에서 ‖g‖ 중앙값을 재고
   step_size = target_disp / (num_steps · ‖g‖),  target_disp = 0.2·√56
4. 각 에피소드에서 균등 간격 프레임을 뽑아 (a) ~ (c) 계산
5. 출력: 성공/실패별 표 + 관절별 변화 히트맵 PNG + 앵커 프레임의 코사인 유사도
```

### 왜 이 진단이 필요한가

지금까지 확인된 것은 **critic 이 상태의 성패를 안다**는 것뿐이다 (step 28,000 에서 AUC 1.000,
종료 100프레임 전 격차 0.51). 하지만 롤아웃에서 쓰이는 것은 **같은 상태에서 액션을 줄 세우는
능력**이고, 그건 별개다. 이전 SARSA critic 에 대한 2×2 스왑 프로브가 `|ΔQ| ~ 1e-4` 로 나와
"액션을 못 가른다" 는 결론이 나왔었다 (다만 그때는 라벨 시점이 할인 지평 밖이라 측정 자체가
무의미했다). 이번에는 그 시점에 격차가 0.10~0.15 있으므로 측정이 성립한다.

---

## 3. 실행 예정 명령

```bash
PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" NO_ALBUMENTATIONS_UPDATE=1 \
third_party/RLDX-1/.venv/bin/python -u -m rl.probe_actopt \
  --exp fuji --data rl-dataset/fuji-rl-dataset --checkpoints checkpoints \
  --critic critic_iql-dist128-t07-g0999-q10all-s0.pt \
  --groups right_arm_joints --num-steps 10 --pairs   # --pairs 면 probe_pairs.json 사용
```

산출물: `checkpoints/fuji-critic/eval_<critic태그>/actopt_*.png` + stdout 표.
