# PA-RL / Cal-QL — 우리 루프에 쓸 수 있나

`third_party/PolicyAgnosticRL` (arXiv 2412.06685, *Policy Agnostic RL: Offline RL and Online RL
Fine-Tuning of Any Policy Class and Backbone*) 을 읽고, **오프라인 critic 초기화를 SARSA 대신
Cal-QL 로 바꾸는 것**이 우리 세팅에서 타당한지 판단한 기록.

결론 먼저:
- **Cal-QL 로 바꾸는 것은 방향이 맞다.** SARSA 가 구조적으로 못 주는 "액션 서열"을 손실함수로
  만들어내고, 우리가 실측한 Q 의 음수 고정점 문제에 직접적인 처방이 있다.
- **그런데 그것만으로는 안 된다.** 우리 실측 병목은 알고리즘이 아니라 **크레딧 경로 길이**다
  (결정적 순간이 보상에서 71 결정 = γ^71 = 5e-4 떨어져 있음). 에피소드 꼬리 자르기 /
  discount 조정이 **선행**돼야 Cal-QL 도 배울 게 생긴다.
- PA-RL 의 **액션 공간 gradient ascent** 는 EXPO 의 residual(edit) actor 를 학습 파라미터 없이
  대체할 수 있어 매력적이다 (temperature/entropy 튜닝이 전부 사라진다).

---

## 1. PA-RL 이 하는 일

핵심 주장: **정책 개선을 파라미터 공간이 아니라 액션 공간에서 한다.** 기존 actor-critic 은
`∇_θ Q(s, π_θ(s))` 로 정책 가중치를 밀지만, VLA/diffusion 같은 큰 백본에서는 미분이 비싸거나
백본을 망가뜨린다. PA-RL 은 세 단계로 우회한다:

```
① global   base policy 에서 M=32 개 샘플 → critic Q 로 top-K=10 선별
② local    그 K 개에 대해 ∇_a Q 로 액션 자체를 gradient ascent
           num_steps=10, step_size=3e-4, 액션공간으로 clip
③ 선택     결과를 Boltzmann(또는 argmax) 로 하나 뽑는다
④ distill  ①~③ 으로 얻은 개선된 액션을 base policy 에 supervised 로 주입
```

- ①~③: `jaxrl_m/agents/continuous/action_optimization.py:280` (`action_optimization_sample_actions`),
  local 부분은 같은 파일 `:44` (`local_optimization_steps`) — `jax.grad(critic_fn)` 로 액션에 대한
  그래디언트를 받아 `actions = actions + step_size * grad` 후 clip.
- ④: `train.py:1022` "Base policy distillation" 블록. 온라인 환경 스텝당 `base_policy_utd` 비율로
  distillation 스텝을 돈다. 그래서 base policy 클래스가 무엇이든(diffusion, OpenVLA,
  autoregressive transformer) 상관없다 = "policy agnostic".
- **PA-RL 에이전트는 critic 만 학습한다**: `parl_calql.py:212` 의 `networks_to_update=frozenset({"critic"})`.
  정책 개선은 전부 ①~③ 의 추론 시점 최적화 + ④ 의 distillation 이다.

critic 은 **Cal-QL** 을 쓴다 (`parl_calql.py` → `cql.py` 상속, `calql.py` 는 `use_calql=True` 만
켜는 6줄 래퍼).

### 하이퍼파라미터 (기본값)

| | 값 | 위치 |
|---|---|---|
| `num_base_policy_actions` M | 32 (OpenVLA 는 8) | `configs/base_config.py:15`, `real_config.py:145` |
| `num_actions_to_keep` K | 10 (OpenVLA 는 4) | 같은 곳 |
| local `num_steps` / `step_size` | 10 / 3e-4 | `base_config.py:17` |
| `cql_alpha` | 0.005 (antmaze) / **0.01 (실기)** | `base_config.py:127`, `real_config.py:79` |
| `cql_n_actions` | 10 (실기 **4**) | `base_config.py:118`, `real_config.py:94` |
| critic ensemble / subsample | 10 / **2** | `base_config.py:123-124` |
| discount | 0.99 | `base_config.py:90` |
| `mixing_ratio` (offline:online 배치) | 0.5 | `base_config.py:84` |
| `distill_argmax` | False (softmax) | `base_config.py:87` |

---

## 2. Cal-QL 이 정확히 무엇인가 (코드 기준)

**CQL** — critic loss 에 보수적 항을 더한다:

```
cql_q_diff = logsumexp_a( Q(s, a) )  −  Q(s, a_data)          # cql.py:288-295
loss += cql_alpha * cql_q_diff
```

`a` 는 (랜덤 액션 n개) + (정책이 낸 next action n개) + (정책이 낸 current action n개) 를 합친
집합이다 (`cql_n_actions` 각). 즉 **데이터 밖 액션의 Q 를 명시적으로 눌러** 오프라인 과대추정을
막는다. 데이터 액션의 Q 는 그대로 두므로 결과적으로 "데이터 액션은 높고 그 밖은 낮다" 는 형태가
강제된다.

**Cal-QL 의 추가** — 그 억압에 바닥을 깐다:

```python
mc_lower_bound = batch["mc_returns"]                    # 참조(=행동) 정책의 실제 리턴
cql_q_samples  = jnp.maximum(cql_q_samples, mc_lower_bound)    # cql.py:223
```

*"참조 정책이 실제로 받은 리턴보다 낮은 값은 더 누르지 않는다."* 순수 CQL 은 OOD 를 과도하게 눌러
Q 전체가 참조 정책 가치 아래로 내려가고, 그러면 온라인 전환 직후 Q 를 다시 끌어올리는 데 시간을
다 써서 성능이 급락한다(unlearning). Cal-QL 은 그 바닥을 데이터의 MC return 으로 **캘리브레이션**해
offline→online 전환의 성능 하락을 없앤다. 이게 논문 제목의 "Offline RL **and Online RL
Fine-Tuning**" 의 실체다.

`mc_returns` 계산 (`jaxrl_m/data/roboverse_dataset.py:203` `calc_return_to_go`):
- 성공 궤적: 뒤에서부터 `r[i] + γ·prev·mask[i]`
- 실패 궤적: `r_last/(1−γ)` 상수 (reward 가 0/−1 스케일이라 `−1/(1−γ)`)

---

## 3. EXPO-FT 와 나란히 놓으면

| | EXPO-FT (지금 우리 것) | PA-RL |
|---|---|---|
| 후보 생성 | base VLA N=8 | base policy M=32 → critic 으로 top-K=10 |
| 개선 | **학습된 residual(edit) actor** + temperature/entropy | **∇_a Q 액션 상승** (학습 파라미터 0) |
| 선택 | target critic argmax | Boltzmann / argmax |
| critic 타깃 | 후보 max (온라인) / 우리 오프라인은 SARSA | Cal-QL (CQL + MC 하한) |
| 앙상블 | REDQ 10, min-2 | 10, subsample 2 (동일) |
| base policy 갱신 | 선택된 액션으로 LoRA BC | 선택된 액션으로 distill (**같은 발상**) |
| **오프라인 사전학습** | **없음** | **있음** |

즉 두 방법의 골격은 거의 같다. 다른 것은 **(a) 개선을 학습된 residual 로 하나 액션 그래디언트로
하나**, **(b) critic 을 오프라인에서 먼저 캘리브레이션하나** 두 가지다.

---

## 4. 우리 상황에 적용했을 때 — 찬성 근거

**① offline→online 이 원래 설계 목적이다.** 우리가 지금 하려는 게 정확히 그것이고, EXPO-FT 에는
오프라인 단계가 아예 없어서(원본 `train_pi_robo.py:292` 는 온라인 에피소드 10개를 모은 뒤 첫
업데이트를 돈다) 우리가 임시로 SARSA 를 붙인 상태다.

**② SARSA 가 구조적으로 못 주는 것을 손실함수로 만든다.** 우리 실측:

```
같은 상태(픽셀거리<0.05)의 프레임끼리 액션 차이 = 전체 분포의 4%(fuji) / 6.6%(openarm)
```

상태당 액션이 사실상 하나라서 SARSA 는 Q 를 V(s) 로 수축시키는 압력을 받는다. CQL 항은
**랜덤/정책 샘플 액션의 Q 를 명시적으로 낮추는** 항이므로, 데이터 다양성이 없어도 액션 방향의
기울기를 만들어낸다. 데이터로 못 얻는 것을 손실함수로 보완하는 방향.

**③ 우리 Q 의 음수 고정점에 직접적인 처방이다.** 실측된 병증:

```
q = -0.94 → -0.36 (온라인 63 update), critic_loss 0.0002   ← 자기 일관적인 음수 고정점
오프라인 critic: 남은결정 40+ 구간에서 Q ≡ -0.013
```

원인은 `r=0` 인 대다수 transition 에서 타깃이 `γ^R·min(2개 중)` 이고 min 편향 δ<0 이 
`Q* ≈ δ/(1−γ_eff) ≈ 13δ` 로 증폭되는 것. Cal-QL 은 MC return 하한으로 clip 하므로 **성공 궤적의
실제 리턴 아래로 내려가지 않는다.** 우리 보상이 성공 종료 1프레임뿐이라 MC return 은 정확히
`γ_eff^(남은 매크로 스텝)` 이고, 그건 우리가 Q 에 기대하는 모양 그 자체다. 하한이 곧 정답 모양인
셈이라 궁합이 좋다. (실패 궤적은 보상이 0 이므로 하한 0 — PA-RL 의 `−1/(1−γ)` 와 달리 우리는
음수 보상이 없다.)

**④ 액션 그래디언트가 우리 구조와 잘 맞는다.** 우리 critic 은 이미 액션에 대해 미분 가능하고,
실측으로 **액션이 Q 변동의 46% 를 설명한다**(가치가 있는 구간에서 `|ΔQ|` 0.15 vs `Q` 0.39).
`(latency+replan)×A = 374` 차원 중 exec 블록만 그래디언트를 흘리면 되고, 그 마스킹은
`ExploreSpec.index` 로 **이미 구현돼 있다**. residual actor / temperature / target_entropy 가
전부 사라진다.

**⑤ CQL 의 추가 비용이 우리 구조에서는 싸다.** CQL 은 액션 `cql_n_actions` 개에 대해 critic
forward 를 더 돌리는데, 우리는 34.1M 인코더가 비용을 지배하고 **인코더는 한 번만** 돌면 된다
(액션만 바뀌므로 critic MLP 3.5M 만 10배). PA-RL 도 같은 이유로 encoding 을 미리 계산해 넘긴다
(`action_optimization.py:311` 의 assert 참고).

---

## 5. 반대 / 주의

**① discount 문제는 Cal-QL 이 못 고친다 — 이게 지금 1순위다.** 실측:

| 끝까지 남은 결정 | Q(성공ep) | Q(실패ep) | 차이 | 이론 γ_eff^남은 | 액션 \|ΔQ\| |
|---|---|---|---|---|---|
| 0–5 | +0.387 | −0.013 | +0.401 | 0.818 | 0.153 |
| 5–10 | +0.304 | −0.014 | +0.317 | 0.547 | 0.139 |
| 10–20 | +0.166 | −0.014 | +0.180 | 0.299 | 0.079 |
| 20–40 | +0.027 | −0.013 | +0.040 | 0.090 | 0.008 |
| 40–70 | −0.013 | −0.014 | +0.000 | 0.012 | 0.004 |
| 70–120 | −0.013 | −0.013 | +0.000 | 0.0005 | 0.003 |

critic 의 유효 지평은 **~20 결정(5초)** 이고, 사람이 라벨한 실패 시점은 **71 결정 밖**이다
(fuji: 실패는 20초, 에피소드 종료는 40~45초). 거기서는 완벽한 critic 도 `γ^71 = 5e-4` 를
출력해야 한다. Cal-QL 을 써도 **MC return 하한 자체가 그 지점에서 5e-4** 라 배울 신호가 없다.

→ 선행 조치 (효과 순):
1. **에피소드 꼬리 자르기.** 결과가 20초에 확정되는데 45초까지 기록하면 크레딧 경로를 스스로
   2배로 늘린다. 종료 조건을 결과 확정 시점으로 옮기면 남은 결정이 71 → 5~10 이 되고, 위 표에서
   그 구간은 이미 잘 작동한다.
2. **discount 올리기.** 71 결정 뒤 보상이 0.1 이상 남으려면 `γ_eff ≥ 0.968` → `γ ≥ 0.996`.

**② JAX 코드베이스라 그대로 못 쓴다.** RLDX-1 은 torch. 다만 옮길 양은 적다:
- Cal-QL critic loss: logsumexp 항 + MC 하한 clip → **~40줄**
- 액션 그래디언트 최적화: `torch.autograd.grad(Q.sum(), action)` 루프 → **~30줄**
- 그들의 데이터 파이프라인 / OpenVLA 캐싱 워커 / bridge 환경은 우리와 무관.

**③ CQL 의 정책 샘플이 VLA 를 부른다.** `logsumexp` 항에 `π(·|s)` 샘플이 들어가는데, 오프라인
학습에서 VLA 를 돌리면 우리가 피하려던 비용(update 당 70초)이 돌아온다. 세 가지 대안:
- (a) 랜덤 액션만 쓰기 — PA-RL 에도 `use_calql_on_random_actions`, `only_use_next_actions_for_cql`
  같은 스위치가 있다 (`base_config.py:120,129`)
- (b) 데이터 액션에 노이즈를 준 것으로 대체 (edit_scale 규모)
- (c) **VLA 후보를 한 번 뽑아 캐시** ← 가장 맞는 답. 오프라인 데이터는 고정이므로 프레임당 후보
  8개를 한 번 계산해 저장하면 (`55,564 × 8 × 40 × 34 float32 = 2.4GB`) 이후 학습에서 VLA 없이
  CQL 항을 계산할 수 있다. PA-RL 도 OpenVLA 용으로 같은 걸 한다
  (`--base_policy_offline_cache_path`, `scripts/openvla_caching_worker.py`).

**④ `cql_alpha` 는 민감한 노브다.** 실기 설정이 0.01, antmaze 가 0.005. 너무 크면 Q 가 눌려
아무것도 못 배우고, 작으면 CQL 효과가 없다. 우리 보상 스케일이 [0,1] 이라 그들의 스케일과 달라
그대로 쓸 수 없고 탐색이 필요하다.

---

## 6. 제안하는 순서

1. **(선행) 에피소드 꼬리 자르기 또는 `discount: 0.99 → 0.996`.** 이게 없으면 무엇을 해도
   결정적 순간에 신호가 없다. → 지금 라벨한 `anno.csv` 의 `fail_sec` 이 그대로 "여기서 잘라라"
   목록으로 쓰인다.
2. **오프라인 critic 을 Cal-QL 로 교체.** `rl/offline_critic_0.py` 의 TD 손실에 두 항 추가:
   - `mc_returns`: 우리 데이터에서 직접 계산 가능 — 성공 궤적은 `γ_eff^(남은 매크로 스텝)`,
     실패는 0
   - CQL penalty: 랜덤 섭동 + (캐시된) VLA 후보에 대한 `logsumexp − Q(s,a_data)`
3. **VLA 후보 캐시**를 한 번 만든다 (`actnorm.npy` 옆에 `candidates.npy`).
4. **온라인에서 액션 그래디언트를 시험.** EXPO 의 edit actor 와 병행/대체.
   `rl/expo.py:select_from_chunks` 에 `∇_a Q` 스텝을 끼우는 형태이고, exec 블록 마스킹은
   `spec.index` 로 이미 된다.

## 7. 읽은 파일 (재확인용)

```
jaxrl_m/agents/continuous/calql.py                Cal-QL 래퍼 (use_calql=True 만 켠다, 17줄)
jaxrl_m/agents/continuous/cql.py:150-300          CQL/Cal-QL critic loss 본체
jaxrl_m/agents/continuous/parl_calql.py:196-240   critic 만 업데이트하는 update()
jaxrl_m/agents/continuous/action_optimization.py  ①global top-K ②local ∇_a Q ③선택
jaxrl_m/data/roboverse_dataset.py:203             calc_return_to_go (mc_returns)
train.py:171-270                                  distillation 배치 만들기
train.py:900-1131                                 offline→online 루프 + base policy distillation
configs/base_config.py:14-20, 85-130              PA-RL / Cal-QL 기본 하이퍼파라미터
configs/real_config.py:69-146                     실기(bridge) 설정 — cql_alpha 0.01, cql_n_actions 4
```

---

## 8. 학습 루프 구조 (train.py 실제 흐름)

### 8.1 오프라인 단계 — critic 만 학습한다

```
for epoch in range(num_offline_epochs):                       # 기본 1000
    for step in range(num_train_steps_per_offline_epoch):     # 기본 1000  → 총 1M critic 스텝
        batch = next(offline_iterator)                        # 오프라인 데이터만
        batch = add_base_policy_actions_to_batch(             # train.py:1202
            ..., num_base_policy_actions=32,
            add_to_next_observations=True,                    # s' 쪽도 필요 (TD 타깃)
            manual_cache_dir=base_policy_offline_cache_path)  # ← 오프라인은 디스크 캐시 사용 가능
        agent.update(batch)                                   # critic 만 (networks_to_update={"critic"})
```

- **base policy 는 전혀 학습되지 않는다.** distillation 블록은 `train.py:1027` 에서
  `i >= num_offline_epochs and num_online_epochs > 0` 로 막혀 있다.
- base policy 는 **액션 샘플 공급자**로만 쓰인다. 두 곳에 들어간다:
  1. **TD 타깃의 next action** — `PARLCalQLAgent.forward_policy` 가
     `action_optimization_sample_actions` 로 오버라이드돼 있어서(`parl_calql.py:101`),
     CQL 의 `_compute_next_actions` → `forward_policy_and_sample`(`cql.py:60`) 가
     **32개 샘플 → top-K 10 → ∇_a Q 상승** 을 거친 액션을 돌려준다. 즉 타깃 안에 정책 개선이
     들어 있다 (EXPO 의 후보 argmax 와 같은 역할).
  2. **CQL penalty 의 OOD 집합** — `logsumexp` 를 취할 액션 집합 = {랜덤 n개, 정책의 next action
     n개, 정책의 current action n개}.
- 그래서 오프라인이라도 batch 마다 base policy 샘플이 필요하고, 그 비용을 없애려고
  `--base_policy_offline_cache_path` 로 **미리 뽑아둔 캐시를 읽는다** (OpenVLA 는 별도 워커:
  `scripts/openvla_caching_worker.py`).

### 8.2 전환

```
if i == num_offline_epochs:
    agent = restart_agent_optimizer_state(agent)    # Adam 상태 리셋 (train.py:909)
    env_data_collection_policy_fn = get_policy_fn(agent, argmax=..., base_policy=...)
```

### 8.3 온라인 단계 — 에피소드 단위로 세 블록이 번갈아 돈다

```
for epoch in range(num_online_epochs):
    ① 롤아웃    num_online_trajectories_per_epoch 개를 action-optimized 정책으로 수집
    ② distill   base policy 학습:  online_env_steps × base_policy_utd 스텝
                  batch = 오프라인:온라인 = mixing_ratio(0.5)
                  action optimization 에 **데이터 액션도 후보로 넣고**
                  (dataset_actions_to_consider=batch["actions"], action_optimization.py:434)
                  argmax(또는 softmax) 한 것을 BC 타깃으로 → base policy supervised 업데이트
    ③ critic    online_env_steps × critic_utd 스텝, 배치는 50:50 혼합
```

핵심: **critic 과 policy 가 스텝 단위로 얽히지 않는다.** 에폭 안에서 블록이 분리돼 있고,
정책 개선은 ②(distillation) 로만 일어난다. actor loss / policy gradient 는 없다.

### 8.4 Cal-QL 판 vs IQL 판 — 우리에게 중요한 차이

| | `parl_calql` | `parl_iql` |
|---|---|---|
| critic 손실 | TD + **CQL penalty**(logsumexp − Q(s,a_data)) + Cal-QL 하한 clip | V: `expectile_τ(Q(s,a_data) − V(s))`, Q: `(r + γV(s') − Q)²` |
| TD 타깃 | `Q_target(s', a')`, **a' = action optimization 결과** | **`V(s')`** — 액션이 필요 없다 |
| 정책 샘플 필요? | **필요** (타깃 + OOD 집합) | **critic 손실에는 불필요** |
| actor | 학습 안 함 (`networks_to_update={"critic"}`) | 학습 안 함 (`train_actor=False`, `parl_iql.py:171`) |
| 하이퍼파라미터 | `cql_alpha`(실기 0.01), `cql_n_actions`(실기 4) | `expectile`(0.7) |

**이것이 우리 세팅에서 결정적이다:** IQL 판은 오프라인 critic 학습에 **VLA 호출이 0** 이다
(타깃이 `V(s')`). 지금 `offline_critic_0.py` 와 같은 비용으로 돌면서, expectile 로 in-support
정책 개선이 들어간다. Cal-QL 판은 후보 캐시(2.4GB)를 먼저 만들어야 한다.

→ **테스트 순서 권고: IQL 판 먼저, 그 다음 Cal-QL 판.**

우리 데이터에서 각각 추가로 필요한 것:

| | 추가 구현 | 우리 데이터에서 계산 |
|---|---|---|
| IQL 판 | V 네트워크(이미지 latent + state → 스칼라) + expectile 손실. **~20줄** | 없음 |
| Cal-QL 판 | CQL penalty + MC 하한 clip. **~40줄** | `mc_returns` = 성공 궤적 `γ_eff^(남은 매크로스텝)`, 실패 0<br>후보 캐시 `candidates.npy` (55,564 × 8 × 40 × 34 f32 = 2.4GB) |

주의: 우리 액션 다양성이 4% 라 **IQL 의 expectile 은 SARSA 로 퇴화할 수 있다**(같은 상태에 액션이
하나면 expectile = 그 점). 다만 타깃이 `min(2개)` 에서 `V` 로 바뀌면서 **REDQ 비관 편향이 사라지므로**
우리가 겪은 음수 고정점(`Q ≡ −0.013`)은 그것만으로도 풀릴 가능성이 있다. 그게 IQL 판을 먼저
돌려볼 실질적 이유다.
