# 오프라인에서 policy 를 뽑는 세 가지 방법 — PA-RL / QAM / Q-VGM

critic 은 이미 있다. 문제는 **그 critic 을 flow-matching action expert 에 어떻게 넣느냐** 다.
세 방법이 이 한 지점에서 갈린다.

관련 문서: [PA-RL.md](PA-RL.md) · [PARL-DISTILL.md](PARL-DISTILL.md) · [DIVL.md](DIVL.md) ·
[ACTION-OPT.md](ACTION-OPT.md)

출처:
- **PA-RL** — Mark et al. 2024. 코드 `third_party/PolicyAgnosticRL/`
- **QAM** — Li & Levine 2026 (arXiv 2601.14234). 코드 `third_party/qam/agents/qam.py`.
  LWD (`rl/2605...LWD.pdf`) 가 이걸 policy extraction 으로 쓴다
- **Q-VGM** — Wang et al. 2026 (arXiv 2606.08015). 코드 없음, 논문 `rl/2606...Q-VGM.pdf`

---

## 0. 먼저: 왜 어려운가

flow policy 는 액션을 **한 번에 안 낸다.** 노이즈에서 시작해 velocity field `v_θ` 를 K번
적분해서 액션 청크를 만든다.

```
x[0] ~ N(0,I)  →  x[1]  →  ...  →  x[K] ≈ A        (K = 10 스텝쯤)
x[k+1] = x[k] + (τ_{k+1} − τ_k) · v_θ(x[k], τ_k, c)
```

여기서 두 가지가 막힌다.

**(1) likelihood 가 없다.** PPO/GRPO 같은 policy gradient 는 `log π(a|s)` 가 필요한데,
반복 디노이징은 그걸 안 내놓는다. 그래서 SAC 를 그대로 쓸 수 없다 — **EXPO-FT 가 SAC 를
쓸 수 있었던 건 액션을 내는 부분이 flow 가 아니라 tanh-Gaussian residual policy 였기
때문이다.** 이 차이가 이 문서 전체를 지배한다.

**(2) 체인을 타고 미분하면 터진다.** `∇_θ Q(s, x[K])` 를 구하려면 K개 스텝 전부를 통과해야
한다 (Diffusion-QL 방식). Q-VGM 이 LIBERO 에서 재보니 SFT 79.0% → **72.6%** 로 오히려
**나빠졌다.** VLA 규모에서는 쓸 수 없다.

그래서 세 방법 전부 **체인 미분을 피한다.** 피하는 방식이 다르다.

| | critic 신호를 무엇으로 바꾸나 | velocity field 를 직접 가르치나 | 추론 시 critic |
|---|---|---|---|
| **PA-RL** | 개선된 **최종 액션** (라벨) | ✗ (BC 라벨만 교체) | 불필요 |
| **QAM** | **adjoint state** (스텝별 회귀 타깃) | ✓ | 불필요 |
| **Q-VGM** | **residual velocity** (스텝별 회귀 타깃) | ✓ | 불필요 |

---

## 1. PA-RL — "라벨을 갈아끼운다"

### 하는 일

1. base policy 에서 후보 M=32개 샘플
2. 앙상블 mean Q 로 상위 K=10 남기고, 데이터셋 액션을 후보에 하나 끼워 넣는다
3. `a ← a + α∇_a Q̄` 를 10스텝
4. argmax (또는 `Categorical(Q)`) 로 하나 고른다
5. **그 액션을 BC 타깃으로 쓴다.** 손실 함수는 원래 것 그대로

```python
# third_party/PolicyAgnosticRL/train.py:253, 1122
batch["actions"] = action_distribution.sample(seed=rng)     # 라벨 교체
base_policy_agent.update(batch)                             # 원래 BC 손실
```

### RL 인가 BC 인가 — 지적하신 그대로다

**BC 다.** 정확히는 **advantage-weighted BC 계열**이다. 손실은 `‖v_θ − (a* − ε)‖²` 로
flow matching 그대로이고, 바뀐 건 `a*` 뿐이다.

이걸 RL 로 보는 관점은 이렇다. KL 정규화 policy improvement 의 최적해는

```
π*(a|s) ∝ π_β(a|s) · exp(Q(s,a)/λ)
```

이고, "π_β 에서 M개 샘플 → Q 로 가중 선택" 은 이 분포에서 샘플링하는 **importance
sampling 근사**다. 거기서 뽑은 샘플로 BC 하면 π* 로 수렴한다. 그래서 AWR/AWAC/IDQL 과
같은 족보에 있다. PA-RL 은 여기에 `∇_a Q` 상승을 얹어 후보를 π_β 밖으로 조금 밀어낸 것이다.

**SAC 와 다른 점**은 명확하다. SAC 는 `∇_θ E[Q(s, π_θ(s))]` 를 정책 파라미터에 직접
흘린다 (정책이 Q 를 최대화하는 방향으로 움직인다). PA-RL 은 gradient 를 **액션 공간에서만**
쓰고 정책에는 지도학습으로 전달한다. 그래서 "policy-agnostic" 이다 — 정책이 미분가능하지
않아도, 심지어 다른 사람이 만든 블랙박스여도 된다.

### 세팅

**offline 과 online 둘 다.** 단 코드의 distillation 은 **online 전용**이다:

```python
# train.py:1027
and i >= FLAGS.num_offline_epochs      # 오프라인 단계에서는 정책을 안 건드린다
```

오프라인에서 PA-RL 이 하는 것은 critic 학습뿐이고, 정책은 "추론 때 후보 뽑아 critic 으로
고르기" 로만 쓴다. 우리가 `PARL-DISTILL.md` 에서 하려는 오프라인 distillation 은 논문 밖의
변형이다 (IDQL 에 gradient refinement 를 얹은 형태로 보면 근거는 있다).

### 약점

Q-VGM 이 지적하는 부분: **velocity field 를 가르치지 않는다.** flow 는 노이즈에서 액션까지의
*경로*를 배우는 모델인데, 최종점 라벨만 바꾸면 그 라벨이 다시 K스텝 경로 전체에 퍼져야 한다.
간접적이고 느리다. LIBERO 실측 88.8% (Q-VGM 92.5%).

우리 세팅의 추가 문제는 **OpenVLA 는 액션 토큰 256-bin CE 라 양자화 손실이 있다**는 점인데,
RLDX-1 은 flow matching 이라 이건 해당 없다.

---

## 2. QAM — "최적 제어의 정확한 해를 스텝별 회귀로 바꾼다"

### 핵심 아이디어

목표는 같은 tilted 분포다:

```
π*(a|s) ∝ π_β(a|s) · exp(Q(s,a)/λ)
```

PA-RL 은 이걸 샘플링으로 근사했다. QAM 은 **"이 분포를 정확히 내는 velocity field 는
무엇인가" 를 푼다.** Adjoint Matching (Domingo-Enrich et al. 2024) 이 그 답을 준다:
memoryless SDE 로 재정의하면, 최적 보정 velocity 가 **adjoint state** 의 스텝별 회귀로
얻어진다.

```
f_δ(s, a_w, w) = f_θ(s, a_w, w) − f_β(s, a_w, w)          잔차 velocity

L_QAM(θ) = E ∫ ‖ 2·f_δ/σ_w + σ_w·g̃_w ‖²  dw            (LWD Eq. 9)

σ_w = √(2(1−w)w)
g̃_1 = −∇_a[ Q_φ(s, a_1)/λ ]                              종단 조건 (LWD Eq. 10)
```

`g̃_w` 는 `g̃_1` 에서 시작해 flow 를 **거꾸로** 적분해 얻는다 (adjoint ODE). `∇_a Q` 는
**최종 액션에서 딱 한 번**만 계산하고, 그걸 역방향 VJP 로 각 스텝에 전파한다. 체인 미분이
아니라 **한 번의 gradient + 역방향 전파**다.

### 코드에서 (`third_party/qam/agents/qam.py`)

**정방향 — memoryless SDE 로 샘플** (`adj_matching`, `qam.py:49-77`):

```python
h = 1 / flow_steps
for i in range(flow_steps):
    t = i / flow_steps
    sigma = jnp.sqrt(2 * (1 - t + h) / (t + h))
    v = self.network.select("actor_fast")(obs, x, t)
    x = x + h * (2*v - x/(t+h)) + jnp.sqrt(h) * sigma * noise   # ← SDE. ODE 가 아니다
    xs.append(x)
```

`2v − x/(t+h)` 와 노이즈 주입이 memoryless SDE 다. Adjoint Matching 이 성립하려면 **반드시
이 SDE 여야 한다** — 보통의 ODE 샘플링으로는 안 된다. 마지막 스텝만 ODE 로 마감한다.

**종단 조건 — `∇_a Q` 한 번** (`qam.py:81-85`):

```python
grad_fn = jax.grad(lambda x, y: critic(x, jnp.clip(y, -1., 1.)).mean(axis=0).sum(), 1)
adj = -grad_fn(obs, xs[-1]) * self.config["inv_temp"]      # inv_temp = 1/λ
```

앙상블 **mean** 을 쓰고 `clip_adj=True` 로 ±1 클립 후 미분한다.

**역방향 — adjoint ODE** (`qam.py:92-100`):

```python
for i in reversed(range(flow_steps)):
    def fn(xi):
        return 2 * actor_slow(obs, xi, t + h) - xi / (t + h)
    vjp = jax.vjp(fn, xs[i])[1](adj)[0]        # VJP 한 번 (전체 체인 미분이 아니다)
    adj = adj + h * vjp
```

**손실** (`qam.py:143`):

```python
adj_loss = jnp.sum(jnp.square((vf_fine - vf_base) * 2 / sigmas + sigmas * adjs), axis=-1)
```

`actor_slow` 가 BC flow (동시에 flow matching 손실로 계속 학습), `actor_fast` 가 보정 항.
`residual=True` 면 `actor_fast` 가 순수 잔차, `False` 면 `actor_fast` 가 전체를 내고
`vf_base` 를 빼서 잔차를 만든다.

**critic** (`qam.py:29-36`): n-step chunk TD + 앙상블 비관 backup

```python
next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)   # rho=0.5
target_q = rewards + discount**horizon_length * masks * next_q
```

`sample_actions` 로 다음 액션을 뽑아 backup 하므로 **IQL 이 아니라 SAC 계열의 actor-critic
backup** 이다. 이게 offline-to-online 을 노린 설계다.

### RL 인가

**진짜 RL 이다.** 세 가지가 근거다.

1. **critic backup 이 정책을 쓴다** (`next_actions = self.sample_actions(...)`). SARSA 가
   아니라 정책 개선이 backup 에 들어간다 — SAC 와 같은 구조다
2. **정책 손실이 지도학습이 아니다.** `g̃_w` 는 `∇_a Q` 에서 만들어진 값이고, 라벨이 아니라
   **critic 이 지정한 방향**이다. Q 가 바뀌면 타깃이 즉시 바뀐다
3. **정책 → critic → 정책 루프가 닫혀 있다**

SAC 와의 대응을 정확히 적으면:

| | SAC | QAM |
|---|---|---|
| 정책 개선 | `∇_θ E[Q(s, π_θ(s))]` — 정책을 통과하는 gradient | adjoint 회귀 — `∇_a Q` 를 스텝별 타깃으로 |
| 정규화 | entropy 보너스 | KL(π ‖ π_β), 온도 λ |
| 왜 다른가 | `π_θ` 가 미분 가능한 한 스텝 | flow 는 K스텝, 체인 미분 불가 |

즉 **QAM = flow policy 를 위한 SAC** 다. entropy 대신 base policy 로부터의 KL 로
정규화하고, reparameterization gradient 대신 adjoint 회귀를 쓴다.

### 세팅

**offline-to-online 둘 다.** 코드가 명시적으로 두 단계다 (`main.py:24-25`
`offline_steps=1e6`, `online_steps=5e5`). LWD 도 offline 사전학습 → online 함대 배포로
같은 구조를 쓴다.

### 보조 장치 두 개 (선택)

QAM 논문은 세 변종을 낸다 (README 의 재현 명령어):

- **QAM** — adjoint matching 만. `inv_temp=3.0`
- **QAM_FQL** — `fql_alpha=300` — flow policy 를 **one-step 정책으로 distill** 하면서
  동시에 그 one-step 정책의 Q 를 최대화 (`fql_q_loss = -fql_q.mean()`). 추론이 1스텝이 된다
- **QAM_EDIT** — `edit_scale=0.1` — flow 출력에 **tanh-Gaussian edit policy** 를 얹고
  SAC 식으로 학습 (`edit_q_loss + entropy + alpha` — 자동 온도까지). **EXPO-FT 의 residual
  policy 와 같은 것이다**

우리 코드에 이미 `rl/nets.py:291` 의 `ResidualActor` 가 있으니 QAM_EDIT 의 절반은 이미 있다.

### LWD 의 설정 (실제 하드웨어 숫자)

```
action chunk H = 30,  γ = 0.9999
정책 lr 2e-5 (AdamW, cosine),  critic/value lr 5e-4 (Adam, cosine)
QAM 온도 λ = 2
EMA 0.005
offline: τ_base = 0.6, α = 0.3 / online: τ_base = 0.9, α = 0.3   (DIVL)
value TD: 장기 태스크 10-step chunk TD, 단기 1-step / online 은 전부 1-step
online replay 는 offline:online = 1:1
데이터 652.5시간 (demo 51.6% / rollout 성공 13.6% / rollout 실패 6.0% / play 28.8%)
```

**초기화 순서**: π0.5 → demo BC → LWD(offline): QAM + DIVL → LWD(online): 함대 배포.

---

## 3. Q-VGM — "액션 개선량을 velocity 로 환산해서 가르친다"

### 핵심 아이디어

QAM 과 출발점이 같다 (KL 정규화 → 최적 제어). 최적 보정은 denoising-time value function 의
gradient 다:

```
ẋ_τ = v_base(x_τ, τ) + h(x_τ, τ)
h*(x_τ, τ) = β ∇_x V(x_τ, τ),   β = 1/λ                  (Q-VGM Eq. 2)
```

문제는 `V(x_τ, τ)` 를 중간(노이즈) 상태에서 모른다는 것. critic 은 **깨끗한 액션**만
평가한다. QAM 은 이걸 adjoint ODE 로 정확히 역전파해서 풀었다. Q-VGM 은 **근사로 우회한다.**

**핵심 근사 — 후반 스텝에서는 Jacobian ≈ I**

```
Â[k]_base = x[k] + (1−τ_k)·v_base(x[k], τ_k, ·)          Euler look-forward (Eq. 5)
V(x[k], τ_k) ≈ Q(s, Â[k]_base)

∇_x V ≈ [I + (1−τ_k)∇_x v_base]ᵀ ∇_A Q  ≈  ∇_A Q(s, Â[k]_base)      (Eq. 6)
```

`τ_k → 1` 에서 정확하고 `τ_k` 가 작아지면 나빠진다. 그래서 **마지막 M=5 스텝에만 적용한다.**

**그 다음이 PA-RL 과 똑같아진다:**

```
Â[k],0 = Â[k]_base
Â[k],j+1 = Â[k],j + α·clip_G(∇_A Q(s, Â[k],j))      J번 gradient 상승   (Eq. 7)
j* = argmax_j Q(s, Â[k],j)                          keep-best 선택
Â[k]_Q = Â[k],j*
```

`j=0` 이 원본이므로 keep-best 는 **개선이 없으면 원본으로 되돌아간다.** PA-RL 이 후보에
데이터셋 액션을 끼워 넣은 것과 같은 안전장치다. 논문은 이걸 "local Q landscape 위의 이산
line search — 샘플별 적응적 β_eff" 로 설명한다.

**PA-RL 과 갈라지는 지점 — 액션이 아니라 velocity 를 가르친다:**

```
ĥ[k]_Q = (Â[k]_Q − Â[k]_base) / (1 − τ_k)                             (Eq. 8)

L_align = Σ_k m_k · ‖ (v_θ(x[k],τ_k) − v_base(x[k],τ_k)) − ĥ[k]_Q ‖²   (Eq. 9)
```

"남은 시간 `1−τ_k` 동안 이만큼 더 가야 하니까, velocity 를 이만큼 바꿔라." 액션 개선량을
**velocity 단위로 환산**한 것이다. 타깃은 전부 detach 되고 gradient 는 `v_θ` 로만 흐른다.

### 알고리즘 (논문 Algorithm 1)

```
for each training iteration:
  s, x[0] ~ N(0,I);  L ← 0
  for k = 0..K−1:
    v[k] ← v_θ(x[k], τ_k)
    if m_k == 1:                                    # 마지막 M=5 스텝만
      Â ← x[k] + (1−τ_k)·v_base(x[k], τ_k)          # look-forward
      for j = 0..J−1:
        Â ← Â + α·clip_G(∇_A Q(s, Â))               # Q 상승
      j* ← argmax_j Q(s, Â[j]);  Â_Q ← Â[j*]        # keep-best
      ĥ_Q ← (Â_Q − Â[0]) / (1−τ_k)
      h_θ ← v[k] − v_base(x[k], τ_k)
      L += ‖h_θ − sg[ĥ_Q]‖²
    x[k+1] ← sg[x[k] + (τ_{k+1}−τ_k)·v[k]]          # stop-gradient. 체인 미분 없음
  update θ
```

### critic 설계가 별도의 기여다

Q-VGM 은 critic 쪽에도 세 가지를 넣는다. **우리 상황과 직접 관련 있다:**

1. **RL token** — frozen VLM prefix 를 autoencoder 로 `z_rl ∈ R^2048` 로 압축. 재구성
   손실을 정규화로 유지: `L_critic = L_IQL + α_rec·‖decode(z_rl) − prefix‖²`.
   **우리의 cog token mean-pooling 이 이 자리다** (더 단순한 버전). 논문 ablation: ResNet
   인코더로 바꾸면 92.5% → **87.4%**. 우리가 ResNet 을 버린 게 맞았다는 증거다
2. **층마다 액션 재주입** — `z_rl` 이 액션보다 훨씬 고차원이라 critic 이 액션을 무시하기
   쉽다. 그래서 **모든 hidden layer 에 액션을 다시 넣는다.** ablation: 없으면 **88.2%**.
   `∇_A Q` 를 쓰는 방법 전부에 필수다 — 우리 critic 은 액션을 입력층에서 한 번만 받는다
3. **stepwise IQL** — 청크 전체에 값 하나가 아니라 위치별 `Q^(i)`, 청크 안에서 다음 위치로
   부트스트랩:
   ```
   y_i = r_i + γ(1−d_i)·V^(i+1)(s)        i < H−1
   y_i = r_i + γ(1−d_i)·V^(0)(s')         i = H−1   (청크 경계)
   ```
   스칼라가 필요하면 `Q(s,A) = Σ_i Q^(i)(s,A)`. Q 헤드 2개의 min (clipped double Q).
   ablation: 단일 헤드면 **90.1%**

### RL 인가

**RL 이다. 다만 QAM 보다 "off-policy 지도학습" 쪽으로 한 발 가 있다.**

- critic 은 **IQL** 이다 — backup 에 정책 샘플을 안 쓴다 (`V` 로 부트스트랩). 완전 off-policy.
  QAM 의 SAC 식 backup 과 다르다
- 정책 손실의 타깃 `ĥ_Q` 는 critic gradient 에서 매번 새로 만들어진다 — 고정 라벨이 아니다
- 하지만 정책은 critic 을 개선하지 않는다 (IQL backup 이 정책 독립). 즉 **actor-critic
  루프가 닫혀 있지 않다**

그래서 위치는 이렇다:

```
BC ────── PA-RL ────── Q-VGM ────── QAM ────── SAC
          라벨 교체     velocity     velocity   정책 통과
                        타깃 +       타깃 +      gradient
                        IQL critic   정책 backup
```

### 세팅

**완전 offline 이다.** 이게 QAM 과의 실질적 차이다.

```
few-shot SFT → 그 정책의 평가 롤아웃을 데이터로 → critic → offline 정책 개선
→ 재배포 → 새 롤아웃 수집 → 반복
```

online RL 이 아니라 **train-deploy-collect-retrain 루프**다. 각 라운드 안은 완전히
오프라인이다. LIBERO 에서 online RL fine-tuning 대비 **롤아웃 에피소드 400배 적게** 썼다.

### 성적 (LIBERO 4 suite, 태스크당 50 롤아웃)

| 방법 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| π0.5 few-shot SFT (출발점) | 85.6 | 84.8 | 83.4 | 62.2 | 79.0 |
| Test-time Q 선택 | 90.2 | 89.6 | 87.8 | 76.2 | 86.0 |
| Test-time Q guidance | 93.8 | 90.6 | 89.8 | 80.4 | 88.7 |
| **Q-개선 액션 distillation (= PA-RL)** | 93.4 | 91.8 | 90.6 | 79.2 | **88.8** |
| Diffusion-QL (체인 미분) | 79.8 | 74.6 | 80.4 | 55.6 | **72.6** ↓ |
| **Q-VGM** | 96.2 | 95.4 | 94.6 | 83.8 | **92.5** |

읽을 점 세 개:

- **체인 미분은 정책을 망친다** (79.0 → 72.6). QAM/Q-VGM 이 존재하는 이유
- **PA-RL distillation 도 잘 된다** (79.0 → 88.8). velocity 를 안 가르쳐도 +9.8%p
- **velocity 를 가르치면 +3.7%p 더** (88.8 → 92.5)

ablation 에서 정책 쪽 세 개:

| 변종 | Avg |
|---|---|
| 전체 | 92.5 |
| keep-best 없음 | 88.6 |
| 전 스텝 정렬 (`m_k=1`) | 86.2 |
| frozen base anchor 없음 | 86.8 |

**전 스텝 정렬이 가장 나쁘다** (−6.3%p) — Jacobian≈I 근사가 초반 스텝에서 깨진다는 걸
직접 확인한 것이다. 마지막 M=5 스텝 제한이 핵심 설계다.

---

## 4. 세 방법 나란히

| | PA-RL | QAM | Q-VGM |
|---|---|---|---|
| **원리** | tilted 분포에서 IS 샘플링 → BC | tilted 분포의 최적 제어 정확해 | 최적 제어 + 후반 스텝 근사 |
| **critic 신호** | 개선된 최종 액션 | adjoint state (역방향 ODE) | residual velocity (look-forward) |
| **`∇_a Q` 계산 횟수** | 청크당 10회 (액션 공간) | 청크당 1회 + 역방향 VJP K회 | 스텝당 J회 × M=5 스텝 |
| **velocity field 지도** | ✗ | ✓ | ✓ |
| **샘플러 제약** | 없음 | **memoryless SDE 필수** | 없음 (ODE 그대로) |
| **critic backup** | Cal-QL/IQL 선택 | 정책 샘플 (SAC 식) | IQL (stepwise) |
| **정책 클래스** | 무엇이든 (미분 불필요) | flow/diffusion | flow |
| **세팅** | offline critic + online distill | offline→online | **완전 offline** |
| **추론 비용** | 정책만 | 정책만 | 정책만 |
| **구현 난이도** | 낮음 | **높음** (SDE 샘플러 + adjoint) | 중간 |
| **LIBERO** | 88.8 | — (LWD 실기 95%) | 92.5 |
| **참조 코드** | ✓ jax | ✓ jax | ✗ |

### EXPO-FT 와의 관계

EXPO-FT 는 **SAC** 를 쓴다. 위 세 방법과 어긋나 보이지만, 실제로는 이렇다:

- EXPO-FT 는 flow policy 를 **건드리지 않는다.** base VLA 를 고정하고 그 출력에
  tanh-Gaussian **residual policy** 를 얹어 그것만 SAC 로 학습한다
- residual policy 는 미분 가능한 한 스텝이라 SAC 의 reparameterization gradient 가 그대로
  성립한다. flow 의 두 문제(likelihood 없음, 체인 미분)를 **정책을 안 건드림으로써** 피한다
- 대가는 표현력이다. 편집량이 `edit_scale` 로 묶여 있어 base 근처만 탐색한다 (우리 fuji
  설정은 0.01, EXPO 기본 0.2)

그리고 **QAM 이 정확히 이 조합을 제공한다** — `QAM_EDIT` (`edit_scale=0.1`) 는 adjoint
matching 으로 flow 자체를 개선하면서, 그 위에 EXPO 식 SAC residual policy 를 같이 얹는다
(`qam.py:174-200`: `edit_q_loss + edit_entropy_loss + edit_alpha_loss`, 자동 온도 조절까지).

```
EXPO-FT     : base 고정 + residual SAC
QAM_EDIT    : base 개선(adjoint) + residual SAC          ← EXPO 의 상위집합
QAM         : base 개선(adjoint)만
QAM_FQL     : base 개선 + one-step 증류 + Q 최대화
```

---

## 5. 우리 상황에서 무엇을 할 것인가

### 이미 갖춘 것

| | 상태 |
|---|---|
| chunk-level IQL critic | ✓ `rl/offline_iql.py` (분포형 128 bin, 앙상블 10) |
| frozen VLM feature | ✓ `cogfeat.npy` (cog token mean-pool 4096d) |
| `∇_a Q` 액션 최적화 | ✓ `rl/probe_actopt.py` (PA-RL 식, 마스킹) |
| PA-RL 액션 relabel | ✓ `rl/relabel_parl.py` |
| action expert LoRA | ✓ RLDX-1 내장 (`action_model_use_lora`) |
| residual policy (EXPO) | ✓ `rl/nets.py:291 ResidualActor` |

### 권하는 순서

**1단계 — PA-RL distillation (이미 만들어 둔 것)**

가장 싸고, LIBERO 에서 +9.8%p 를 보인 방법이다. `relabel_parl.py` → LoRA 학습.
여기서 개선이 안 나오면 critic 문제이므로, 더 정교한 방법으로 가도 소용없다.

**2단계 — Q-VGM 으로 갈아타기**

PA-RL 이 먹히면 Q-VGM 이 자연스러운 다음 단계다. 근거:

- **완전 offline 이다.** 우리 세팅(수집→학습→재배포)과 정확히 일치한다. QAM 은 online 을
  전제로 설계됐다
- **`relabel_parl.py` 의 부품을 거의 그대로 쓴다.** `∇_A Q` 상승 + keep-best 는 이미 있다.
  추가할 것은 look-forward(`Eq. 5`), velocity 환산(`Eq. 8`), 후반 M스텝 마스크뿐이다
- **샘플러를 안 바꾼다.** QAM 은 memoryless SDE 샘플러를 새로 넣어야 하는데, RLDX-1 의
  추론 경로(RTC prefix 주입 포함)를 건드리는 건 위험하다

Q-VGM 구현 시 필요한 변경:

```
(a) 액션 expert forward 를 스텝별로 열어야 한다 (x[k], τ_k, v_base 접근)
    → RLDX-1 의 flow 적분 루프에 훅. rl/vla_rldx.py 의 expanded() 와 같은 자리
(b) v_base = 학습 전 정책의 velocity. LoRA 를 끈 forward 를 한 번 더 = frozen base
    → PEFT 는 adapter 를 disable 할 수 있어서 같은 가중치로 둘 다 얻는다
(c) critic 의 층마다 액션 재주입          ← 우리 critic 은 입력층 1회. ablation −4.3%p
(d) stepwise Q^(i)                        ← 우리는 청크 전체 스칼라 1개
(e) 마스크: explore_groups × [latency, latency+replan) 는 그대로 유지
```

**(c) 와 (d) 는 critic 재학습이 필요하다.** 그래서 1단계를 먼저 돌려 보고, PA-RL 이
"critic 은 개선 방향을 알지만 distillation 이 잘 안 옮긴다" 로 판명될 때 착수하는 게 순서다.

**3단계 — QAM_EDIT 은 online 갈 때**

우리는 이미 EXPO 식 residual policy 가 있으므로, online 단계에서 base 를 adjoint matching
으로 개선하면서 residual 을 SAC 로 학습하는 조합이 자연스럽다. 다만 memoryless SDE 샘플러가
전제라 그때 착수한다.

### 가장 먼저 손볼 것

세 방법 **전부** `∇_A Q` 의 품질에 달려 있다. 그래서 방법 선택보다 먼저 확인할 것은
Q-VGM 의 critic ablation 두 개다:

- **층마다 액션 재주입** (−4.3%p). 우리 critic 은 액션을 **입력층에서 한 번만** concat 한다
  (`rl/nets.py:166-176` `StateActionValue`: `concat(latent, state, action)` → MLP → head). `z_rl` 4096 차원 대 액션 280 차원의 불균형은 우리가 더 심하다
  (Q-VGM 은 2048 대 청크). **`probe_actopt.py` 의 `ΔQ/ens.std` 가 1 미만이면 이게 원인일
  가능성이 높다**
- **stepwise Q** (−2.4%p). 우리는 219프레임 에피소드에 보상이 끝 1프레임뿐이라 청크 하나에
  값 하나는 신호가 얇다

둘 다 `rl/nets.py` 와 `rl/offline_iql.py` 수정이고, 어느 policy extraction 을 고르든
이득이다.

---

## 6. 구현 (2026-08 기준)

### 무엇을 학습하는가 — 세 방법

| | `parl` / `qvgm` | `edit` |
|---|---|---|
| 학습 대상 | **action expert LoRA** r=16 α=32 (4.78M = 0.38%) | **ResidualActor** (~0.5M) + Temperature 1개 |
| action expert | LoRA 로 바뀐다 | **안 건드린다** (영구 동결) |
| VLM 백본 | ❄ 동결 | ❄ 동결 |
| critic | ❄ 동결 | ❄ 동결 |
| 학습 루프에 VLA | 들어간다 (매 스텝 forward/backward) | **안 들어간다** (base 청크 캐시) |
| 추론 시 추가 비용 | 없음 (정책 자체가 좋아짐) | 0.5M MLP 한 번 |
| 편집 상한 | 없음 | `edit_scale` |

`edit` 는 EXPO-FT 의 residual policy 를 오프라인에서 학습하는 것이다:

```
a = a_base + edit_scale * tanh(...)
L = E[ α * log_prob − Q(s, a_base + edit) ],   α 는 목표 엔트로피(−out_dim/2)로 자동 조절
```

`rl/expo.py:313 update_residual_actor` + `update_temperature` 와 같은 식이다.
**base 가 영구 동결이므로 base 청크를 한 번 캐시하면 학습 루프에 VLA 가 들어가지 않는다** —
결정 8,367 × 후보 8 × 280차원 = **75MB**, 실측 13.5 결정/s 로 약 10분. 그 뒤로는 0.5M MLP
학습이라 초당 수천 스텝이다.

`edit_scale` 을 우리가 측정한 "1프레임 자연 변화"(REF1 = 0.02177/차원) 단위로 환산하면:

| | edit_scale | 최대 편집 |
|---|---|---|
| EXPO 기본 / openarm yaml | 0.2 | **9.2 프레임치** |
| fuji 설정 | 0.01 | **0.5 프레임치** ← 노이즈 이하. fuji 에서 edit 이 아무 일도 못 한 이유 |

9.2 프레임치면 청크(8프레임)를 통째로 바꿀 수 있는 폭이다.

**언제 무엇을 쓰나.** Q-VGM 논문 Table 1 에 정확히 이 비교가 있다 —
`Test-time Q guidance`(∇_A Q 국소 보정) **88.7** vs Q-VGM(velocity field 지도) **92.5**.
amortized edit policy 는 guidance 계열이므로 상한이 낮다. 대신 **수십 초**면 끝나므로
"critic 에 쓸 만한 액션 정보가 있는가" 를 배포 가능한 정책 수준에서 먼저 검증하는
관문으로 쓰기 좋다. QAM 도 둘을 대체가 아니라 **보완**으로 본다 (`QAM_EDIT`, `qam.py:174-200`
= adjoint matching **에 더해** edit policy).

덤: `README_cp.md` 에 "round 0 은 랜덤 residual 로 돌아 성공률이 base BC 보다 낮다" 고 적혀
있다. 오프라인에서 edit policy 를 학습해 두면 **EXPO round 0 이 랜덤이 아닌 residual 로
시작**한다 — 온라인 루프에 그대로 쓰이는 부품이라 버려지지 않는다.

### `parl` / `qvgm` 이 학습하는 것 (상세)

```
VLM 백본 (Qwen3-VL 36층 4096)         ❄ 완전 동결. gradient 그래프가 생기지 않는다
MSAT action expert (DiT 1.24B)        ❄ 동결 + LoRA 주입 (r=16, alpha=32)
  LoRA 대상: vl_qkv, vl_proj, sa_qkv, sa_proj, linear1, linear2
state_encoder / action_encoder / action_decoder / vlln   ❄ 동결
critic                                ❄ 동결 (별도로 학습해 둔 것)
residual / edit policy                ✗ 쓰지 않는다
```

`vla.setup_training(lr, lora=True)` (`rl/vla_rldx.py:209`) 가 RLDX-1 내장
`action_model_use_lora` 를 켜고 `tune_projector/tune_diffusion_model/tune_vlln=False` 로
잠근다. 학습 파라미터는 **약 4.8M** (전체 대비 0.4%), 저장되는 것은 `lora_*` 텐서뿐이다.

EXPO-FT 식 residual(edit) policy 는 **어느 쪽에도 없다.** 편집을 정책 가중치에 굽는 것이
policy extraction 의 목적이고, residual 은 온라인 SAC 단계용이다 (`rl/nets.py:291`
`ResidualActor` 는 그대로 남아 있다).

### RLDX-1 을 건드린 곳 — **없다**

`third_party/RLDX-1` 은 한 줄도 수정하지 않았다. 필요한 것을 전부 public 속성으로 재구성했다.

`rl/policy_flow.py` 가 `get_action_with_features` 안의 지역 클로저 `_dit_forward`
(`rldx/model/core/rldx.py:623-640`) 와 **같은 계산을 밖에서** 만든다:

```
action_encoder → (+position_embedding) → concat(state_features) → MSAT → action_decoder
```

가능한 이유를 확인해 두었다:

| 확인 | 값 | 왜 중요한가 |
|---|---|---|
| `get_action_with_features` 데코레이터 | `@torch.no_grad()` **없음** | gradient 가 흐른다 (`get_action` 쪽에만 걸려 있다) |
| 액션 입력 VJP 선례 | `rldx.py:701` RTC guided | 같은 DiT·bf16 에서 이미 돌아가는 연산 |
| `use_physics` | `False` | physics 토큰이 NoOp → 샘플러가 궤적 의존이 아니다 |
| `use_memory` | `False` | 백본 출력이 프레임 독립 |
| `action_dim` (DiT 내부) | `max_action_dim = 64` | 적분은 패딩된 64차원에서. 28차원으로 하면 추론과 다른 분포 |
| `action_horizon` | 16 | 롤아웃 토큰 수 |
| `num_inference_timesteps` | **4** | 배포 K=4. 학습은 `--flow-steps` 로 따로 준다 |
| MSAT dropout | 0.2 (+`final_dropout`) | `flow.eval_mode()` 로 끈다 — `v_base` 가 확률적이면 타깃이 노이즈에 묻힌다 |
| `v_base` | LoRA adapter off | `peft BaseTunerLayer.enable_adapters(False)`. 사본 없음 → VRAM 증가 0 |

### 새로 만든 파일

| 파일 | 역할 |
|---|---|
| `rl/policy_flow.py` | 미분 가능한 velocity field. `FlowPolicy.context/velocity/rollout`, `adapters_disabled`, `chunk_mask` |
| `rl/offline_iql_qvgm.py` | Q-VGM 식 critic — stepwise IQL + 층마다 액션 재주입 |
| `rl/train_policy.py` | `--method parl \| qvgm \| edit` 정책 학습기 |
| `configs/exp/openarm_rim_policy.yaml` | 서빙용 (`N: 1`, `n_edit_samples: 0` 두 줄만 다름) |
| `sbatch/offline_rl/policy_{1..4}_*.sbatch` | 4개 실험 |

### 손댄 기존 파일

| 파일 | 변경 |
|---|---|
| `rl/nets.py` | `StepwiseQ` / `StepwiseEnsemble` / `StepwiseV` 추가 (층마다 액션 재주입) |
| `rl/critic_io.py` | `StepwiseCritic` + `load_stepwise_critic` 추가 |
| `rl/relabel_parl.py` | `--npy-out` / `--no-parquet` — relabel 결과를 `(T, action_dim)` npy 로 저장해 데이터셋 사본 없이 학습 가능 |
| `rl/offline_iql.py` | 체크포인트에 `features` / `feat_mu` / `feat_sd` 저장 (다운스트림이 같은 latent 를 재현하려면 필요), `--train-eps all\|success\|fail`, 태그에 holdout·train-eps 반영, 평가 곡선 5개(Q min/mean, V, A 2종), 비디오 주기를 `--eval-every` 로 통일, stride/fps 를 데이터셋에서 자동 계산 |
| `rl/probe_actopt.py` | `--features` (cog critic 지원), `--ascend mean\|min` (상승과 표시를 같은 축약으로), OOD 점수 눈금 (로그/셔플/난수 3점 보정), jerk·프레임환산 지표 |
| `rl/offline_critic.py`, `rl/offline_critic_0.py` | `normalize_all` 이 콜러블도 받는다 → **오프라인 학습에 모델 가중치 13.8GB 가 더 이상 필요 없다** (processor 만으로 actnorm 을 굽는다) |

### 검증한 것 (CPU, GPU 없이)

- **stepwise 보상 중복**: 청크 창을 `ep_end` 로 클램프하면 성공 종단 보상 1 이 **1,085 프레임에서
  최대 8번 복제**된다. 범위 밖을 `inb` 로 죽여 중복 0 을 확인
- **`chunk_mask` 좌표**: `spec.index` 는 `(latency+replan, 28)` 평탄 인덱스인데 DiT 는
  `(16, 64)` 다. `view(-1)[:280]` 로 심으면 **64개 관절 전부가 뒤섞인다.** 좌표 변환 후
  활성 스텝 `[2..9]`, 활성 관절 `[2..8, 16..21]` 로 정확히 일치
- **액션 층마다 재주입**: `|∇_a Q|` 4.96 vs 3.49 (미주입) — 민감도가 실제로 올라간다
- 전 모듈 `argparse` 참조 감사 + 미정의 전역 검사 통과

**GPU 는 이 셸에서 안 보여서 (`nvidia-smi` 미응답) forward/backward 실측은 못 했다.**
첫 실행은 `--steps 5 --batch 2` 로 shape 을 확인하고 늘릴 것.

---

## 7. 실행 순서

### 공통 (한 번만)

```bash
cd /rlwrld2/home/junmo_cho/ws/rd-rl
export L_PY="third_party/RLDX-1/.venv/bin/python"
export L_ENV='PYTHONPATH=$PWD/third_party/RLDX-1:$PWD NO_ALBUMENTATIONS_UPDATE=1'

# (0) cog feature — 이미 뽑았으면 건너뛴다 (images.mm 도 여기서 만들어진다)
PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" NO_ALBUMENTATIONS_UPDATE=1 $L_PY -u -m rl.extract_cogfeat \
  --exp openarm_rim --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints --batch 64 --resume
```

> **전제**: `configs/exp/openarm_rim.yaml` 의 `base_policy` 가 **데이터를 만든 체크포인트**
> 여야 한다. 지금 `checkpoint-meta.yml` 은 `0825_openarm` step 60000 인데 yaml 은 다른 런을
> 가리키고, 그 디렉토리에는 safetensors 샤드가 없다 (메타데이터만). policy 학습은 가중치가
> 반드시 필요하다 — `--model-path` 로 지정하거나 yaml 을 고칠 것.

### A. PA-RL

```bash
# (1) critic — 기존 offline_iql
$L_PY -u -m rl.offline_iql \
  --exp openarm_rim --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints --features cogfeat.npy \
  --discount 0.995 --bins 128 --expectile 0.7 \
  --steps 30000 --holdout 0.1 --eval-every 3000 --video-eps 30

# (2) 관문 — 액션 최적화가 쓸 만한지 (POLICY-EXTRACTION.md 5절 기준)
$L_PY -u -m rl.probe_actopt \
  --exp openarm_rim --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints --features cogfeat.npy \
  --critic iql-cog-dist128-t07-g0995-q10all-s0-h01/critic_latest.pt \
  --holdout 0.1 --auto-step 0.1 --video-eps 6
#   통과 조건: OOD p95 < shuffled 수준  AND  ΔQ/ens.std 중앙 > 1

# (3) 액션 relabel — 먼저 dry-run 으로 통계만
$L_PY -u -m rl.relabel_parl \
  --exp openarm_rim --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints --features cogfeat.npy \
  --critic iql-cog-dist128-t07-g0995-q10all-s0-h01/critic_latest.pt \
  --num-samples 32 --num-keep 10 --num-steps 10 --auto-step 0.05 --temp 0 \
  --dry-run --limit 200
#   "로그된 액션이 이긴 비율" 이 70% 넘으면 distillation 이 거의 no-op 이다

# (4) 본 relabel — npy 만 (rl.train_policy 로 학습)
$L_PY -u -m rl.relabel_parl \
  --exp openarm_rim --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints --features cogfeat.npy \
  --critic iql-cog-dist128-t07-g0995-q10all-s0-h01/critic_latest.pt \
  --auto-step 0.05 --temp 0 --no-parquet
#   → checkpoints/openarm_rim-critic/parl_actions.npy

# (5) LoRA distillation
$L_PY -u -m rl.train_policy --method parl \
  --exp openarm_rim --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints --targets parl_actions.npy \
  --steps 2000 --batch 8 --lr 1e-4 --lora-rank 16 --holdout 0.1
#   → checkpoints/openarm_rim-policy/parl-lora16-lr0.0001-s0/lora_latest.pt
```

**RLDX-1 자체 트레이너로 돌리는 대안** (`--no-parquet` 을 빼면 데이터셋 사본이 생긴다):

```bash
cd third_party/RLDX-1
.venv/bin/python -m rldx.experiment.launch_train \
  --base-model-path <데이터를 만든 체크포인트> \
  --dataset-path ../../rl-dataset/0825_openarm_f1_inference-parl \
  --embodiment-tag general_embodiment --action-model-use-lora True \
  --tune-projector False --tune-llm False --tune-visual False \
  --learning-rate 1e-4 --global-batch-size 64 --max-steps 3000 \
  --save-steps 500 --save-trainable-only True \
  --output-dir ../../checkpoints/openarm_parl_lora
```

둘의 차이: `rl.train_policy` 는 결정 프레임만 쓰고 우리 이미지 캐시를 타므로 빠르고 Q-VGM 과
직접 비교된다. RLDX-1 트레이너는 증강·EMA·스케줄러 등 원본 학습 설정을 그대로 쓴다.
**손실 함수와 RTC prefix 샘플링은 양쪽 다 RLDX-1 원본** (`rldx.py:438`) 이다 —
`rl.train_policy --method parl` 도 `vla.train_step` → `model.forward` 를 부른다.

### B. Q-VGM

```bash
# (1) critic — stepwise IQL (별도 파일)
$L_PY -u -m rl.offline_iql_qvgm \
  --exp openarm_rim --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints --features cogfeat.npy \
  --discount 0.995 --expectile 0.7 --num-qs 2 \
  --steps 30000 --holdout 0.1 --eval-every 3000
#   → checkpoints/openarm_rim-critic/qvgm-cog-t07-g0995-q2-s0-h01/critic_latest.pt

# (1b) ablation 이 필요하면
$L_PY -u -m rl.offline_iql_qvgm ... --no-inject      # 층마다 재주입 끄기 (-4.3%p 예상)

# (2) value-gradient matching
$L_PY -u -m rl.train_policy --method qvgm \
  --exp openarm_rim --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints \
  --critic qvgm-cog-t07-g0995-q2-s0-h01/critic_latest.pt \
  --flow-steps 10 --late-steps 5 --ascent-steps 4 --auto-step 0.05 \
  --steps 2000 --batch 8 --lr 1e-4 --lora-rank 16 --holdout 0.1
#   → checkpoints/openarm_rim-policy/qvgm-lora16-lr0.0001-K10M5J4-s0/lora_latest.pt
```

첫 실행은 반드시 작게:

```bash
$L_PY -u -m rl.train_policy --method qvgm ... --steps 5 --batch 2
```

로그에서 확인할 것:

```
[qvgm] DiT 공간 (16, 64) 중 편집 104개 = spec.index 104개  (일치해야 함)
[auto-step] median‖g‖/√d = ... → α = ...
  step    20  loss ...  ΔQ +0.0xxx   ...s/step
  [eval] step   250  Q(정책) +x.xxx  Q(로그) +x.xxx  차이 +x.xxx
```

- `ΔQ` = keep-best 가 look-forward 액션을 얼마나 개선했는가. **0 이면 학습 신호가 없다**
  (α 를 올리거나 critic 을 의심)
- `[eval]` 의 **차이** 가 학습이 진행되며 커져야 한다 — 정책이 로그 액션보다 좋은 액션을 내기
  시작했다는 뜻

### C. 추론용 병합

`rl.train_policy` 는 LoRA 텐서만 저장한다. RLDX-1 추론 로더가 그대로 읽는 통짜 체크포인트로
합치려면 RLDX-1 의 병합 스크립트를 쓴다 (LoRA 를 base Linear 에 흡수하고 PEFT 래퍼를 벗긴다):

```bash
cd third_party/RLDX-1
.venv/bin/python scripts/merge_lora_checkpoint.py --help
```

`rl.train_policy` 산출물은 `{"lora": state_dict, ...}` 형식이므로, 그 스크립트가 기대하는
`--trainable-ckpt` 레이아웃으로 풀어 주는 변환이 한 단계 필요하다 (아직 안 만들었다).
`ExpoServer` 로 바로 서빙할 때는 base 모델을 올린 뒤 `lora` state_dict 를 `load_state_dict(...,
strict=False)` 로 얹으면 된다.

---

## 8. 하이퍼파라미터 정리

| | PA-RL (우리) | Q-VGM (우리) | 논문 값 |
|---|---|---|---|
| 학습 대상 | action expert LoRA r=16 α=32 | 동일 | PA-RL: OpenVLA 전체 LoRA r=32 / Q-VGM: action expert |
| lr | 1e-4 | 1e-4 | LWD 2e-5 (AdamW+cosine) |
| batch | 8 (결정 프레임) | 8 | LWD global 64 |
| steps | 2000 (~1 epoch) | 2000 | PA-RL: online UTD 1 |
| 후보 수 M | 32 | — (look-forward 1개) | PA-RL 32 |
| top-K | 10 (+데이터 액션 1) | — | PA-RL 10 |
| ∇_A Q 상승 | 10 스텝 | J=4 스텝/디노이징스텝 | PA-RL 10 / Q-VGM J |
| step size | `--auto-step 0.05` | `--auto-step 0.05` | PA-RL 3e-4 (우리 스케일에선 무의미) |
| 선택 | argmax (`--temp 0`) | keep-best | PA-RL softmax(Q) 기본 |
| 롤아웃 K | — | 10 (배포 4) | Q-VGM K |
| 후반 M 스텝 | — | 5 | Q-VGM M=5 |
| clip_G | — | 1.0 | Q-VGM clip_G |
| critic | 분포형 128bin, 앙상블 10, chunk 스칼라 | 스칼라, Q헤드 2 min, **stepwise 8** | Q-VGM: 2 헤드 min, stepwise H |
| critic γ | 0.995 (프레임) | 0.995 (프레임) | LWD 0.9999 (H=30) |
| expectile τ | 0.7 | 0.7 | LWD τ_base 0.6→0.9 (적응) |
| 편집 범위 | explore_groups × 스텝 [2,10) | 동일 | 논문은 전 차원 |

---

## 9. 실측 (A100 80GB, 2026-08-27)

| 단계 | 실측 | 비고 |
|---|---|---|
| `offline_iql_qvgm` | **7ms/step** | 30,000 스텝 ≈ 4분. 학습 파라미터 3.3M, 체크포인트 26MB |
| base policy 로드 | 95초 | 샤드 3개 13GB |
| LoRA 주입 | 4,784,128 / 1,249,410,896 = **0.38%** | 64 텐서, 체크포인트 **9.6MB** |
| `train_policy --method parl` | **0.58s/step** (batch 16) | 30,000 스텝 **약 4.8시간** |
| `train_policy --method qvgm` | **1.30s/step** (batch 16) | 30,000 스텝 **약 10.8시간** |
| LoRA 대상 필터 | `p_qkv/p_proj` 자동 제외 | physics 미사용이라 없는 모듈 |
| `relabel_parl` (결정 64개, M=8) | 끝까지 동작 | 아래 통계 |

`relabel_parl` 스모크 출력 (결정 64개, `--num-samples 8 --num-keep 4 --auto-step 0.05`):

```
ΔQ (선택 - 로그)  평균 +0.0992  중앙 +0.0254  p95 +0.3845  개선된 비율 100.0%
이동거리/차원      평균 0.1085  중앙 0.0835  p95 0.2540   (액션 공간 ±1)
로그된 액션이 이긴 비율 23.4%
raw 액션이 바뀐 프레임 506/65928 (0.8%)  최대 변화 0.9685
```

- **로그 액션 승률 23.4%** → 5절 관문의 70% 기준을 통과한다. critic 이 base policy 를
  개선할 여지를 찾고 있다
- **최대 변화 0.97 rad 은 놀랄 일이 아니다.** PA-RL 의 후보는 base policy 의 *다른 샘플*
  이므로 선택만으로도 로그 액션과 크게 다를 수 있다. gradient 편집량이 아니다
- 바뀐 프레임 506 = 결정 64개 × 8프레임 (= 512, 에피소드 경계에서 6개 잘림) ✓

`train_policy --method qvgm` 스모크 (45 스텝, 2000 스텝짜리 임시 critic):

```
[auto-step] median‖g‖/√d = 2.956e-03 → α = 4.228 (목표 이동 0.05/차원)
  step    20  loss 0.10308  ΔQ +0.0470  0.91s/step
  step    40  loss 0.05871  ΔQ +0.0295  0.88s/step
  [eval] step    45  Q(정책) +0.1233  Q(로그) +0.1095  차이 +0.0138
```

### 배치 크기 — 키워도 처리량이 안 늘어난다

batch 16/32/64 를 실측했다. **OOM 은 64 에서도 안 났지만, 스텝 시간이 배치에 정확히 비례한다:**

| batch | qvgm s/step | 샘플/s | parl s/step | 샘플/s |
|---|---|---|---|---|
| 16 | 1.30 | 12.3 | 0.58 | 27.6 |
| 32 | 2.29 | 14.0 | 1.09 | 29.4 |
| 64 | 4.39 | 14.6 | 2.17 | 29.5 |

처리량이 batch 16 에서 이미 포화다 (64 로 4배 키워도 +19% / +7%). 메모리가 아니라 **GPU 가
이미 포화**라는 뜻이고, 배치 선택은 속도 문제가 아니라 **gradient 노이즈** 문제가 된다.

같은 벽시계 시간에 batch 16 은 batch 64 보다 **4배 많은 옵티마이저 스텝**을 밟는다. LoRA
4.8M 파라미터에 30,000 스텝이면 batch 16 이 맞다. batch 32 로 올리면 qvgm 이 19시간이 되어
24시간 한도가 빡빡해진다.

읽는 법:

- **`ΔQ` 가 줄어드는 것이 정상이다** (+0.047 → +0.030). keep-best 가 look-forward 액션을
  개선할 여지가 줄었다 = 정책이 critic 이 원하는 액션에 가까워지고 있다. **0 에 붙으면 학습
  신호가 소진된 것**이고, 처음부터 0 이면 α 나 critic 을 의심해야 한다
- **`차이` 가 양수로 커져야 한다.** 45 스텝에서 이미 +0.0138 로 정책이 로그 액션을 앞선다
  (다만 이 critic 은 2000 스텝짜리 임시본이므로 수치 자체는 의미 없다)
- `loss` 는 velocity 정렬 오차다. ΔQ 가 줄면 타깃 `ĥ_Q` 도 작아지므로 같이 줄어든다

LoRA 주입 로그 (정상):

```
[ActionModel LoRA] Skipping absent target modules: ['p_qkv', 'p_proj']
[ActionModel LoRA] target_modules=['vl_qkv','vl_proj','sa_qkv','sa_proj','linear1','linear2'], r=16, alpha=32
[ActionModel LoRA] trainable params: 4784128 / 1249410896 (0.38%)
[MSAT] Tune action model projector: False / diffusion model: False / vlln: False
[정책] ... 학습 파라미터 4.78M (64 텐서), 백본 학습 텐서 0     ← 백본이 0 이어야 한다
[qvgm] DiT 공간 (16, 64) 중 편집 104개 = spec.index 104개  (일치해야 함)
[auto-step] median‖g‖/√d = 2.956e-03 → α = 4.228 (목표 이동 0.05/차원)
```

### 스모크 테스트에서 잡은 버그

**stepwise 헤드의 손실 가중치.** 에피소드 범위 밖 위치를 가중치 0 으로 두면 그 헤드가
아무 값이나 내고 `Σ_i Q^(i)` 가 오염된다. 실측: 성공 에피소드 마지막 프레임에서
**Q_sum = 3.04** (실제 리턴은 1). 그 위치의 타깃은 `reward 0 + γ·mask 0·V = 0` 으로
정확히 정의되므로 가중치를 1 로 두고 **0 으로 회귀**시키면 된다 → 수정 후 1.82 (수렴 중).
`done=1` 이 에피소드 마지막 프레임에 반드시 있으므로(실측 300/300) 경계 부트스트랩은
mask 로 자동 차단되고 별도 valid 플래그가 필요 없다.

---

## 10. 4개 실험 — critic × 추출 방식 2×2

### 먼저: **edit arm 은 PA-RL 도 Q-VGM 도 아니다**

Q-VGM 의 정의는 "액션 개선량을 residual **velocity** 로 환산해 `v_θ` 를 지도한다" 이므로
**DiT 에 gradient 가 들어가야 성립한다.** edit policy 는 DiT 를 전혀 건드리지 않고 그
출력에 유계 보정을 얹는다 — Q-VGM 논문이 자기 baseline 으로 둔 `Test-time Q guidance`
계열이다 (88.7 vs 92.5).

같은 이유로 `edit` 는 PA-RL 도 아니다. PA-RL 의 distillation **자체가** BC 단계이고,
그것을 SAC-on-edit 으로 바꾼 것은 다른 알고리즘(EXPO-FT / FQL 계열)이다.

그래서 4개는 이렇게 읽어야 한다:

| | **LoRA** (velocity field 를 바꾼다) | **edit policy** (DiT 동결 + 유계 보정) |
|---|---|---|
| **iql critic**<br>분포형 128bin, 앙상블 10, 청크 스칼라 | `policy_1_parl_lora`<br>= **PA-RL** | `policy_2_iqlcritic_edit`<br>= EXPO-FT 식 |
| **qvgm critic**<br>stepwise 8, Q헤드 2 min, 층마다 액션 재주입 | `policy_3_qvgm_lora`<br>= **Q-VGM** | `policy_4_qvgmcritic_edit`<br>= EXPO-FT 식 |

- 논문 방법은 **1번(PA-RL)과 3번(Q-VGM) 둘뿐이다**
- 2번·4번은 **같은 방법**(EXPO-FT 식 edit policy)을 서로 다른 critic 으로 돌린 것이다.
  재는 것은 "critic 설계가 더 좋은 `∇_A Q` 를 주는가" 이고, 추출은 값싼 방법으로 읽어낸다
- **행**을 비교하면 critic 설계 효과 (stepwise + 액션 재주입)
- **열**을 비교하면 velocity field 를 가르치는 것 vs 국소 보정 (Q-VGM 의 +3.8%p 주장)

### Q-VGM 은 별도 residual 네트워크를 쓰지 않는다 — action expert 를 미세조정한다

논문이 명시한다:

> **"Only the action expert is trained; inference runs `v_θ` alone, without critic queries or
> search."** (Fig. 2 캡션)
>
> **"We train the action expert so that its residual velocity matches `ĥ_Q`"** (4.2절)
>
> "At inference, the policy samples with `v_θ = v_base + h_θ`, so the critic guidance is
> amortized into the action expert" — `h_θ ≡ v_θ − v_base` 는 **표기**다. 별도 모듈이 아니라
> 미세조정된 field 와 원래 field 의 차이를 그렇게 부른 것이다

그래서 **LoRA 가 논문에 충실한 파라미터화다** (action expert 를 미세조정하는 값싼 방법).

`v = v_base + h` 를 **두 네트워크로 구현하는 것은 QAM** 이다 — `actor_slow`(BC flow) +
`actor_fast`(보정), `residual=True` (`qam.py:139-143`). Q-VGM 의 손실에 QAM 의 파라미터화를
붙이는 변형은 가능하지만 세 가지가 걸린다:

1. **값싸지지 않는다.** `h_θ(x[k], τ_k, c)` 가 VL prefix 와 디노이징 중간 상태를 조건으로
   받아야 하므로 base 청크 캐시로 대체할 수 없고 매 스텝 백본이 필요하다 — LoRA 와 같은 비용
2. **새 모듈 설계가 필요하다.** 4096차원 prefix 에 cross-attention 을 해야 하니 사실상 작은
   DiT 를 하나 더 만드는 일이다. `action_model_use_lora` 를 켜는 것과 비교가 안 된다
3. **논문의 선택과 달라 비교 기준이 흐려진다**

LoRA 용량이 걱정이면 (rank 16 의 표현력) 새 모듈보다 `--lora-rank 32` 가 훨씬 싸다
(4.78M → 9.6M, 코드 변경 0).

### 던지는 순서 — 3번을 먼저

```bash
cd /rlwrld2/home/junmo_cho/ws/rd-rl
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/junmo_cho/
sbatch ./sbatch/offline_rl/policy_3_qvgm_lora.sbatch   # ← 먼저. qvgm critic 을 만든다 (12분)
sbatch ./sbatch/offline_rl/policy_1_parl_lora.sbatch
sbatch ./sbatch/offline_rl/policy_2_iqlcritic_edit.sbatch
sbatch ./sbatch/offline_rl/policy_4_qvgmcritic_edit.sbatch  # 3번의 critic 을 최대 90분 기다린다
```

4번에 대기 루프가 들어 있다 (30초마다 확인, 90분 타임아웃). `set -e` 와 `[ -f ] && break`
조합이 안전한지 실제로 확인했다 (bash 는 `&&` 리스트의 앞쪽 실패를 -e 대상에서 제외한다).

예상 시간: 1번 ~5시간, 2번 ~25분(캐시 10분 포함), 3번 ~11시간, 4번 ~25분.

### 산출물 이름

```
checkpoints/openarm_rim-policy/
  parl-lora16-lr0.0001-s0/               lora_latest.pt        9.6MB
  qvgm-lora16-lr0.0001-K10M5J4-s0/       lora_latest.pt        9.6MB
  edit-iql-n8-es0.2-lr0.0003-s0/         edit_latest.pt        ~6MB
  edit-qvgm-n8-es0.2-lr0.0003-s0/        edit_latest.pt        ~6MB
checkpoints/openarm_rim-critic/
  base_chunks_n8.npz                     75MB  (2번·4번 공유)
  parl_actions.npy                       7MB   (1번이 만든다)
```

edit 태그에 **critic 종류가 들어간다** — 안 넣으면 2번과 4번이 같은 디렉토리를 덮어쓴다.

---

## 11. 학습한 정책을 서버로 띄우기

**세 방법이 저장하는 것이 다르므로 `--artifacts` 에 넣는 파일도 다르다.**

| 학습 | 저장 키 | 파일 | 서버가 하는 일 |
|---|---|---|---|
| `parl` / `qvgm` (LoRA) | `lora` | `lora_latest.pt` (9.6MB) | LoRA 를 주입하고 얹는다 → **base 정책 자체가 바뀐다** |
| `edit` | `residual`, `temp` | `edit_latest.pt` (~6MB) | 보정 정책을 얹는다 → base 는 그대로, 출력에 편집이 더해진다 |

`ExpoServer._load` (`rl/vla_rldx.py:381`) 가 "있는 키만 채운다" 라서 두 형식이 같은 플래그로
들어간다:

```python
pairs = {"enc": …encoder, "critic": …critic, "target": …target_critic,
         "residual": …residual, "temp": …temp}
for k, m in pairs.items():
    if k in sd: m.load_state_dict(sd[k])
if sd.get("lora"):
    self.vla.setup_training(lora=True)            # 먼저 주입해야 로드된다
    self.vla.model.load_state_dict(sd["lora"], strict=False)
```

### LoRA 판 — 오늘 그대로 된다

```bash
cd third_party/RLDX-1
B=$A_CKPT/openarm_0818_0819_0821_rh56f1_teleop_all598ep_egostereo_ptimg_framewt_drop03_rtc12tr_bs128_30k_4gpu_mlxp

# 1/4 PA-RL LoRA
ROS_DOMAIN_ID=106 PYTHONPATH="$PWD:$A_RL" pixi run -e rldx python -u -m rl.vla_rldx serve \
  --exp openarm_rim_policy \
  --model-path $B \
  --artifacts $A_CKPT/openarm_rim-policy/parl-lora16-lr0.0001-s0/lora_latest.pt \
  --host 127.0.0.1 --port 5555

# 3/4 Q-VGM LoRA — --artifacts 만 다르다
  --artifacts $A_CKPT/openarm_rim-policy/qvgm-lora16-lr0.0001-K10M5J4-s0/lora_latest.pt

# 기준선 (base BC) — --artifacts 를 아예 뺀다
```

- `--exp openarm_rim_policy` 를 써야 한다. 원본 `openarm_rim.yaml` 은 `N: 8` base 후보 +
  `n_edit_samples: 8` 을 만들어 **critic argmax 로 고르므로** distillation 결과만 재려면
  그 선택이 개입해선 안 된다. 서빙용 yaml 은 `N: 1, n_edit_samples: 0` 두 줄만 다르다
- 로그에 `[산출물] … lora(64텐서)` 가 나와야 실제로 얹힌 것이다
- `lora_latest.pt` 는 심볼릭 링크다 — actor 로 옮길 때 **실체 파일**을 복사할 것
- `meta.json` 이 없다는 sha256 경고는 정상이다 (learner 라운드 산출물이 아니므로)
- LoRA rank 는 서버가 체크포인트 config 의 기본값 **16** 으로 주입한다. `--lora-rank` 를
  16 이 아닌 값으로 학습했으면 shape 이 어긋난다

### edit 판 — **`ExpoServer` 를 고쳐야 한다** (아직 안 됨)

`edit_latest.pt` 를 그대로 `--artifacts` 에 넣으면 `residual`/`temp` 는 로드되지만 **critic
쪽에서 막힌다.** 그리고 이건 edit 만의 문제가 아니다 — **지금 `ExpoServer` 는 cog feature
critic 자체를 서빙할 수 없다.**

두 가지가 걸린다.

**(1) `_critic_obs` 가 이미지 → ResNet 인코더 경로다** (`rl/vla_rldx.py:440`). cog feature 로
학습한 critic·edit policy 는 학습 때 본 latent 를 서버에서 못 받는다.

고치는 것은 쉽다 — **cog feature 가 이미 계산돼 있다.** `get_action_with_features` 의 반환값에
`backbone_features` 가 들어 있고 (`rldx.py:729-733`), `PolicyRuntime._forward` 가 그 dict 를
그대로 돌려주므로 (`policy_runtime.py:405-409`) `_run_inference` 의 `pred` 에서 바로 꺼낼 수
있다. **백본을 한 번 더 돌 필요가 없다:**

```python
f = pred["backbone_features"][:1].clone()     # (1, seq, 4096). clone 은 inference_mode 탈출용
z = f[:, -n_cog:, :].float().mean(1)          # extract_cogfeat.py 와 같은 mean-pool
z = (z - feat_mu) / feat_sd                   # 학습 때의 표준화 (ckpt 에 저장해 뒀다)
lat = torch.cat([proj(z), state], -1)         # critic / edit policy 의 입력
```

**(2) 진짜 작업은 critic 을 다시 만드는 것이다.** `EXPOLearner` 가 만드는 critic 과 cog
체크포인트의 shape 이 다르다 — 실측:

```
ExpoServer 의 CriticEnsemble 입력차원 = 512(latent_dim_image) + 64(latent_dim_state) + 280 = 856
cog ckpt 의 실제 입력차원             = 512(Proj) + 28(state raw)   + 280 = 820
→ load_state_dict 실패
```

cog 판은 state 를 **raw 로** 붙이고 `include_state=False` 이기 때문이다. 게다가 qvgm critic 은
아예 다른 클래스(`StepwiseEnsemble`)다. 그래서 `ExpoServer` 가 자체 네트워크를 만드는 대신
`rl/critic_io.py` 의 로더를 쓰도록 바꿔야 한다.

**범위**: `rl/vla_rldx.py` 의 `__init__` / `_load` / `_critic_obs` / `_run_inference` 네 곳.
`third_party/RLDX-1` 은 건드리지 않는다. 서빙 경로라 로봇에서 확인이 필요하다.

**그때 같이 얻는 것**: cog critic 으로 EXPO 후보 선택이 실제로 동작하고 (지금 랜덤 ResNet 으로
고르고 있다), 서빙 경로에서 34.1M ResNet 인코더와 `_critic_obs` 의 이미지 리사이즈·종횡비
보정 로직이 전부 빠진다.

### EXPO 롤아웃과 함께 쓰려면

distill 된 정책 위에 **다시** critic 선택 + edit 을 얹고 싶으면 `--exp openarm_rim` (원본
yaml) 으로 띄우고 `--artifacts` 에 critic 까지 든 파일을 준다. 한 파일로 합치면 된다:

```python
import torch
lora = torch.load(".../lora_latest.pt")
edit = torch.load(".../edit_latest.pt")
crit = torch.load(".../critic_latest.pt")
torch.save({"lora": lora["lora"], "residual": edit["residual"], "temp": edit["temp"],
            "enc": crit["enc"], "critic": crit["critic"], "target": crit["target"]},
           ".../theta.pt")
```

키 이름이 `ExpoServer._load` 의 규약과 같으므로 그대로 읽힌다. 단 위 (1)(2) 를 고친 뒤에만
의미가 있다.

---

## 12. 지금까지 정리 (2026-08-27)

### 결정한 것

| 항목 | 값 | 근거 |
|---|---|---|
| 데이터셋 | `rl-dataset/0825_openarm_f1_inference` | 300ep / 65,928f / 20Hz / 성공 155 (52%) / 5세션 전부 혼합 |
| critic 특징 | cog token mean-pool (frozen VLM) | Q-VGM ablation: ResNet 인코더면 −5.1%p. fuji 에서 ResNet 이 에피소드를 암기했다 |
| 할인 γ | **0.995** (프레임 단위) | `1 − 1/L`, L=219f. 지평 204f = 에피소드의 93%. fuji 의 0.999 는 458% 로 과함 |
| 학습 스텝 (critic) | 100,000 | 7ms/step → 12분. parl 쪽 offline_iql 과 예산 맞춤 |
| 학습 스텝 (policy) | 30,000 | 아래 배치 실측 참고 |
| 배치 (policy) | **16** | 처리량이 16에서 포화. 64로 키워도 +7~19%. 같은 시간에 4배 많은 스텝 |
| 배치 (edit) | 256 (MLP), 캐시는 `--cache-batch 128` | edit 학습에 VLA 가 없다 |
| 편집 범위 | `explore_groups` × 청크 스텝 `[2,10)` = 280차원 중 104개 | 배포의 RTC 실행 구간과 일치 |
| LoRA | r=16 α=32, action expert 만 (4.78M = 0.38%) | RLDX-1 내장 기본값. 서버 로더도 같은 값으로 주입 |

### 실측 (A100 80GB)

```
offline_iql_qvgm (critic)     7ms/step      100k → 12분
base policy 로드              95초          샤드 3개 13GB
LoRA 주입                     4.78M/1.25B   0.38%, 64텐서, ckpt 9.6MB
train_policy --method parl    0.58s/step    batch 16 → 30k = 4.8시간
train_policy --method qvgm    1.30s/step    batch 16 → 30k = 10.8시간
train_policy --method edit    캐시 13.5 결정/s (10분) + MLP 학습은 초당 수천 스텝
relabel_parl                  ΔQ +0.099 / 개선 100% / 로그 액션 승률 23.4%
```

### 구현 중 잡은 버그 (전부 실측으로 확인)

1. **stepwise 보상 중복** — 청크 창을 `ep_end` 로 클램프하면 성공 종단 보상 1 이
   1,085 프레임에서 최대 8번 복제된다
2. **stepwise 손실 가중치** — 범위 밖 위치를 가중치 0 으로 두면 그 헤드가 자유롭게 떠서
   `Σ_i Q^(i)` 가 오염된다 (마지막 프레임 Q_sum 3.04, 실제 리턴 1). 타깃이 정확히 0 으로
   정의되므로 0 으로 회귀시킨다
3. **`chunk_mask` 좌표** — `spec.index` 는 `(latency+replan, 28)` 평탄 인덱스인데 DiT 는
   `(16, 64)` 다. `view(-1)[:280]` 으로 심으면 64개 관절 전부가 뒤섞인다
4. **`np.savez` 파일명** — 경로가 `.npz` 로 끝나지 않으면 `.npz` 를 덧붙인다 →
   `os.replace` 가 실패한다. 열린 핸들을 넘겨야 한다
5. **parquet 액션 컬럼 순서** — parquet 내부 순서가 canonical 과 다르다
   (openarm: `right_arm` 9:16 vs 15:22, `left_hand` 16:22 vs 9:15). 그대로 쓰면 relabel 이
   관절을 조용히 뒤섞는다. 300/300 에피소드 왕복 검증 통과
6. **`probe_actopt` 가 mean 으로 올리고 min 을 그렸다** — 화면의 곡선이 실제로 올리는 양이
   아니었다. `--ascend` 로 상승과 표시를 묶었다
7. **`offline_iql` 태그에 holdout 이 없었다** — 세션별 교차검증 두 런이 같은 디렉토리를
   덮어쓴다
8. **비디오 주기 불일치** — `--video-every 2500` 과 `--eval-every 3000` 의 최소공배수
   15000 에서만 걸렸다. `--eval-every` 로 통일
9. **`edit` 태그에 critic 종류가 없었다** — 2번·4번 실험이 같은 디렉토리를 덮어쓴다

### 아직 안 한 것

| | 상태 |
|---|---|
| **`ExpoServer` 를 cog feature critic 에 맞추기** | 11절 참고. edit 판 서빙의 전제이고, **지금 cog critic 으로 EXPO 롤아웃이 이미 깨져 있다** (랜덤 ResNet 으로 후보를 고른다). shape 856 vs 820 |
| LoRA → 통짜 체크포인트 병합 | `scripts/merge_lora_checkpoint.py` 가 기대하는 레이아웃으로 푸는 변환 한 단계 |
| Cal-QL | 우선순위 밖 |
| QAM | 11절 검증 결과 기술적으로 가능하지만 memoryless SDE 샘플러 + SAC backup 두 공사가 필요. online 단계에 적합 |
| 세션 단위 교차검증 (critic A 로 최적화 → B 로 채점) | `--holdout <세션이름>` 두 번 돌리면 된다 (태그가 분리됨) |

---

## 13. edit arm 에서 잡은 SAC 버그 두 개 (2026-08-27, 실측)

스모크에서 edit arm 이 **5,000 스텝 학습해도 이득 -0.0007** 로 아무 일도 하지 않았다.
원인이 두 겹이었다.

### ① 목표 엔트로피의 공간 불일치

`ResidualActor.sample` 은 EXPO-FT 규약대로 log_prob 에 `-out_dim·log(edit_scale)` 보정을
넣는다 (`rl/nets.py:409`). 그래서 보고되는 엔트로피는 **스케일 공간** 기준인데,
`ExploreSpec.target_entropy` 는 **tanh 공간** 기준(`-out_dim/2`)이다. `out_dim=104`,
`edit_scale=0.2` 에서:

```
보정량                      = -104·log(0.2) = +167.4
스케일 공간 달성 가능 최대   = 104·log(2·0.2) = -95.3
실측 엔트로피                = -96.3            ← 이미 최대치
목표 (-out_dim/2)            = -52.0            ← 최대값보다 높다 = 도달 불가
```

목표가 물리적으로 불가능하므로 alpha 가 무한히 커진다 (실측 6.3, 계속 증가).
같은 공간으로 옮기면 `-out_dim/2 + out_dim·log(edit_scale) = -219.4` 이다.

### ② 고쳐도 Q 를 무시한다 — Q 와 log_prob 의 스케일이 400배 어긋난다

목표를 고쳐 alpha 가 0.75 → 0.094 로 내려갔는데도 이득이 0 이었다. 이유:

| alpha | 엔트로피 항 `α·\|log_prob\|` | Q 항 | 비율 |
|---|---|---|---|
| 6.3 | 604.8 | 0.24 | 2,520배 |
| 0.094 | 9.0 | 0.24 | 38배 |
| 0.001 | 0.10 | 0.24 | 0.4배 |

`Q ∈ [0,1]` (실측 0.24) 인데 `log_prob` 은 104차원이라 ~96 이다. 표준 SAC 는 Q 가 리턴
스케일(10~1000)이라 균형이 맞지만 우리는 안 맞는다. 결정적 증거: **`|edit|` 이 10,000 스텝
내내 0.1176 으로 소수 4자리까지 고정**이었고, 그 값이 `RMS(uniform[-1,1])·edit_scale
= 0.577·0.2 = 0.1155` 와 일치했다 — 즉 **순수 랜덤 편집**에 머물러 있었다.

**해결: `--entropy-scale` 을 추가하고 기본값 0.** 오프라인 추출에는 탐색이 필요 없으므로
손실이 `-Q` 가 된다. 온라인 EXPO 로 갈 때는 다시 켜야 한다.

### 고친 뒤 (6,000 스텝, iql critic)

```
[eval] Q(base+edit) +0.3506  Q(base) +0.2423  Q(로그) +0.2425
       Q(난수편집) +0.2426   이득(vs base) +0.1084
       앙상블std 0.0572->0.0639 (1.12배)  편집량 0.1966 = 9.2 프레임치 (상한 9.3)
       이득/std 1.70
```

네 지표가 모두 좋은 방향이다:

- **이득 +0.1084** — Q 를 44% 올렸다
- **난수편집 = base 와 같다** (+0.2426 vs +0.2423) — 같은 크기의 랜덤 편집은 효과가 없다.
  즉 **크기가 아니라 방향** 때문에 오른 것이다. 이 대조군이 가장 중요한 확인이다
- **앙상블 std 1.12배** — critic 10개가 여전히 거의 일치한다 = 외삽이 아니다
- **이득/std 1.70 > 1** — 개선이 앙상블 노이즈보다 크다

**다만 편집량이 상한에 붙어 있다** (9.2 / 상한 9.3 프레임치). critic 이 `edit_scale` 보다 더
멀리 가고 싶어한다는 뜻이다. `--edit-scale 0.4` 로 올려 보면 갈린다 — 이득이 비례해 늘면서
std 가 유지되면 진짜 개선 여지가 더 있는 것이고, std 가 터지면 critic 착취다.

### 온라인 루프에도 같은 문제가 있을 수 있다

`rl/expo.py:313 update_residual_actor` 와 `update_temperature` 가 같은 규약을 쓴다.
`ExploreSpec` 은 온라인 경로가 공유하므로 건드리지 않고 `train_policy` 안에서만 보정했다.
`README_cp.md` 의 "round 0 은 랜덤 residual 로 돌아 성공률이 base BC 보다 낮다" 가 이것과
관련 있을 수 있다 — **EXPO 롤아웃 품질에 직접 영향이 있으므로 따로 확인할 것.**

### 태그 접미사 함정 (같이 잡음)

`offline_iql_qvgm` 의 `--holdout` 기본값이 `0.1` 이었다. 태그 접미사는 **기본값과 다를 때만**
붙으므로 `--holdout 0.1` 을 주면 `-h01` 이 안 붙는다. 그런데 `offline_iql` 은 기본값이 `0.2`
라서 같은 `--holdout 0.1` 에 `-h01` 이 붙는다 — 두 critic 의 디렉토리 이름이 비대칭이 되고,
sbatch 3/4 가 참조하는 경로가 어긋난다 (3번은 critic 을 못 찾고 4번은 90분 기다리다 죽는다).
`offline_iql_qvgm` 의 기본값을 `0.2` 로 맞췄다.

**교훈**: 태그 문자열을 sbatch 에 하드코딩하면 이런 게 생긴다. 던지기 전에 태그를
**정확히**(substring 아님) 예측·대조하는 검증을 돌릴 것.
