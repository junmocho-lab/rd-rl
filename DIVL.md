# DIVL (Distributional Implicit Value Learning) — 논문 정리 + 구현 설계

출처: **Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies**
(arXiv 2605.00416v2, AGIBOT Finch / Shanghai Innovation Institute, 2026-06-03)
로컬 원본: `rl/2605. Learning While Deploying- ....pdf`

LWD 는 두 조각으로 이루어진다: **DIVL**(value learning) + **QAM**(policy extraction).
이 문서는 **DIVL 만** 다룬다. QAM 은 flow-based VLA actor 를 업데이트하는 부분이라
`rl/offline_iql.py` 처럼 critic 만 학습하는 스크립트에는 들어가지 않는다 (§7 에서 경계만 짚는다).

결론 먼저:

- DIVL 은 **IQL 의 V 회귀를 통째로 갈아끼운 것**이다. Q 쪽은 손대지 않는다.
  IQL: `V ← expectile_τ(Q̄(s,a_data))` (스칼라 회귀)
  DIVL: `p_ψ(v|s) ← categorical CE against Q̄(s,a_data)` 후 `V := Quant_τ(p_ψ)`
- **우리 `offline_iql.py` 의 `--bins` 와는 방향이 정반대다.** 지금 코드는 *Q* 를 distributional
  (HL-Gauss) 로 만들고 *V* 를 스칼라로 둔다. DIVL 은 *V* 가 categorical 이고 *Q* 는 스칼라 MSE 다.
  이걸 헷갈리면 구현이 통째로 어긋난다. (§3.1)
- DIVL 은 **distributional RL 이 아니다.** return distribution 에 대한 distributional Bellman
  backup 이 없다. `p_ψ(v|s)` 는 "그 상태에서 데이터 액션들에 critic 이 매긴 스칼라 값들의 분포"다.
  C51 은 표현 방식(atom + projection)만 빌려온다. (§2.2)
- 논문의 주장(Proposition 1): 분포를 맞추고 τ-quantile 을 뽑는 것은 `ρ_{τ,1}` 직접 회귀와
  **같은 최적해**를 갖는다. 즉 DIVL = "IQL 의 expectile(p=2) 을 quantile(p=1) 로 바꾼 것"을
  2단계로 나눠 구현한 것이고, **나눈 덕분에 공짜로 얻는 게 두 개** 있다:
  (a) τ 를 바꿔도 재학습이 필요 없다 → **상태별 adaptive τ** 가 가능해진다
  (b) 분포의 **엔트로피**가 불확실성 신호로 쓰인다
  이 두 개가 DIVL 의 실질적 내용 전부다. (§2.4, §2.5)

---

## 1. 세팅과 표기 (논문 §III-A)

| 기호 | 의미 | 논문 값 |
|---|---|---|
| `M = (S, A, T, r, γ)` | MDP | γ ∈ (0,1] |
| `s = (o, ℓ_k)` | 관측 + 언어 지시. task `k ∈ K` | 멀티태스크 8개 |
| `r` | **sparse binary**. 에피소드가 성공 종료할 때만 `r=1`, 그 외 0 | — |
| `a_t ≡ a_{t:t+H}` | action chunk. 통째로 실행 후 replan | **H = 30** (30Hz → 1초) |
| `r_t ≡ Σ_{i=0}^{H-1} γ^i r_{t+i}` | chunk reward (Eq 2) | — |
| `(s_t, a_t, r_t, s_{t+H}) ~ D` | 리플레이 샘플 | offline: `B_off`, online: `B_off ∪ B_on` |
| `γ` | 프레임당 할인 | **0.9999** (→ γ^H = 0.9970) |

정책도 critic 도 전부 **action chunk 단위**로 동작한다. 하나의 generalist 정책을 8개 태스크가
공유한다.

> **우리 레포와의 표기 충돌 주의.** 논문의 `H` 는 "chunk horizon = replan 간격" 이다.
> 우리 레포에서 그 역할을 하는 건 `exp["replan_steps"]` (= `R`) 이고, 우리 `H =
> exp["action_horizon"]` 은 **다른 것**(VLA 가 뱉는 청크 전체 길이)이다.
> 논문의 `γ^H` ↔ 우리 `GEFF = cfg.discount ** R` (`rl/offline_iql.py:108`).
> 논문의 critic 액션 입력 `a_t` ↔ 우리 `act(i)` = `chunk[0 : LAT+R]` 평탄화 (`rl/data.py:450` 주석).

### 1.1 비교 기준: IQL (논문 §III-B, Eq 3–6)

```
ρ_{τ,2}(u) = |τ − 1(u<0)| · u²                                       (Eq 4)
L_V^IQL(ψ)  = E_D[ ρ_{τ,2}( Q_φ̄(s_t,a_t) − V_ψ^IQL(s_t) ) ]            (Eq 3)
y_t^IQL     = r_t + γ^H · V_ψ^IQL(s_{t+H})                            (Eq 5)
L_Q^IQL(φ)  = E_D[ ( Q_φ(s_t,a_t) − y_t^IQL )² ]                      (Eq 6)
```

`Q_φ̄` 는 EMA target. τ > 1/2 면 V 가 데이터 액션 Q 분포의 위쪽을 보므로 `max_a Q` 없이
in-support 정책 개선이 들어간다. **LWD 는 이 asymmetric bootstrap 원리를 그대로 유지하고,
스칼라 expectile 회귀만 distributional + quantile 로 바꾼다.**

---

## 2. DIVL 본체 (논문 §IV-A + Appendix A1, A2)

### 2.1 무엇을 모델링하는가 (Eq 11)

```
p_ψ(v | s_t) = P( v = Q_φ(s_t, a_t) | a_t ~ D(· | s_t) )              (Eq 11)
```

`D(·|s_t)` = `s_t` 에 조건부인 **경험적 리플레이 액션 분포**.
따라서 **`V_ψ(s_t)` 는 스칼라 값 추정이 아니다.** "상태 `s_t` 에서 리플레이 액션들에 critic 이
매긴 스칼라 값들의 분포" 다.

> **왜 상태당 액션이 하나뿐인데 분포가 나오나?** 나오지 않는다 — 개별 샘플에서는 안 나온다.
> 분포는 (a) 비슷한 상태들에 걸친 네트워크의 일반화, (b) 같은/유사 상태를 여러 번 방문한
> 이질적 리플레이(demo / rollout 성공 / rollout 실패 / play), 이 둘에서 나온다.
> IQL 의 expectile 회귀가 작동하는 이유와 정확히 같다 — IQL 도 상태당 샘플 하나로
> "데이터 액션 Q 분포의 상위 expectile" 을 맞춘다. DIVL 은 그 암묵적 분포를 **명시적으로**
> 들고 있을 뿐이다. 논문이 이 세팅에서 이게 중요하다고 주장하는 근거는 §5 참고
> (fleet 데이터의 return 이 multi-modal / heavy-tailed 하다).

### 2.2 학습 (Eq 12, Eq 20–21)

```
L_V(ψ) = E_{(s_t,a_t)~D} [ − log p_ψ( Q_φ̄(s_t, a_t) | s_t ) ]         (Eq 12)
```

**EMA target critic `Q_φ̄` 의 스칼라 출력**이 라벨이다. 온라인 `Q_φ` 가 아니다.

Appendix A1 의 구체적 파라미터화:

```
고정 categorical support  {V_i}_{i=1}^K,  [v_min, v_max]
  실기 실험:  K = 201 atoms,  [v_min, v_max] = [−0.1, 1.1]     → bin width 0.006

p_ψ(i|s) = softmax(V_ψ(s))_i                                          (Eq 20)

타깃:  Q_φ̄(s,a) 를 [v_min, v_max] 로 clip → 이웃한 두 atom 에 **선형 배분**
       (C51 projection, Bellemare et al. 2017) → 타깃 분포 m(s,a)

L_Z(ψ) = − E_{(s,a)~D} [ Σ_{i=1}^{K} m_i(s,a) · log p_ψ(i|s) ]        (Eq 21)
```

즉 **cross entropy**. Eq 12(NLL)와 Eq 21(CE)은 같은 것의 두 표기다 — 구현은 Eq 21 을 쓴다.

> support 를 `[0,1]` 이 아니라 `[−0.1, 1.1]` 로 잡은 것이 포인트다. 보상이 `{0,1}` 이고 γ<1 이면
> return ∈ [0,1] 이지만, 학습 중 `Q_φ̄` 가 잠깐 경계를 넘거나 정확히 경계에 놓일 때 C51 의
> 두-atom 배분이 퇴화한다. 양쪽 0.1 패딩이 그걸 막는다.

### 2.3 부트스트랩 통계 = τ-quantile (Eq 13, 22–23)

```
F_ψ(v | s)               = CDF induced by p_ψ(v|s)
Quant_τ(V_ψ(s))          ≜ inf { v : F_ψ(v|s) ≥ τ }                   (Eq 13)

이산 형태:
F_ψ(V_j | s)             = Σ_{i≤j} p_ψ(i|s)                           (Eq 22)
Quant_τ(V_ψ(s))          = V_{ min{ j : F_ψ(V_j|s) ≥ τ } }            (Eq 23)
```

= "누적확률이 τ 를 처음 넘는 atom 의 값". 이게 스칼라 부트스트랩 값이 된다.

### 2.4 Critic TD (Eq 14–15)

```
y_Q        = r_t + γ^H · Quant_τ( V_ψ(s_{t+H}) )                      (Eq 14)
L_Q(φ)     = E_D [ ( Q_φ(s_t,a_t) − y_Q )² ]                          (Eq 15)
```

**Q 는 스칼라 MSE 다.** distributional 아니다.

τ-quantile 은 "전체 액션 공간에 대한 명시적 max backup" 이 아니라 **리플레이 액션에 대한
in-distribution optimistic 통계**다. 데이터 밖으로 공격적으로 외삽하지 않으면서 고가치
리플레이 액션 쪽으로 타깃을 기울인다 — IQL 이 expectile 로 하던 일과 같은 목적.

### 2.5 Proposition 1 — 왜 이게 IQL 과 "같은 것"인가 (Eq 16, 25–31, Appendix A2)

일반화된 asymmetric loss:

```
ρ_{τ,p}(u) = |τ − 1(u<0)| · |u|^p                                     (Eq 16 = Eq 25)
   p = 2  →  expectile   (IQL 이 쓰는 것)
   p = 1  →  quantile    (DIVL 이 쓰는 것)
```

**Proposition 1.** 이 family 의 임의의 고정된 asymmetric loss 에 대해,
(A) 데이터 action-value 에 대한 직접 스칼라 회귀와
(B) "상태조건부 Q 값 분포를 먼저 적합 → 대응하는 asymmetric 통계를 추출" 하는 2단계 절차는
**같은 최적 스칼라 값**을 낸다.

증명 스케치 (Appendix A2):

```
직접:   J_direct(v) = E_{a~D(·|s)} [ ρ_{τ,p}( Q(s,a) − v ) ]           (Eq 26)
        d/dv J_direct = ∫ D(a|s) · d/dv ρ_{τ,p}(Q(s,a) − v) da = 0     (Eq 27)

DIVL:   무한히 미세한 이산화 + 충분한 capacity 극한에서 CE 의 최적해는
        v = Q(s,a) 사상에 의한 a~D(·|s) 의 pushforward 를 복원한다:
          p_ψ(v|s) = P(v = Q(s,a) | a ~ D(·|s))                       (Eq 28)
        따라서 임의의 적분가능 f 에 대해
          E_{v~p_ψ}[f(v)] = E_{a~D}[f(Q(s,a))]                        (Eq 29)
        2단계는  J_dist(v) = E_{u~p_ψ}[ ρ_{τ,p}(u − v) ]               (Eq 30)
        을 최소화하고, 그 1차 조건 (Eq 31) 은 변수변환 후 Eq 27 과 동일. ∎
```

**실무적 함의**: 최적해가 같다면 왜 2단계로 나누나? 분포를 들고 있으면
1. τ 를 바꿔도 **스칼라 V 를 재적합할 필요가 없다** → 상태별 adaptive τ 가 가능 (§2.6)
2. 분포의 **엔트로피**가 공짜로 나온다 → 그 adaptive 의 신호가 된다 (§2.6)
3. 진단 가능성: 값 분포 자체를 시각화할 수 있다 (Fig 6, Fig 9)

논문이 §IV-A 에서 명시하는 이유가 정확히 이 셋이다.

### 2.6 Adaptive τ (Eq 17–18, Eq 24)

분포의 **정규화 엔트로피**를 불확실성 신호로 쓴다:

```
H(s_{t+H}) = − (1 / log C) · Σ_{c=1}^{C} p_{ψ,c}(s_{t+H}) · log p_{ψ,c}(s_{t+H})   (Eq 17)
           ∈ [0, 1]                                                   (Eq 24, 여기선 C = K)

τ(s_{t+H}) = clip( τ_base − α · H(s_{t+H}),  τ_min,  τ_max )          (Eq 18)
```

- `τ_base`: 확신 있는 상태에 대한 목표 τ
- `α ≥ 0`: 불확실성 민감도
- **분포가 퍼져 있으면(H↑) τ↓ → 보수적** / **뾰족하면(H↓) τ↑ → 낙관적**
- **`τ(s_{t+H})` 는 TD 타깃 계산 시 stop-gradient** (논문 명시)
- Eq 17 의 `C` 와 Eq 24 의 `K` 는 같은 것(atom 개수). 논문 표기 불일치일 뿐이다.

측정된 동작 (Fig 8):

| 단계 | τ_base | α | 관측 H 범위 | 관측 τ 범위 | 학습 스텝 |
|---|---|---|---|---|---|
| Offline | 0.6 | 0.3 | 0.55 → 0.20 | 0.44 → 0.54 | ~40k |
| Online  | 0.9 | 0.3 | 0.48 → 0.32 | 0.76 → 0.80 | ~5k |

`0.6 − 0.3·[0.20, 0.55] = [0.435, 0.54]`, `0.9 − 0.3·[0.32, 0.48] = [0.756, 0.804]` —
Fig 8 과 정확히 일치한다. 즉 **논문 실험에서 clip 이 한 번도 걸리지 않았다.**
(τ_min/τ_max 실제 값은 "Appendix B2 에 보고" 라고 써 놓고 **정작 B2 에 없다.** §6 참고)

엔트로피는 offline→online 내내 감소한다 = value 가 점점 확신을 갖는다. 그에 따라 τ 가 올라가
정책이 더 높은 값의 해를 선호하게 된다.

> **주의 — 엔트로피는 "multi-modality" 를 재지 않는다.** 실측(§7.5-0)으로 확인한 것:
> atom 201개 위에서 0.2 와 0.9 에 질량이 반씩 몰린 **이봉 분포**의 정규화 엔트로피는 **0.131** 로,
> 균등 분포(1.0) 는 물론이고 웬만한 단봉 분포보다도 낮다. 즉 Eq 17 이 재는 것은
> **atom 축 위의 퍼짐(spread)** 이지 봉우리 개수가 아니다.
> 결과적으로 adaptive τ 는 "이봉이면 보수적으로" 가 아니라 **"값이 넓게 흩어져 있으면 보수적으로"**
> 로 동작한다. 논문 §5 의 multi-modality 서사(값 분포로 rare high-return mode 를 보존한다)는
> **quantile 추출**이 담당하고(이봉 분포에서 τ=0.5→0.2, τ=0.6→0.896 으로 위쪽 mode 를 집는다),
> **엔트로피 신호**는 그것과 별개의 축이다. 두 메커니즘을 하나로 묶어 이해하면 디버깅 때 헷갈린다.

### 2.7 n-step chunk-level TD (Eq 19) — **offline 전용**

long-horizon 태스크는 수천 스텝에 보상이 극도로 sparse 해서 1-step 타깃으로는 성공 신호가
너무 느리게 전파된다. 그래서 **offline 단계에서만** n-step 을 쓴다:

```
y_Q = Σ_{i=0}^{n-1} γ^{iH} · r_{t+iH}  +  γ^{nH} · Quant_{τ(s_{t+nH})}( V_ψ(s_{t+nH}) )   (Eq 19)
```

- `n = 1`  — 짧은 태스크 (grocery restocking)
- `n = 10` — long-horizon 태스크
- **n-step 창 안에서 에피소드가 종료하면: 종료 청크에서 return 을 잘라내고 부트스트랩 항을 제거한다.**
- **online 은 모든 태스크에서 n = 1.**
  논문이 밝힌 이유: 온라인 궤적은 policy transition 과 human intervention 이 섞여 있다.
  긴 backup 은 이 두 소스를 가로지를 확률이 높아, TD 경로가 **하나의 정책 실행에 대응하지 않게**
  된다. critic/value 는 이미 offline 초기화가 되어 있으므로 1-step 으로 충분하다.

주의: 이건 **chunk 단위** n-step 이다. 프레임 단위가 아니라 `H` 프레임씩 n 번 건너뛴다.
(우리 `rl/data.py:417 nstep()` 은 **청크 내부** 집계 = 논문 Eq 2 에 해당한다. Eq 19 는 그 위에
한 겹 더 필요하다. §3.4)

### 2.8 알고리즘 2 — 한 번의 LEARNER 업데이트

```
Require: mini-batch B_mini = {(s_t, a_t, r_t, s_{t+H})}
         critic Q_φ (target Q_φ̄), distributional value V_ψ,
         policy π_θ (reference π_β), EMA rate ρ

// Distributional Implicit Value Learning
1: ψ ← minimize Eq 12                      # 값 분포 먼저 갱신
2: y_Q ← Eq 19                             # **갱신된** ψ 로 타깃 계산
3: φ ← minimize Eq 15
4: φ̄ ← ρ·φ̄ + (1−ρ)·φ

// Policy Extraction via QAM  (이 문서 범위 밖)
5: a⁰_t ~ N(0, I)
6: π_β 로 reference trajectory {a^w_t} 롤아웃
7: endpoint a¹_t = a_t
8: θ ← minimize Eq 9,  g̃_1 = −∇_a[Q_φ(s,a¹_t)/λ]  (Eq 10)
9: return (Q_φ, V_ψ, π_θ, Q_φ̄)
```

**순서가 중요하다**: `ψ` 를 먼저 갱신하고, **그 갱신된 `ψ`** 로 TD 타깃을 만든다.
그리고 같은 미니배치 하나로 1~3 을 전부 처리한다.

---

## 3. 아키텍처 (논문 §IV-D + Appendix B2)

### 3.1 네트워크 구성

```
                     ┌─ Value head  → logits over K=201 atoms   (categorical)
 s_t ─ VLM backbone ─┤                                   ↑ readout token hidden state z_t
       (공유)         └─ Critic head ← concat(z_t, pool(a_t))
                                     → 스칼라 ×2  (clipped double-Q)
```

- **Value `V_ψ` 와 Critic `Q_φ` 가 backbone 을 공유**한다. 헤드만 다르다.
- backbone: Gemma 3-270M-IT + SigLIP-So400M (공개 체크포인트로 초기화).
  visual projection layer 와 value/critic head 는 **scratch 초기화**.
- **readout token** 의 최종 hidden state `z_t` 가 value/critic 공용 상태 표현.
- value head: 고정 categorical support 위의 logits. C51 projection 으로 만든 `m_t` 에 CE.
- critic head: `z_t` + action chunk. action chunk 는 **learned temporal attention pooling**
  으로 인코딩해서 `z_t` 와 concat.
- **critic head 2개, clipped double-Q. 최솟값을 DIVL 타깃 구성과 TD backup 에 쓴다.**
- **정책 네트워크와 value/critic 네트워크는 완전히 별개 모듈.** 로봇 fleet 에는 정책
  체크포인트만 배포되고, value/critic 은 중앙 learner 에만 남는다.
- actor 는 π0.5 flow VLA (PaliGemma: Gemma-2B + SigLIP, + Gemma-300M action expert).

**학습 대상 (단계별)**

| | actor | value/critic |
|---|---|---|
| Offline RL | 전체 fine-tune | 전체 fine-tune |
| Online | **VLM backbone freeze**, action expert 만 갱신 | 전체 fine-tune (mixed replay) |

온라인에서 정책 backbone 을 얼리는 이유: 업데이트 효율 + 사전학습된 vision-language 표현 보존.
value/critic 은 계속 전체 학습해서 변하는 리플레이 분포에 적응하고 최신 개선 신호를 준다.

### 3.2 하이퍼파라미터 (Appendix B2) — 전부

| 항목 | 값 |
|---|---|
| action chunk horizon `H` | 30 |
| policy optimizer | AdamW, base lr **2e-5**, cosine decay |
| value/critic optimizer | Adam, base lr **5e-4**, cosine decay |
| discount `γ` | **0.9999** (프레임당) |
| offline `τ_base` / `α` | **0.6** / **0.3** |
| online `τ_base` / `α` | **0.9** / **0.3** |
| `τ_min` / `τ_max` | **논문에 없음** (§6-1) |
| target critic·value EMA rate `ρ` | **0.005** |
| QAM temperature `λ` | 2 |
| offline TD | long-horizon **10-step**, grocery **1-step** (chunk 단위) |
| online TD | 전 태스크 **1-step** |
| online 배치 offline:online 비율 | **≈ 1:1** |
| categorical atoms `K` | 201 |
| value support | [−0.1, 1.1] |
| batch size | **논문에 없음** |
| offline / online 스텝 수 | 명시 없음. Fig 8 x축 기준 ~40k / ~5k |

### 3.3 초기화 순서 (Appendix B3, Algorithm 1)

```
1. π0.5 사전학습 VLA 를 demonstration 데이터로 BC → imitation checkpoint (= SFT reference)
2. π_β ← 이 checkpoint 로 **고정** (reference flow. offline·online 내내 안 바뀐다)
3. LWD (Offline): π_θ 를 이 checkpoint 로 초기화, Q_φ / V_ψ 는 scratch, Q_φ̄ ← Q_φ
   → Adjoint Matching 으로 정책, DIVL 로 critic+value 학습
4. LWD (Online): 3의 체크포인트에서 정책·value 모듈 **전부** 이어받아 mixed replay 로 계속
```

### 3.4 데이터 (Appendix B1, Table IV)

offline buffer 는 세 소스:

| 소스 | 성격 | 시간 | 비중 |
|---|---|---|---|
| Demonstration | 사람 전문가, **항상 성공** | 336.6 h | 51.6% |
| Rollout (success) | 과거 정책 롤아웃 | 88.8 h | 13.6% |
| Rollout (failure) | 과거 정책 롤아웃 | 39.2 h | 6.0% |
| Play | 사람이 실패 모드·엣지케이스를 유도 탐색, **항상 실패로 취급** | 187.9 h | 28.8% |
| **합계** | | **652.5 h** | 성공 65.2% / 실패 34.8% |

- 세 소스 전부 **online replay 와 동일한 chunked transition 포맷**으로 변환된다.
- 보상은 **terminal success/failure 라벨**로 sparse binary 할당.
- 태스크별 분포: grocery 18.8% / long-horizon 81.2% (long-horizon 에피소드가 길어서).
- **버퍼의 약 1/3 이 실패 데이터다.** BC 계열 baseline 은 이걸 못 쓰고, LWD 는 쓴다 —
  이게 논문이 강조하는 데이터 활용도 차이.
- online: **human intervention 구간도 일반 online transition 으로 저장**된다
  (실행된 corrective action 을 액션으로). 보상은 동일하게 terminal 라벨 기준.

---

## 4. 실험 결과 중 DIVL 관련 부분

**Table II / Table V — DIVL vs 스칼라 expectile 회귀** (나머지 전부 동일)

| | Short-Horizon (offline) | (online) | Long-Horizon (offline) | (online) |
|---|---|---|---|---|
| Expectile Regression | 0.96 | 0.97 | 0.72 | 0.78 |
| **DIVL** | 0.97 (+1.0%) | 0.99 (+2.1%) | **0.79 (+9.7%)** | **0.91 (+16.7%)** |

전체 8태스크 평균: offline 0.84 → 0.88, online 0.88 → 0.95.
**이득이 long-horizon 에 몰려 있다.** grocery 는 이미 포화(0.96+)라 여지가 없다.

**Table III — adaptive τ vs constant τ** (offline 만)

constant baseline 은 adaptive 런에서 관측된 τ 의 경험적 평균 **τ = 0.52** 로 고정.

| | Restocking | Correction | Freezer | Open-Cooler | Gongfu Tea | Fruit Juice | Cocktail | Shoebox | Avg |
|---|---|---|---|---|---|---|---|---|---|
| constant τ=0.52 | 0.85 | 0.88 | **0.94** | 0.95 | 0.70 | **0.76** | 0.70 | **0.90** | 0.84 |
| adaptive τ | **1.00** | **1.00** | 0.92 | 0.95 | **0.72** | 0.74 | **0.83** | 0.86 | **0.88** |

개별 태스크에서는 constant 가 이기는 경우도 있지만(Freezer, Fruit Juice, Shoebox)
평균은 adaptive 가 0.84 → 0.88.

**Fig 6 / Fig 9 — 값 분포 진단 (재현할 가치 있음)**
- 성공 에피소드: 예측 분포가 **unimodal 유지**, mode 가 0.4 → 1.0 으로 꾸준히 상승
- 실패 에피소드: mode 가 0.5 → 0.6 에서 정체
- sparse terminal reward 만으로도 학습된 값이 **task progress 를 추적**한다는 증거

부수 효과: critic-guided 업데이트로 **cycle time 이 SFT 대비 평균 23.75초 감소**.
값 함수가 확실히 진척을 내는 청크를 선호해서 망설임·재시도·불안정한 중간 동작이 줄었다.

---

## 5. 논문이 밝힌 "왜 distributional 인가"

fleet 배포 세팅의 특수성이다:

- 로봇들이 **비동기로, 다양한 조건에서** 데이터를 모은다
- 그래서 **같은 state-action pair 에 붙는 return 이 multi-modal 이고 heavy-tailed** 하다
- 스칼라 critic 은 이걸 평균으로 뭉개서 **드물지만 재현 가능한 성공을 가린다**
- distributional critic 은 그 high-return mode 를 보존한다
- 근거로 Kumar et al. 2022 (`[55]`, offline Q-learning on diverse multi-task data) 인용 —
  categorical 표현이 다양한 멀티태스크 offline RL 에서 도움된다는 선행 결과

### 5.1 "PA-RL 의 distributional Q 와 같은 것 아닌가?" — 아니다, 그런데 반쯤 맞다

혼동하기 쉬운 지점이라 따로 정리한다. 둘 다 "IQL + categorical head" 처럼 보이지만
**분포에서 무엇을 읽는가(readout statistic)** 가 다르다.

| | PA-RL distributional Q | DIVL distributional V |
|---|---|---|
| 분포를 붙인 대상 | `Q(s,a)` | `V(s)` |
| 분포가 표현하는 것 | **스칼라 타깃 하나를 번진 것** (실제 randomness 없음) | 상태 `s` 의 **리플레이 액션들에 걸친 Q̄ 값의 퍼짐** (실제 randomness) |
| 타깃 투영 | HL-Gauss (가우시안 smear) | C51 (이웃 두 atom 선형 배분) |
| **읽는 통계** | **평균** `Σ p·centers` | **τ-quantile + 엔트로피** |
| 스칼라 헤드로 바꾸면 | 값은 거의 같다 (최적화/경계 이득만 손실) | **메커니즘이 통째로 사라진다** |

PA-RL 코드로 확인한 것:
- `jaxrl_m/networks/distributional.py:99` `transform_from_probs = Σ(probs · centers)` — **평균**이다.
- `jaxrl_m/agents/continuous/iql.py:259-266` — logits 는 **CE 손실에만** 쓰이고, 반환되는 `q` 는
  평균 스칼라다.
- `iql.py:281-291` `value_loss_fn` — `q = jnp.min(forward_target_critic(...), axis=0)` 즉
  **평균-readout Q 두 개의 min**. V 는 그 스칼라에 expectile 회귀. 분포 모양은 V 에 전혀 안 들어온다.

→ PA-RL 의 distributional Q 는 **"classification as regression"** (Farebrother et al. 2024,
  *Stop Regressing*) 이다. 분포는 정보로서는 **잉여**고, 이득은 두 가지 구조적인 것뿐이다:
  (a) CE 가 L2 보다 조건수가 좋다 (b) support 가 유계라 발산이 구조적으로 불가능
  — 우리가 `rl/offline_iql.py:198` 주석에 적어둔 "음수 고정점 -0.013 이 표현 불가" 가 (b) 다.

→ DIVL 의 distributional V 는 **평균을 안 읽는다.** τ-quantile 과 엔트로피, 둘 다 분포의 모양에
  의존한다. 여기서는 분포가 **load-bearing** 이다.

**그런데 여기서 한 겹 더 들어가야 한다.** 논문 스스로 Proposition 1 (§2.5) 에서
*"고정 τ 라면 분포 적합 후 통계 추출 = 직접 asymmetric 회귀, 같은 최적해"* 라고 인정한다.
즉 **τ 가 고정이면 DIVL 은 IQL 의 expectile(p=2) 을 quantile(p=1) 로 바꾼 것에 불과하다.**

그리고 논문의 ablation 숫자가 정확히 그걸 보여준다:

| offline 8태스크 평균 | 출처 |
|---|---|
| Expectile regression (= IQL) | **0.84** (Table V) |
| DIVL, **constant** τ = 0.52 | **0.84** (Table III) |
| DIVL, **adaptive** τ | **0.88** (Table III = Table V) |

(Table III 의 adaptive 행과 Table V 의 DIVL offline 행은 태스크별 숫자까지 완전히 동일하다 —
같은 런이다. constant-τ 행과 expectile 행은 태스크별로는 다르지만 평균이 둘 다 0.84.)

**→ offline 에서 DIVL 이 IQL 을 이긴 이득(0.84 → 0.88)은 전부 adaptive τ 에서 나온다.**
"distributional 이라서" 가 아니다. 분포는 adaptive τ 를 **가능하게 하는 수단**이지 그 자체가
기여가 아니다.

정리하면 DIVL 의 이득은 세 갈래고, 그중 둘은 PA-RL 의 distributional Q 와 **공유**된다:

| | 이득 | PA-RL distributional Q 도 갖는가 |
|---|---|---|
| (a) | CE 가 L2 보다 최적화가 잘 된다 | ✅ 공유 |
| (b) | 유계 support → 발산 구조적 불가 | ✅ 공유 |
| (c) | **quantile readout + 엔트로피 → 상태별 adaptive τ** | ❌ **DIVL 고유** |

**우리에게 주는 결론 세 개:**

1. **둘은 직교하므로 합칠 수 있다.** 논문이 Q 를 스칼라로 둔 건 선택이지 제약이 아니다.
   `--bins 128`(categorical Q, 평균 readout, 발산 방지) + `--atoms 201`(categorical V, quantile
   readout, adaptive τ) 조합이 정당하고, 우리가 실측한 Q 음수 고정점 문제를 생각하면
   **아마 우리가 실제로 원하는 건 이 조합**이다.
2. **재현 우선순위는 adaptive τ 다.** categorical V 만 넣고 τ 를 고정하면 논문 기준
   IQL 대비 이득이 0 이다. `--alpha` 를 0 이 아니게 두는 게 핵심이지 `--atoms` 가 핵심이 아니다.
3. **더 싼 대안이 있다.** adaptive τ 에 필요한 건 "상태별 불확실성 신호" 하나뿐이다.
   우리는 이미 Q 앙상블이 10개 있으므로 **V 앙상블 분산**으로도 같은 스케줄을 만들 수 있다
   (categorical 헤드 없이). 다만 정규화 엔트로피는 `[0,1]` 로 유계라 `τ_base − α·H` 의
   하이퍼파라미터가 스케일 독립적인 반면, 앙상블 std 는 Q 스케일에 따라 α 를 다시 튜닝해야 한다.
   → **논문 재현은 categorical 로, 빠른 검증은 앙상블로** 가 합리적이다.

**남는 의문 (논문에 ablation 없음):** online 의 `τ_base = 0.9` 는 offline 의 0.6 보다 훨씬
공격적이다. Table II 의 online 이득(0.88 → 0.95)이 adaptive 때문인지 단순히 **τ 가 높아서**인지
구분할 데이터가 논문에 없다. 우리 실험에서는 `--tau-base` 를 따로 스윕해서 분리해야 한다.

---

> **우리 세팅에 대한 판단.** 이 논리는 *멀티태스크 + 이질적 소스 + fleet 비동기* 를 전제로 한다.
> 단일 태스크·단일 소스 데이터셋이면 논문이 주장하는 이득의 근거가 약해진다.
> 다만 (a) adaptive τ, (b) 값 분포 진단, (c) categorical 헤드의 구조적 발산 방지 —
> 이 셋은 태스크 수와 무관하게 유효하다. 특히 (c) 는 우리가 이미 `--bins` 로 Q 에 적용한
> 것과 같은 동기다 (`rl/offline_iql.py:198` 주석의 "음수 고정점 -0.013 이 표현 불가").

---

## 6. 논문이 안 알려주는 것 / 우리가 정해야 하는 것

구현 전에 결정이 필요한 지점을 전부 나열한다.

1. **`τ_min` / `τ_max` 값이 없다.** Eq 18 에 clip 이 있고 "값은 Appendix B2 에" 라고 써놓고
   B2 에 안 적혀 있다. Fig 8 로 역산하면 두 단계 모두 **clip 이 걸리지 않았다**.
   → 느슨하게 잡으면 된다. 권장: `τ_min=0.5, τ_max=0.95`. (τ<0.5 면 IQL 의 in-support 개선
   방향이 뒤집히므로 0.5 를 하한으로 두는 게 안전하다.)

2. **`V_ψ` 에 target network 가 있는가 — 본문끼리 모순.**
   - Appendix B2: "Target critic **and value** networks are updated with EMA rate 0.005"
   - Algorithm 2 `Require`: target 은 `Q_φ̄` 만 나열. line 4 도 `φ̄` 만 갱신.
   - Eq 14/19: `Quant_τ(V_ψ(·))` — **온라인** ψ 다 (ψ̄ 아님)
   → **수식/Algorithm 2 를 따른다: `V` 에는 target 을 두지 않는다.** IQL 도 그렇다.
     단 논문은 backbone 을 공유하므로 그 EMA 사본이 자연히 value 쪽도 덮는다 — B2 문장은
     그걸 가리킨 것으로 읽는 게 자연스럽다. 우리는 backbone(=encoder)을 공유하니
     `tenc` 하나만 EMA 로 두면 논문과 동치가 된다.

3. **`Q_φ̄` 를 어디에 쓰는가.** Eq 12(값 분포 라벨)에는 명시적으로 `Q_φ̄`. Eq 15 의 타깃 `y_Q` 는
   `V_ψ` 만 거치므로 `Q_φ̄` 가 안 들어간다. 그런데 §IV-D 는 "minimum critic estimate 를
   **DIVL target construction 과 TD backups 양쪽에**" 쓴다고 한다.
   → 일관되게 읽는 방법: **스칼라 `Q̄` 값이 필요한 모든 자리에서 두 target head 의 min 을 쓴다.**
     실질적으로 그 자리는 Eq 12 하나뿐이다. (우리 `rl/offline_iql.py:254-256` 이 이미 그렇다.)

4. **batch size 미보고.** 우리 `cfg.batch_size` (기본 64) 유지.

5. **Q head 개수 = 2 (clipped double-Q).** 우리 기본 `num_qs=10` 과 다르다. 논문 재현이
   목적이면 2, 우리 기존 세팅과 비교가 목적이면 10 유지 — 둘 다 돌려볼 것.

6. **C51 projection vs HL-Gauss.** 논문은 C51 (이웃 두 atom 선형 배분). 우리 기존 코드는
   HL-Gauss (`rl/offline_iql.py:211-216`, 가우시안으로 번진 soft label, `sigma=0.75·binwidth`).
   HL-Gauss 는 C51 의 label-smoothing 일반화다 — `sigma→0` 이면 C51 에 수렴.
   → **논문 재현은 C51, 옵션으로 HL-Gauss 를 남긴다.** 201 atoms 면 bin width 0.006 이라
     C51 두-atom 배분이 지나치게 뾰족할 수 있어 HL-Gauss 쪽이 나을 가능성도 있다.

7. **`p_ψ` 의 엔트로피는 `s_{t+H}` 에서 계산한다** (Eq 17 인자가 `s_{t+H}`). 즉 **다음 상태**의
   불확실성으로 그 상태의 부트스트랩 낙관도를 정한다. `s_t` 가 아니다 — 헷갈리기 쉽다.

8. **n-step 은 offline 만.** online 전환 시 반드시 1-step 으로 내려야 한다.

9. **γ = 0.9999 는 프레임당**이다. 우리 fuji 기본 0.999 보다 훨씬 1 에 가깝다
   (지평 10,000 프레임 vs 1,000 프레임). 논문 태스크가 3–5분 × 30Hz = 5,400–9,000 프레임인
   걸 생각하면 "지평 ≈ 에피소드 길이" 로 맞춘 것이다. `PA-RL.md` §4 의 크레딧 경로 길이
   논의와 직접 연결된다.

---

## 7. 구현 설계 — `rl/offline_divl.py`

`rl/offline_iql.py` 를 베이스로 한다. **데이터 파이프라인·인코더·평가·저장은 그대로 재사용**하고,
바뀌는 건 §7.2 의 다섯 곳뿐이다.

### 7.1 먼저: 무엇을 바꾸지 *않는가*

`offline_iql.py` 의 다음은 손대지 않는다.

- 1~188행: 인자 파싱, 모달리티/세션 로딩, `images.mm` VRAM 상주, 상태/액션 정규화,
  holdout 분할, `obs()/act()/st()` 헬퍼 — 전부 동일
- `BatchEncoder` + `CriticEnsemble` 구조 (`rl/nets.py:136`, `:180`)
- `tenc` (target encoder) 를 쓰는 설계 (`:226`) — §6-2 에 따라 이게 논문의 "target value network"
  문장을 커버한다
- head 초기화 `×1e-2` (`:233-236`) — V 가 0 근처에서 시작하게. **categorical 로 바꿔도 유지**
  (logits 이 0 근처 = 균등분포 시작 = 엔트로피 1.0 에서 시작 → 초기 τ 가 가장 보수적.
  이게 우리가 원하는 초기 동작이다)
- polyak (`:268-271`), 평가/AUC/플롯/저장 (`:277-309`)

**QAM 은 이 스크립트에 넣지 않는다.** `offline_iql.py` 는 actor 를 학습하지 않고
(`PA-RL.md` 의 `train_actor=False` 와 같은 구성) 정책은 롤아웃 시 후보 argmax 로 만든다.
DIVL 은 그 critic 을 그대로 대체하는 drop-in 이다. QAM 도입은 flow actor 가 준비된 뒤
별도 작업 (`rl/expo.py` 의 residual actor 를 QAM 으로 바꾸는 게 자연스러운 경로).

### 7.2 바뀌는 다섯 곳

#### (1) V 를 categorical 헤드로 — `offline_iql.py:228-236` 교체

지금:
```python
value = CriticEnsemble(cfg.latent_dim_image, snorm.shape[1], 0, 1, ...)   # 스칼라 1개
```

바꿀 것: 액션폭 0, 앙상블 1 은 그대로 두고 **헤드만 `K` 출력으로 교체**한다.
`--bins` 가 Q 헤드에 하던 것(`:205-206`)과 같은 트릭.

```python
lo_v, hi_v = (float(x) for x in a.v_range.split(","))          # 기본 "-0.1,1.1"
V_ATOMS = torch.linspace(lo_v, hi_v, a.atoms, device=dev)      # 기본 201
DZ = (hi_v - lo_v) / (a.atoms - 1)

value = CriticEnsemble(cfg.latent_dim_image, snorm.shape[1], 0, 1, cfg.latent_dim_state,
                       cfg.include_state, cfg.hidden_dims, cfg.critic_layer_norm).to(dev)
value.qs[0].head = xavier_(nn.Linear(value.qs[0].body.out_dim, a.atoms)).to(dev)
```

head `×1e-2` 초기화는 `critic.qs` 와 `value.qs` 둘 다에 그대로 적용 (`:233-236` 유지).
logits≈0 → 균등분포 → `H≈1.0` → `τ = τ_base − α` 로 시작. offline 기본이면 0.6−0.3 = 0.3
인데 이건 `τ_min=0.5` 에 걸린다. **초기 몇 백 스텝은 clip 이 실제로 작동하는 구간**이므로
`τ_min` 을 대충 잡으면 안 된다 (§6-1 에서 0.5 를 권한 이유).

#### (2) V 손실: expectile → C51 cross-entropy — `offline_iql.py:262-263` 교체

지금:
```python
d = qt - v
loss_v = (torch.where(d > 0, a.expectile, 1 - a.expectile) * d ** 2).mean()
```

바꿀 것 (Eq 21):
```python
def c51_project(y, atoms, dz):
    """스칼라 타깃 y (B,) → 이웃 두 atom 에 선형 배분한 (B, K) 분포. Bellemare 2017."""
    K = atoms.numel()
    y = y.clamp(atoms[0], atoms[-1])
    b = (y - atoms[0]) / dz                       # [0, K-1] 안의 실수 위치
    l, u = b.floor(), b.ceil()
    m = torch.zeros(y.shape[0], K, device=y.device, dtype=y.dtype)
    eq = (l == u).float()                         # 정확히 atom 위에 놓인 경우 질량 유실 방지
    m.scatter_add_(1, l.long().unsqueeze(1), (eq + (1 - eq) * (u - b)).unsqueeze(1))
    m.scatter_add_(1, u.long().unsqueeze(1), ((1 - eq) * (b - l)).unsqueeze(1))
    return m

v_logits = value(lat, st(i), none_act)[0]                      # (B, K)
m_t = c51_project(qt, V_ATOMS, DZ)                             # qt = min_2 Q̄(s,a_data), no_grad
loss_v = -(m_t * v_logits.log_softmax(-1)).sum(-1).mean()
```

`qt` 계산은 `offline_iql.py:254-256` 을 **그대로** 쓴다 — target critic + target encoder 의
min. §6-3 대로 이게 논문의 "minimum critic estimate for DIVL target construction" 이다.

HL-Gauss 옵션을 남기려면 기존 `q_loss` 의 erf 블록(`:211-216`)을 `V_ATOMS` 로 재사용하면 된다.

#### (3) Q 타깃: 스칼라 V → τ-quantile — `offline_iql.py:249-251` 교체

지금:
```python
nv = value(enc(obs(j), stop_gradient=True), st(j), none_act)[0]
tq = reward + (cfg.discount ** R) * mask * nv
```

바꿀 것 (Eq 13/17/18/19). **전부 `torch.no_grad()` 안에서** — τ 도 quantile 도 stop-grad:
```python
LOGK = math.log(a.atoms)

def value_stats(idx_next):
    """다음 상태의 (τ-quantile, normalized entropy). Eq 17/18/22/23."""
    lg = value(enc(obs(idx_next), stop_gradient=True), st(idx_next), none_act)[0]  # (B,K)
    logp = lg.log_softmax(-1)
    p = logp.exp()
    ent = -(p * logp).sum(-1) / LOGK                                    # Eq 17/24, ∈[0,1]
    tau = (a.tau_base - a.alpha * ent).clamp(a.tau_min, a.tau_max)      # Eq 18
    cdf = p.cumsum(-1)
    # Eq 23: 누적확률이 τ 를 처음 넘는 atom
    j = torch.searchsorted(cdf.contiguous(), tau.unsqueeze(-1).contiguous())
    j = j.clamp(max=a.atoms - 1).squeeze(-1)
    return V_ATOMS[j], ent, tau
```

주의점 세 개:
- **엔트로피는 `s_{t+nH}` (다음 상태) 에서** 계산한다 (§6-7).
- `searchsorted` 는 `cdf[j] >= tau` 인 첫 `j` 를 준다 = Eq 23 그대로.
- quantile 은 atom 값이라 애초에 미분 불가 — 그래도 `no_grad` 로 감싸서 의도를 명시한다.

부산물 `ent`/`tau` 는 **반드시 로깅**한다. Fig 8 재현이 곧 "DIVL 이 의도대로 도는가" 의
1차 확인이다. 엔트로피가 안 떨어지면 값 분포가 학습되지 않고 있다는 뜻.

#### (4) chunk-level n-step (Eq 19) — `rl/data.py` 에 함수 추가

우리 `nstep()` (`rl/data.py:417`) 은 **청크 내부** 집계 = 논문 Eq 2 다. Eq 19 는 그 위에
`R` 프레임씩 `n` 번 건너뛰는 겹이 하나 더 필요하다.

```python
def chunk_nstep(flat, idx, replan_steps=8, discount=0.99, n=1):
    """Eq.19 — 청크 단위 n-step. 청크 내부 집계는 nstep() 을 n 번 부른다.

    반환:
      ret      Σ_{k<n} γ^{kR} · (청크 k 의 내부 할인 보상)
      boot     부트스트랩 계수 ∈ {0,1}. n-step 창 안에서 종료했으면 0
               (= "종료 청크에서 return 을 자르고 부트스트랩 항 제거")
      next_idx 부트스트랩할 상태 인덱스 (t + n·R)
      valid    첫 청크의 valid (기존 nstep 과 동일 의미)
    """
    idx = np.asarray(idx, dtype=np.int64)
    gR = discount ** replan_steps
    ret = np.zeros(len(idx), np.float32)
    alive = np.ones(len(idx), np.float32)
    cur, valid = idx.copy(), None
    for k in range(n):
        c = nstep(flat, np.minimum(cur, len(flat) - replan_steps - 1), replan_steps, discount)
        if k == 0:
            valid = c["valid"]
        ret += (gR ** k) * c["reward"] * alive
        alive = alive * c["mask"]          # 이 청크에서 종료 → 이후 청크 기여 0, 부트스트랩도 0
        cur = c["next_idx"]
    return {"ret": ret, "boot": alive, "valid": valid,
            "next_idx": np.minimum(cur, len(flat) - 1)}
```

`n=1` 이면 기존 `nstep()` 과 정확히 동일한 결과를 낸다 (`ret=reward`, `boot=mask`) —
**회귀 테스트로 이걸 먼저 확인할 것.**

학습 루프에서:
```python
cn = chunk_nstep(flat, i, R, cfg.discount, a.nstep)
j = cn["next_idx"]
with torch.no_grad():
    qv, ent, tau = value_stats(j)
    tq = (torch.from_numpy(cn["ret"]).to(dev)
          + (GEFF ** a.nstep) * torch.from_numpy(cn["boot"]).to(dev) * qv)   # Eq 19
```

**학습 인덱스 경계도 넓혀야 한다** (`offline_iql.py:159`):
```python
train = np.flatnonzero(~hold[:len(flat) - a.nstep * R])     # 기존: len(flat) - R
```

에피소드 경계를 넘어가는 건 `alive` 가 0 이 되어 처리된다 (`flat` 은 에피소드를 이어붙인
평평한 배열이고 `flat.mask = 1 - done` 이므로). 다만 `nstep()` 내부에서 `idx + i` 가 배열
끝을 넘지 않도록 위 `np.minimum` 클램프가 필요하다.

#### (5) 인자 + 태그

```python
p.add_argument("--atoms", type=int, default=201,      help="V 의 categorical atom 수 (논문 201)")
p.add_argument("--v-range", default="-0.1,1.1",       help="V support (논문 [-0.1, 1.1])")
p.add_argument("--tau-base", type=float, default=0.6, help="논문: offline 0.6 / online 0.9")
p.add_argument("--alpha", type=float, default=0.3,    help="엔트로피 민감도 (논문 0.3)")
p.add_argument("--tau-min", type=float, default=0.5,  help="논문 미보고 — §6-1")
p.add_argument("--tau-max", type=float, default=0.95, help="논문 미보고 — §6-1")
p.add_argument("--nstep", type=int, default=1,        help="청크 단위 n-step. 논문: long-horizon 10")
p.add_argument("--v-proj", default="c51", choices=("c51", "hlgauss"),
               help="V 타깃 투영. 논문은 c51")
```

`--expectile` 은 제거한다 (τ 는 이제 adaptive 다). 대신 **`--alpha 0` 으로 두면 constant τ**
= Table III 의 ablation baseline 이 그대로 재현된다.

태그(`:112-115`):
```python
TAG = a.tag or (f"divl-a{a.atoms}-tb{str(a.tau_base).replace('.','')}"
                f"-al{str(a.alpha).replace('.','')}-n{a.nstep}"
                f"-g{f'{cfg.discount:g}'.replace('.','')}-q{cfg.num_qs}{a.v_min}-s{a.seed}")
```

### 7.3 최종 학습 루프

```python
for step in range(1, a.steps + 1):
    i  = train[rng.integers(0, len(train), cfg.batch_size)]
    cn = chunk_nstep(flat, i, R, cfg.discount, a.nstep)
    j  = cn["next_idx"]
    lat = enc(obs(i), stop_gradient=cfg.freeze_critic_encoder)

    # --- Algorithm 2 line 1: ψ 먼저 --------------------------------------
    with torch.no_grad():                                    # Eq 12 라벨: min_2 Q̄(s, a_data)
        mem = None if a.v_min == "all" else critic.subsample(cfg.num_min_qs, gen)
        qt = q_of(target(tenc(obs(i), stop_gradient=True), st(i), act(i),
                         members=mem)).min(dim=0).values
    v_logits = value(lat, st(i), none_act)[0]
    loss_v = -(c51_project(qt, V_ATOMS, DZ) * v_logits.log_softmax(-1)).sum(-1).mean()

    opt_v.zero_grad(set_to_none=True); loss_v.backward(); opt_v.step()

    # --- Algorithm 2 line 2: **갱신된** ψ 로 타깃 -------------------------
    with torch.no_grad():
        qv, ent, tau = value_stats(j)                        # Eq 17/18/23
        tq = (torch.from_numpy(cn["ret"]).to(dev)
              + (GEFF ** a.nstep) * torch.from_numpy(cn["boot"]).to(dev) * qv)   # Eq 19

    # --- Algorithm 2 line 3: φ ------------------------------------------
    valid = torch.from_numpy(cn["valid"]).to(dev)
    ql = critic(lat, st(i), act(i))
    loss_q = q_loss(ql, tq, valid)                           # Eq 15 (기본은 스칼라 MSE)
    opt_q.zero_grad(set_to_none=True); loss_q.backward(); opt_q.step()

    # --- Algorithm 2 line 4 ----------------------------------------------
    with torch.no_grad():
        for tp, pp in zip(list(target.parameters()) + list(tenc.parameters()),
                          list(critic.parameters()) + list(enc.parameters())):
            tp.mul_(1 - cfg.tau).add_(pp, alpha=cfg.tau)

    if step % 100 == 0:
        print(f"  step {step:5d}  q {float(q_of(ql).mean()):+.4f}  "
              f"v_q{a.tau_base} {float(qv.mean()):+.4f}  loss_q {float(loss_q):.5f}  "
              f"loss_v {float(loss_v):.5f}  H {float(ent.mean()):.3f}  tau {float(tau.mean()):.3f}")
```

**옵티마이저를 두 개로 쪼갠 이유**: Algorithm 2 는 `ψ` 를 먼저 갱신하고 그 결과로 타깃을
만든다. 지금처럼 `loss_q + loss_v` 를 합쳐 한 번에 backward 하면 (`offline_iql.py:264-267`)
그 순서가 성립하지 않는다. 인코더가 공유돼 있으므로 `opt_v` / `opt_q` 중 **한쪽에만** 인코더
파라미터를 넣어야 이중 갱신이 안 난다:

```python
opt_v = torch.optim.Adam(list(value.parameters()) + list(enc.parameters()), lr=cfg.critic_lr)
opt_q = torch.optim.Adam(list(critic.parameters()), lr=cfg.critic_lr)
```

인코더를 `opt_v` 에 붙이는 게 논문 구조(backbone 이 value 와 공유되고 value 가 먼저 갱신됨)에
가깝다. 다만 `lat` 을 두 손실이 재사용하므로 `loss_v.backward()` 후 그래프가 해제된다 —
`loss_q` 용으로 `lat` 을 다시 계산하거나 `retain_graph=True` 가 필요하다. **인코더 재계산이
싸지 않으므로**(ResNet), 실무적으로는 이 중 하나를 택한다:

- **A (논문 충실)**: `loss_v.backward(retain_graph=True)` → `opt_v.step()` → `lat` 재사용해
  `loss_q.backward()`. `opt_q` 는 critic 만 잡고 있으므로 인코더가 두 번 갱신되지 않는다.
  단 `lat` 은 `opt_v.step()` **이전** 인코더로 계산된 값이라 미세하게 stale.
- **B (단순, IQL 과 동일)**: 합쳐서 한 번 backward. 대신 `qv` 를 **직전 스텝의 ψ** 로 계산한
  꼴이 된다. 실질 차이는 1스텝 지연뿐이라 EMA 없는 V 에서는 무시할 만하다.

**B 로 시작하고, 수렴이 이상하면 A 를 시도**하는 걸 권한다. `--update-order {joint,seq}`
플래그로 둘 다 남겨두면 비교가 쉽다.

### 7.4 평가에 추가할 것

기존 AUC/Q-플롯(`:277-298`)은 그대로 두고 **값 분포 진단**을 추가한다 (논문 Fig 6/9 재현):

```python
# 홀드아웃 에피소드마다 시간축 × quantile 밴드
for q_level in (0.1, 0.25, 0.5, 0.75, 0.9):
    band[q_level] = quantile_of(value_logits_along_episode, q_level)
```
- 성공 에피소드에서 mode 가 단조 상승하는가 (논문: 0.4 → 1.0)
- 실패 에피소드에서 정체하는가 (논문: 0.5 → 0.6)
- 분포가 unimodal 을 유지하는가

이게 sparse terminal reward 로 학습한 값이 task progress 를 추적하는지에 대한 **직접적인**
확인이고, AUC(마지막 프레임 Q 하나)보다 훨씬 정보량이 많다.

추가 스칼라 로깅: `mean(H)`, `mean(τ)`, `mean(Quant_τ)`, atom 분포의 argmax 위치.

### 7.5 검증 순서

0. **(완료)** `c51_project` / quantile 추출 스니펫은 수치 검증했다 —
   질량 합 = 1 (경계·clip·atom 정확히 일치 케이스 포함), 복원 평균 오차 < 3e-8,
   atom 위에 정확히 놓인 타깃도 질량 유실 없음. quantile 은 이봉 분포에서
   τ=0.5 → 0.2 / τ=0.6 → 0.896 으로 기대대로 위쪽 mode 를 집는다.
1. `--nstep 1` 로 `chunk_nstep` == 기존 `nstep` 확인 (수치 동일)
2. `--alpha 0 --tau-base 0.7` 로 돌려서 기존 `offline_iql.py --expectile 0.7` 과
   **비슷한 AUC** 가 나오는지. Proposition 1 상 expectile(p=2) 과 quantile(p=1) 이라 완전히
   같진 않지만 크게 벗어나면 구현 버그다.
3. `--alpha 0.3 --tau-base 0.6` (논문 offline) → Fig 8 처럼 **엔트로피 하강 / τ 상승**이 보이는지
4. `--nstep 10` 으로 sparse reward 전파 속도가 빨라지는지 (우리 에피소드가 길면 필수)
5. 값 분포 플롯으로 Fig 6/9 패턴 확인

### 7.6 예상 실행 커맨드

```bash
source configs/paths.sh

# 논문 offline 설정 (short-horizon 상당)
PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" NO_ALBUMENTATIONS_UPDATE=1 \
third_party/RLDX-1/.venv/bin/python -u -m rl.offline_divl \
  --exp fuji --data rl-dataset/fuji-rl-dataset --checkpoints checkpoints \
  --steps 40000 --holdout 0.2 --eval-every 1000 \
  --atoms 201 --v-range "-0.1,1.1" --tau-base 0.6 --alpha 0.3 --nstep 1 \
  --num-qs 2 --discount 0.9999

# long-horizon 상당 (n=10)
  ... --nstep 10 --tau-base 0.6 --alpha 0.3

# Table III ablation: constant τ = 0.52
  ... --alpha 0 --tau-base 0.52
```

---

## 8. `offline_iql.py` 대비 변경 요약표

| 항목 | `offline_iql.py` (현재) | `offline_divl.py` (DIVL) | 논문 근거 |
|---|---|---|---|
| **V 표현** | 스칼라 (`CriticEnsemble(action_dim=0, num_qs=1)`) | **categorical K=201 logits, support [−0.1,1.1]** | Eq 20, A1 |
| **V 손실** | expectile `ρ_{τ,2}` (`:262-263`) | **C51 projection + cross entropy** | Eq 12, 21 |
| **V 타깃 라벨** | `min_2 Q̄(s,a_data)` (`:254-256`) | 동일 — 변경 없음 | Eq 12, §IV-D |
| **부트스트랩 값** | `V(s')` 스칼라 (`:249`) | **`Quant_τ(p_ψ(·\|s'))`** | Eq 13, 23 |
| **τ** | 고정 `--expectile` | **`clip(τ_base − α·H(s'), τ_min, τ_max)`, stop-grad** | Eq 17, 18 |
| **Q 표현** | 옵션 distributional (HL-Gauss, `--bins`) | **스칼라 MSE (논문 기본)** — `--bins` 는 우리 확장으로 유지 | Eq 15 |
| **Q 앙상블** | `num_qs=10`, `--v-min all/sub` | **2 (clipped double-Q)** 권장, 10 도 유지 | §IV-D |
| **TD 스텝** | 1-step (`nstep()`, 청크 내부만) | **chunk 단위 n-step (offline), n=1 (online)** | Eq 19 |
| **업데이트 순서** | `loss_q + loss_v` 동시 (`:264`) | **ψ → 타깃 → φ → EMA** (옵션) | Algorithm 2 |
| **target network** | Q + encoder (`target`, `tenc`) | 동일 — V 에는 target 없음 | §6-2 |
| **γ** | 0.999 (fuji) | 논문 **0.9999** | B2 |
| **EMA rate** | `cfg.tau=0.005` | 동일 (0.005) | B2 |
| **critic lr** | `cfg.critic_lr=3e-4` | 논문 **5e-4** | B2 |
| **actor** | 학습 안 함 | 학습 안 함 (QAM 은 별도) | — |

---

## 9. 읽은 위치 (재확인용)

| 내용 | 위치 |
|---|---|
| DIVL 본문 | §IV-A, p.5–6 (Eq 11–18) |
| n-step chunk TD | §IV-C, p.6 (Eq 19) |
| Algorithm 1 (offline→online 파이프라인) | p.7 |
| Algorithm 2 (LEARNER 1스텝) | p.7 |
| 아키텍처 | §IV-D, p.7–8 |
| DIVL vs expectile ablation | Table II p.11, Table V p.18 |
| adaptive τ ablation | Table III p.12 |
| 값 분포 시각화 | Fig 6 p.11, Fig 9 p.17 |
| categorical 이산화 상세 | Appendix A1, p.14 (Eq 20–24) |
| Proposition 1 증명 | Appendix A2, p.14–15 (Eq 25–31) |
| flow 직접 역전파 분석 (QAM 동기) | Appendix A3, p.15 (Eq 32–33) |
| offline 데이터 구성 | Appendix B1 p.15, Table IV p.16, Fig 7 p.16 |
| 하이퍼파라미터 | Appendix B2, p.15 |
| 체크포인트 초기화 | Appendix B3, p.15 |
| τ / 엔트로피 곡선 | Fig 8, p.17 |
| 분산 인프라 | Appendix D, p.17–18, Fig 10 |

우리 코드 기준점: `rl/offline_iql.py`, `rl/nets.py:180` (`CriticEnsemble`),
`rl/data.py:417` (`nstep`), `rl/expo.py` (`ExpoConfig`).
