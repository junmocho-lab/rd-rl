# PA-RL 오프라인 distillation — RLDX-1 action expert LoRA

critic 이 찾은 더 좋은 액션을 RLDX-1 정책에 심는다. PA-RL 의 distillation 을 **오프라인**으로
옮긴 것이다.

관련 문서: [PA-RL.md](PA-RL.md) (알고리즘 전체), [ACTION-OPT.md](ACTION-OPT.md) (액션 최적화
진단), [DIVL.md](DIVL.md).

---

## 1. 왜 손실을 안 건드리는가

PA-RL 의 distillation 은 정책의 손실 함수를 바꾸지 않는다. 배치의 `actions` 만 최적화된 액션으로
갈아끼우고 원래 BC 손실을 그대로 쓴다 (`third_party/PolicyAgnosticRL/train.py`):

```python
# train.py:249, 264 — preprocess_batch_with_action_optimization 안
batch["actions"] = action_distribution.sample(seed=rng)
# train.py:1122
base_policy_update_info = base_policy_agent.update(batch, timer=timer)
```

정책이 diffusion 이면 denoising MSE (`ddpm_bc.py:29`), OpenVLA 면 액션 토큰 CE
(`openvla.py:340`) — **그대로다**. critic 은 정책 내부를 모르고, 정책은 critic 을 모른다.
접점은 배치의 `actions` 필드 하나다. 그래서 "policy-agnostic" 이다.

우리는 그 교체를 **디스크 레벨**에서 한다. 데이터가 LeRobot parquet 이므로 `action` 컬럼을
덮어쓴 데이터셋 사본을 만들면, RLDX-1 의 학습 스택(RTC prefix 샘플링 / flow matching / 증강 /
EMA / 체크포인트)을 한 줄도 건드리지 않고 `--action_model_use_lora` 만 켜서 돌릴 수 있다.

RLDX-1 의 손실은 flow matching MSE 다 (`rldx/model/core/rldx.py:436`):

```python
action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * loss_mask
loss = action_loss.sum() / (loss_mask.sum() + 1e-6)
```

`velocity = actions - noise` 이므로 `actions` 를 바꾸면 타깃 속도장이 바뀐다. 그게 전부다.

## 2. 프레임 정렬 — 청크 타깃을 프레임 액션으로 되돌리는 유일한 방법

parquet 의 `action` 은 **프레임당 1스텝 28차원**이다 (청크는 학습 파이프라인이
`action[t:t+16]` 로 조립한다). 우리 최적화 결과는 청크 모양이라 되돌려야 하는데, RTC 실행
의미론이 그것을 하나로 결정한다.

```
결정 프레임 t (replan 8 간격)에서 최적화
  → 편집 창은 청크 [latency, latency+replan) = [2, 10)      ← 로봇이 실제 실행하는 구간
  → 전역 프레임 t+2 … t+9
다음 결정은 t+8 → 프레임 t+10 … t+17
```

**latency 이후 모든 프레임이 정확히 한 번 덮인다.** 겹침도 빈틈도 없다. 에피소드 앞 2프레임만
원본 액션을 유지한다. openarm 실측:

```
결정 8301개 → 덮는 프레임 65,328 / 65,928 (99.1%)
중복 덮인 프레임 0
안 덮인 프레임 600 = 에피소드당 2.0, 위치 [0, 1]      ← latency 와 정확히 일치
```

latency prefix 를 편집하지 않는다는 마스킹 규칙(우리가 `ExploreSpec` 에 넣은 것)이 오히려 이
정렬을 깔끔하게 만들어 준다.

### 관절 순서 함정

parquet 의 `action` 컬럼 **내부 순서**가 canonical concat 순서와 다르다. openarm 실측:

| | canonical (`flat.action`) | parquet `action` |
|---|---|---|
| neck_joints | 0:2 | 0:2 |
| left_arm_joints | 2:9 | 2:9 |
| right_arm_joints | 9:16 | **15:22** |
| left_hand_joints | 16:22 | **9:15** |
| right_hand_joints | 22:28 | 22:28 |

`right_arm` 과 `left_hand` 가 바뀐다. 그대로 쓰면 관절이 조용히 뒤섞인다. `mod.action` 의
`(key, s, e)` 가 원본 슬라이스이므로 그것으로 되돌리고, 되돌린 결과를 다시 gather 해서
원래 값이 나오는지 왕복 검증한다 (`relabel_parl.py`, 300/300 통과).

## 3. 후보 구성 — EXPO 온라인 경로와 같게

`rl/expo.py:225-231` 와 같은 규약을 쓴다:

1. base policy 에서 **M=32** 청크를 뽑고 앞 `latency+replan = 10` 스텝만 쓴다 (280차원)
2. latency prefix 블록(56차원)을 **로그된 값으로 덮는다** — 실행이 이미 확정된 구간이라
   후보끼리 같아야 한다
3. 앙상블 **mean** Q 로 상위 K-1=9 개를 남기고, **로그된 액션 자체를 후보로 하나 넣는다**

3번이 PA-RL 의 안전장치다 (`action_optimization.py:435-455`, 주석이 `skip the worst action`).
오프라인에서는 롤아웃으로 회복할 수 없으니 온라인보다 더 중요하다 — 최악의 경우 원래 BC 로
수렴한다. `relabel_parl.py` 는 **로그된 액션이 이긴 비율**을 찍는다. 이 값이 높으면 critic 이
base policy 를 개선하지 못한다는 뜻이고, distillation 이 거의 no-op 이 된다.

4. `a ← a + step_size · ∇_a Q̄` 를 10스텝. 마스크는 `explore_groups × 스텝 [2,10)` 뿐
5. 최종 Q 로 argmax (또는 `Categorical(Q/temp)`)

### PA-RL 기본값을 그대로 쓸 수 없는 두 곳

- **`step_size=3e-4`**: 우리 Q·액션 스케일에서는 이동이 1e-9 수준이라 사실상 아무 일도
  안 한다. `--auto-step D` 로 `‖g‖` 를 재서 `step_size = D/(num_steps·median‖g‖)` 로 잡는다
  (`probe_actopt.py` 와 같은 규칙)
- **온도**: PA-RL 은 `logits = q_values` 를 그대로 Categorical 에 넣는다
  (`action_optimization.py:521`). 우리 Q ∈ [0,1] 이라 후보 간 격차가 0.01 수준 → 거의 균등
  샘플링이 된다. `--temp 0` (argmax, PA-RL 의 CALVIN 설정 `calvin_config.py:55`) 이 기본이고,
  분포를 남기고 싶으면 `--temp 0.02` 정도

## 4. 이게 PA-RL 인가

**아니다 — PA-RL 은 온라인에서만 distillation 한다** (`train.py:1027`):

```python
and i >= FLAGS.num_offline_epochs        # 오프라인 단계에서는 정책을 건드리지 않는다
and FLAGS.num_online_epochs > 0
```

오프라인에서 PA-RL 이 하는 것은 critic 학습뿐이고, 정책은 "추론 시점에 후보를 뽑아
critic 으로 고르는" 방식으로만 쓴다.

다만 근거 없는 변형은 아니다. **IDQL** 이 정확히 이 구조다 — BC 정책에서 N개 샘플하고
advantage 로 재가중/선택해서 BC 한다. 우리는 거기에 PA-RL 의 gradient refinement 를 얹은
것이다. 이름을 붙이면 "IDQL + local action optimization, distilled".

**리스크는 되돌릴 수 없다는 점이다.** `probe_actopt.py` 처럼 추론 시점에 최적화하면 critic
오차가 매 스텝 버려지지만, LoRA 가중치에 들어가면 남는다. 그래서 아래 관문을 먼저 통과한다.

## 5. 관문 — distillation 전에 확인할 것

`probe_actopt.py` 로 본다. 세 가지가 동시에 성립해야 진행한다.

| 확인 | 통과 기준 | 실패면 |
|---|---|---|
| critic 이 성패를 구분하는가 | 홀드아웃 AUC > 0.8, **초반 프레임 AUC ≈ 0.5** | 에피소드 암기 — 데이터/특징 문제 |
| 최적화가 액션을 움직이는가 | 이동거리/차원이 실패 구간에서 커진다 | critic 이 액션을 무시 — 재학습 |
| 로그된 액션이 항상 이기지 않는가 | 로그 승률 < 70% | 개선 여지 없음 — distillation 이 no-op |

더 강한 검증이 필요하면 **critic 두 개 교차검증**: 세션을 나눠 A/B 를 각각 학습하고, A 로
최적화한 액션을 B 로 점수 낸다. B 가 로그 액션보다 높게 주면 개선이 실재하는 신호다.

```bash
... -m rl.offline_iql --holdout 161556 ...     # critic A: 세션 161556 을 평가로 뺀다
... -m rl.offline_iql --holdout 152925 ...     # critic B: 세션 152925 을 평가로 뺀다
```

기본값이 아닌 `--holdout` 은 태그에 `-h<값>` 으로 들어가므로 두 런이 서로 다른 디렉토리에
저장된다 (`iql-cog-dist128-t07-g0995-q10all-s0-h161556`).

## 6. 실행

### (0) 전제 — base policy 가 데이터를 만든 그 모델이어야 한다

`rl-dataset/0825_openarm_f1_inference/checkpoint-meta.yml` 은
`training_run_id: 0825_openarm, step: 60000` 인데, `configs/exp/openarm_rim.yaml` 의
`base_policy` 는 `openarm_0818_0819_0821_..._30k_...` 다. **다른 체크포인트다.**

critic 학습은 정규화 통계와 cog feature 만 쓰므로 이 불일치가 치명적이지 않지만,
distillation 은 **데이터를 만든 정책을 튜닝해야 한다**. `0825_openarm` step 60000 을
`checkpoints/` 로 받아 `--model-path` 로 지정하거나 yaml 의 `base_policy` 를 고칠 것.

### (1) 액션 relabel

```bash
PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" NO_ALBUMENTATIONS_UPDATE=1 \
third_party/RLDX-1/.venv/bin/python -u -m rl.relabel_parl \
  --exp openarm_rim --data rl-dataset/0825_openarm_f1_inference \
  --checkpoints checkpoints --features cogfeat.npy \
  --critic iql-cog-dist128-t07-g0995-q10all-s0/critic_latest.pt \
  --num-samples 32 --num-keep 10 --num-steps 10 --auto-step 0.02 --temp 0 \
  --out rl-dataset/0825_openarm_f1_parl
```

먼저 `--dry-run --limit 200` 으로 통계만 보고 판단할 것. 출력 지표:

```
ΔQ (선택 - 로그)   평균 / 중앙 / p95 / 개선된 비율
이동거리/차원      평균 / 중앙 / p95        (액션 공간 ±1)
로그된 액션이 이긴 비율                     ← 높으면 no-op
raw 액션이 바뀐 프레임 수
```

산출물은 `<out>/` 에 세션별로 parquet 재작성 + `meta/` 복사 + **`videos/` 심링크** (24GB 를
다시 쓰지 않는다).

비용: 결정 8,301개 × 후보 32개. 백본은 결정당 1회만 돌고 (`vla.sample` 의 `expanded()` 가
feature 를 복제한다) action expert 롤아웃만 32배다.

### (2) action expert LoRA 로 학습

RLDX-1 에 이미 다 있다 (`rldx/model/core/rldx.py:219` `_apply_action_model_lora`): MSAT DiT
전체를 freeze 하고 PEFT LoRA 만 주입한다. 체크포인트 config 의 기본값:

```
action_model_lora_rank 16, alpha 32, dropout 0.0
target_modules [vl_qkv, vl_proj, sa_qkv, sa_proj, p_qkv, p_proj, linear1, linear2]
```

```bash
cd third_party/RLDX-1
.venv/bin/python -m rldx.experiment.launch_train \
  --base-model-path <데이터를 만든 체크포인트> \
  --dataset-path ../../rl-dataset/0825_openarm_f1_parl \
  --embodiment-tag general_embodiment \
  --action-model-use-lora True \
  --tune-projector False --tune-llm False --tune-visual False \
  --learning-rate 1e-4 --global-batch-size 64 --max-steps 3000 --save-steps 500 \
  --save-trainable-only True \
  --output-dir ../../checkpoints/openarm_parl_lora
```

`--action-model-use-lora True` 를 켜면 `tune_diffusion_model` 은 자동으로 꺼진다
(`rldx/experiment/assembly.py:146-148`). `--save-trainable-only` 로 어댑터만 저장한다.

**max-steps 는 작게 잡을 것.** PA-RL 은 UTD 1 로 "그 에폭에 모은 환경 스텝 수" 만큼만
업데이트한다 (`train.py:1049`). 오프라인이라 대응값이 없지만, 데이터셋 1~2 epoch
(65,928/64 ≈ 1,030 스텝/epoch) 정도가 정신이다. 오래 돌리면 critic 오차에 과적합한다.

### (3) 추론용으로 병합

```bash
.venv/bin/python scripts/merge_lora_checkpoint.py \
  --trainable-ckpt ../../checkpoints/openarm_parl_lora/checkpoint-XXXX \
  --base-ckpt <데이터를 만든 체크포인트> \
  --out ../../checkpoints/openarm_parl_merged
```

병합 후에는 `AutoModel.from_pretrained` 가 그냥 읽는다 — 추론 코드 변경 없음.

### (4) 평가

`--model-path openarm_parl_merged` 로 `probe_actopt.py` 를 다시 돌린다. distillation 이
먹었다면 **최적화 전 Q(로그 액션) 가 올라가 있고, 이동거리는 줄어야 한다** — 정책이 이미
critic 이 원하는 액션을 내고 있다는 뜻이다. 이동거리가 그대로면 LoRA 가 타깃을 못 따라간
것이고, Q 는 올랐는데 실기 성공률이 안 오르면 critic 을 착취(exploit)한 것이다.

## 7. 우리 쪽에서 달라지는 것 정리

| PA-RL | 우리 |
|---|---|
| 액션 1스텝 7차원 (`shape[1] == 1` assert) | 청크 10스텝 × 28관절 = 280차원 |
| 전 차원 gradient | `explore_groups × [latency, latency+replan)` 만 (`spec.index`) |
| OpenVLA 7B 전체에 LoRA (`all-linear`, r=32, lr 2e-5) | action expert MSAT 만 (r=16, backbone 동결) |
| 액션 토큰 256-bin CE → 양자화 손실 | flow matching MSE → 양자화 손실 없음 |
| `logits = Q` 그대로 | Q ∈ [0,1] 이라 온도 필요 |
| 온라인 UTD 1 | 오프라인 1~2 epoch |
| 배치 메모리에서 relabel | 데이터셋 디스크에서 relabel |
