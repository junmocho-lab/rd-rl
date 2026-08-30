# DexJoCo IQL

## 1. BC

claude한테 dexjoco bc dataset 받아서 RLDX-1 학습에 맞는 데이터셋 포맷으로 변환해달라고 요청. bc 학습은 `sbatch/dexjoco/bc` 파일 참고.

## 2. Rollout
rldx-bc policy를 rollout한다. 현재 버전은 hammer nail 태스크. 그리고 seed를 설정하여 환경 시작 state를 고정할 수 있음. `sim/dexjoco/rollout_dexjoco.py` 에서 rollout 진행. rollout은 `sbatch/dexjoco/rollout` 참고.

## 3. Critic Learning
rollout한 데이터를 기반으로 critic을 학습한다. `sbatch/dexjoco/critic` 참고.

## 4. Rollout with Critic
학습한 critic을 활용하여 rollout을 진행한다. Test-time Q selection, Test-time Q guidance 같은 방법을 활용할 수 있다. action distillation을 하기전에 실제 critic으로 최적화한 action이 좋게 작동하는지 확인이 가능하다. `sbatch/dexjoco/rollout_w_critic` 참고.

## 1. 태스크: `hammer_nail`

### 1.1 왜 이 태스크인가

DexJoCo 논문 Table 2, `rand-obj` regime. RLDX-1 과 계보가 가장 가까운 모델은
**GR00T N1.5** (둘 다 GR00T 파생 + flow-matching action head + LoRA finetune) 이므로
그 열을 BC 기대값으로 잡는다.

| task | GR00T N1.5 | π0.5 | 평균 길이 | arm |
|---|---|---|---|---|
| **hammer_nail** | **67.3 ± 4.2** | 84.7 | **215.7 frame / 7.2 s** | single |
| water_plant | 72.7 ± 1.2 | 88.7 | 277.4 / 9.2 s | single |
| click_mouse | 85.3 ± 3.1 | 64.7 | 326.8 / 10.9 s | single |
| pick_bucket | 72.0 ± 6.0 | 84.0 | 433.0 / 14.4 s | single |
| fold_glasses | 27.3 ± 2.3 | 72.0 | 536.3 / 17.9 s | single |
| pinch_tongs | 12.7 ± 2.3 | 24.0 | 400.6 / 13.4 s | single |
| bimanual 5종 | 0.7 ~ 50.7 | 5.3 ~ 70.0 | 342 ~ 1073 | dual |

선정 근거 — offline RL 검증에 필요한 조건 순서대로:

1. **BC 성공률이 중간 대역 (67%)**. 0% 면 롤아웃에 positive 가 없어 critic 이 학습 자체가
   안 되고, 90% 면 개선 폭이 노이즈에 묻힌다. 67% → 100 에피소드 롤아웃에서 성공 67 / 실패 33
   으로 **양쪽 라벨이 다 충분히** 모인다. offline IQL 은 expectile 회귀라 positive/negative
   비율이 극단이면 V 가 붕괴한다.
2. **11 태스크 중 최단 horizon (215.7 frame)**. 롤아웃 100 에피소드 = 약 21.6k step.
   실패 에피소드는 step cap 1000 까지 가지만 우리 롤아웃 루프에서 400 으로 잘라 쓸 수 있다
   (§6.4). n-step return 의 크레딧 전파 거리도 짧아 discount 튜닝이 덜 예민하다.
3. **성공 조건 안에 단조 진행량이 있다.** `nail_depth` 가 타격마다 증가하고 성공은
   depth ≥ 0.04 m. 보상은 여전히 sparse terminal 이지만, 내부 상태가 단조라서 학습된 V 가
   "말이 되는지" 를 눈으로 검증할 수 있다 (`offline_iql.py --eval-every` 가 뽑는 Q/V/A 궤적
   플롯이 실제로 읽힌다). openarm/fuji 에서 critic 이 에피소드를 암기했는지 판단하기
   어려웠던 문제가 여기서는 훨씬 쉽다.
4. **guidance 가 개입할 여지가 물리적으로 존재한다.** 못 삽입량이 타격 속도에 비례한다
   (§1.3) — 즉 같은 관측에서 후보 청크들의 Q 가 실제로 갈라질 이유가 있다.
   `candidate_q_std ≈ 0` 이면 guidance 가 무의미한데, 그 위험이 낮다.

백업: **`water_plant`** (72.7%). 조금 길지만 **실패 시 조기 종료** (플랜트 영역 밖에서
트리거를 당기면 즉시 terminate) 라 negative 가 짧고 깔끔하다. 컨버터/모달리티 설정이
동일해 태스크 이름만 바꾸면 된다. 데이터는 이미 변환해 뒀다.

### 1.2 태스크 정의

- 언어 지시: `"Use the hammer to drive the nail into the wooden board."`
- 씬: 테이블 위에 망치(hammer), 못(nail), 나무판(wood board).
- 목표: 망치를 집어 못을 나무판에 박는다.
- 파일: `dexjoco/dexjoco/sim/envs/panda_hammer_nail_env.py`,
  XML `xmls/arena_arm_hand_hammer_nail.xml`

### 1.3 성공 조건과 보상 (`panda_hammer_nail_env.py`)

```
성공      nail_depth >= success_depth (0.04 m)                       :601
삽입      hammer geom ↔ nail geom 접촉 && 접촉 직전 vz < -0.02 m/s     :637
          delta = 0.008 * min(3.0, |vz| / 0.02)                      :642-643
          nail_depth += delta,  clip(0, 0.0726)                      :607
보상      r = 1.0 if success else 0.0                                 (step)
종료      success  or  env_step >= 1000                               :571,585
info      {"succeed": bool, "nail_depth": float, "hammer_hit": bool}
```

즉 최소 5회의 유효 타격이 필요하고, **세게 칠수록 적은 횟수로 끝난다** (scale 최대 3배 →
최소 2회). `nail_depth` / `hammer_hit` 가 info 로 나오므로 롤아웃 때 per-frame 으로 같이
기록해두면 critic 진단용 보조 라벨이 된다 (학습에는 쓰지 않는다 — 보상은 sparse 로 유지).

### 1.4 랜덤화

| | rand_obj (우리가 쓰는 것) | rand_full |
|---|---|---|
| 물체 배치 | hammer xy ∈ [-0.25,-0.35]×[-0.40,-0.50], yaw ±10°<br>nail xy ∈ [-0.10,0.00]×[0.00,0.10] | 동일 |
| 테이블 높이 | delta_h ~ U(0, 0.05) m (다리 길이까지 같이 늘어남) | 동일 |
| 카메라/조명/텍스처 | 고정 | 3인칭 카메라 50 프리셋 중 랜덤, 조명 pos/dir/diffuse, 테이블 텍스처 20종 |
| dynamics | 고정 | `--randomize-dynamics` 로 별도 (물체 질량 ×0.75~1.25) |

**rand_obj 를 쓴다.** 목적이 "critic 이 효과가 있는가" 이므로 시각 일반화 난이도를 올려
BC 를 깎을 이유가 없다 (rand_full 에서 GR00T 는 67.3 → 38.7 로 떨어진다). `randomize=True`
로 켜면 env 가 `front` 카메라 키를 **`random_camera`** 로 바꿔 내보내므로 롤아웃 쪽 키
매핑도 같이 바꿔야 한다는 점만 기억.

---

## 2. 로봇 / 손

| | 내용 |
|---|---|
| 팔 | Franka Emika Panda, 7 DOF. joint 이름 `joint1..joint7`, actuator `actuator1..actuator7` (**토크** actuator) |
| 홈 자세 | `(0, -0.785, 0, -2.35, 0, 1.57, π/4)` |
| 손 | Wonik Allegro right hand, 16 DOF = 4 finger × 4 joint. **position** actuator, `kp=1` |
| 손 joint 순서 | `ffj0..3, mfj0..3, rfj0..3, thj0..3` (index, middle, ring, thumb) |
| 손 ctrlrange | base `±0.47`, proximal `-0.196~1.61` 등 joint 별로 다름 |
| 홈 자세(손) | reset 시 `_allegro_default_pose` = `[0.47,0.333,-0.0015,-0.227, 0.47,0.309,…, 1.058,-0.105,0.365,-0.162]` |
| EEF site | `attachment_site` (Panda flange). 센서 `franka/flange_pos`, `franka/flange_quat` |
| 카메라 | `front` (3인칭), `wrist` (`handcam_rgb`), `ego_left`, `ego_right`. 640×640 렌더 |
| 자산 라이선스 | panda Apache-2.0, allegro BSD-2-Clause |

**팔은 토크 제어 + opspace, 손은 위치 제어**라는 비대칭이 뒤에서 계속 중요하다 (§4).

---

## 3. 제어 주기

여기가 문서/코드가 서로 다른 숫자를 말하는 지점이라 정확히 적어둔다.

| 이름 | 값 | 의미 |
|---|---|---|
| `physics_dt` | 0.002 s | MuJoCo `opt.timestep`. 즉 물리 500 Hz |
| `control_dt` | 0.02 s | `_n_substeps = control_dt // physics_dt = 10`. **env.step 1회 = 물리 10 스텝 = 20 ms** → 제어 50 Hz |
| `hz` | 30 | `step()` 끝의 `time.sleep(max(0, 1/hz - dt))`. **벽시계 throttle 일 뿐 sim 시간과 무관** |
| `data_fps` | 30 | 레코더가 `timestamp = arange(T)/30` 으로 기록 |
| `video_fps` | 30 | mp4 인코딩 fps |

정리: **sim 상으로 1 step = 20 ms (50 Hz)** 인데 **데이터셋 timestamp/영상은 30 fps 로
적혀 있다**. 벤치마크 자체의 불일치다. 우리에게 중요한 것은 "한 step 에 한 action" 이라는
사실뿐이므로, 데이터셋 `fps` 는 영상·타임스탬프와 맞춰 **30 으로 선언**했다 (dexjoco 의
openpi eval 도 30 으로 간주하고 동작한다). 물리 시간 기준의 절대 속도를 논할 때만
50 Hz 를 쓸 것.

롤아웃에서 `time.sleep` 은 **반드시 없애야 한다** — 그냥 두면 실시간 30 Hz 로 묶여서
100 에피소드가 최소 20분 + 추론시간이 된다. `hz` 를 큰 값으로 주거나
(`get_environment(..., hz=10000)`) `step` 을 monkeypatch 한다.

action horizon 32 는 데이터 기준 32/30 = **1.07 s**, 물리 기준 32×20 ms = **0.64 s**.

---

## 4. Action space / State (proprioception)

### 4.1 env 가 실제로 받는 action

`SingleArmPolicyWrapper` (`tasks/policy_wrappers.py`) → shape **(23,)**:

```
action[0:3]    EEF 목표 위치 xyz         (world frame, 절대값)
action[3:7]    EEF 목표 자세 quat wxyz   (절대값)
action[7:23]   Allegro 16 joint 목표각    (절대값)
```

env 내부 처리 (`panda_hammer_nail_env.step`):

```
data.mocap_pos[0]  = action[0:3]        # mocap body 에 목표 pose 를 꽂는다
data.mocap_quat[0] = action[3:7]
for _ in range(10):                     # 10 substep
    tau = opspace(model, data, site_id=attachment_site, dof_ids=panda_dof_ids,
                  pos=mocap_pos, ori=mocap_quat, joint=_PANDA_HOME,
                  pos_gains=(400,400,400), damping_ratio=4, gravity_comp=True)
    data.ctrl[panda_ctrl_ids]  = tau         # 팔: 토크
    data.ctrl[allegro_ctrl_ids] = action[7:23]   # 손: 위치 목표
    mj_step(...)
```

> pose 가 전부 0 이면 (`allclose(tpos,0) and allclose(tquat,0)`) 이전 mocap 을 유지한다
> = "hold". 데이터 변환에서 이 프레임을 걸러내야 하는 이유 (§5.3).

**즉 팔의 action 은 구조적으로 EEF pose 다.** joint 목표를 직접 주는 경로가 없다 —
Panda actuator 가 토크형이고 그 토크를 만드는 것이 opspace 이기 때문이다. joint action 으로
가려면 벤치마크에 joint PD 컨트롤러를 새로 붙여야 하고, 그러면 논문 숫자와의 비교
기준(67.3%)이 무의미해진다.

### 4.2 env 가 내보내는 state

`DexjocoObsAdapter` 가 태스크별 `proprio_keys` 를 순서대로 concat 한다.
hammer_nail: `[tcp_pose, gripper_pose, hammer_ori_pose, nail_ori_pose, table_delta_height]`

```
state[0:7]    tcp_pose   = flange_pos(3) + flange_quat wxyz(4)   ← proprio
state[7:23]   gripper_pose = Allegro 16 joint 측정각              ← proprio
state[23:30]  hammer_ori_pose   (리셋 시 망치 pose)               ← privileged
state[30:37]  nail_ori_pose                                       ← privileged
state[37:38]  table_delta_height                                   ← privileged
```

**앞 23 차원만 proprioception** 이고 뒤는 privileged (벤치마크의 replay/state 복원용).
DexJoCo README 도 "policy training 은 앞 23 차원만" 이라고 명시한다.

### 4.3 ⚠️ 열린 결정: openarm 과 얼마나 똑같이 맞출 것인가

목표는 **openarm + RH56-F1 실험과 최대한 같은 모양** 이다. 그 기준선을 먼저 정확히
적어둔다 (`modality/openarm_lefthand/modality.json`, `openarm_inspire_config.py`,
`configs/exp/openarm_rim.yaml`):

| | openarm (real, 기준) | dexjoco 선택지 A (지금 구현된 것) |
|---|---|---|
| state | `observation.joint_position` 28 = neck 2 + L arm 7 + L hand 6 + R arm 7 + R hand 6 | `observation.state` 25 = EEF xyz 3 + rot6d 6 + hand 16 |
| action | 같은 28 joint position, 절대 | EEF pose(rot6d) 9 + hand joint 16, 절대 |
| privileged | 없음 (이미지 + proprio 만) | **없음** — `state[23:38]` (망치/못 pose, 테이블 높이) 를 버린다 ✅ |
| 카메라 | ego stereo 2대 | front + wrist 2대 |
| horizon / replan / latency | 16 / 8 / 2 | 32 / (16) / (4) — 제안값 |
| explore_groups | `right_arm_joints` | `eef_position` 등 |

**privileged 를 안 준다는 조건은 이미 만족한다.** 남은 차이는 **action/state 가 joint 인가
EEF 인가** 하나다.

문제: **dexjoco 의 팔은 구조적으로 joint 제어가 아니다.**

- Panda actuator 는 전부 `<motor ctrlrange="-87 87">` = **토크** actuator
  (`franka_emika_panda/panda.xml:362`). 그 토크를 만드는 것이 opspace 이고 opspace 의 입력은
  EEF pose 다. joint 목표를 직접 넣는 경로가 없다.
- 손 16개는 `<position kp="1">` = **위치 actuator** 이므로 손은 이미 joint 제어다.
- **팔 joint 각도가 raw 데이터에 아예 없다.** `proprio_keys` 는 `tcp_pose` 뿐이고
  `data.qpos[panda_dof_ids]` 는 기록되지 않는다. tcp_pose 에서 IK 로 복원하는 것도
  7-DOF 팔이라 nullspace 1차원이 남아 유일하지 않다.

그래서 세 가지 길이 있다.

#### A. EEF pose + hand joint (지금 구현·검증 완료)

추가 작업 0. 벤치마크 native action 이라 롤아웃이 정확하고 논문 67.3% 기준선이 살아 있다.
RLDX-1 쪽도 검증된 조합이다 — `libero_config.py` 는 state `[eef_pos_absolute,
eef_rot_absolute, gripper_close]`, `droid_*` 는 `[end_effector_position,
end_effector_rotation, gripper_position]` 로 EEF state/action 을 쓴다.
`rl/` 스택은 그룹 이름·차원을 하드코딩하지 않으므로 (`rl/data.py::Modality` 가
`modality.json` + 등록 config 에서 전부 유도) critic/guidance 코드도 그대로 돈다.

단점: **openarm 과 proprio/action 의 의미가 다르다.** 즉 "joint 공간에서 critic 이
액션을 편집한다" 는 real 실험의 성질을 그대로 재현하지 못한다.
(rot6d 6차원은 스케일이 서로 얽혀 있어서, 정규화 공간에서 한 차원을 미는 것이 관절 하나를
미는 것과 물리적 의미가 다르다.)

#### B. 완전 joint 공간 (openarm 과 동형)

```
state  = observation.joint_position 23 = arm_joints 7 + hand_joints 16
action = 같은 23, 절대 joint position
```

필요한 것 두 가지:

1. **데이터**: `scripts/replay_demos_zarr.py` 로 100 데모를 replay 하면서
   `data.qpos[panda_dof_ids]` (7) + `data.qpos[allegro_dof_ids]` (16) 을 추가 로깅한다.
   replay 는 `policy_mode=True` env 에 기록된 action 을 그대로 다시 넣고
   `--restore_state` 로 초기 물체 pose·테이블 높이를 복원하며, **데모마다 성공 여부를
   보고한다** (`replay_demos_zarr.py:337`, `--save_failed` 없으면 실패는 버린다).
   즉 replay 충실도를 **측정할 수 있다** — 이게 이 경로의 게이트다.
   - action 정의: 팔은 joint command 가 기록에 없으므로 **a_t = q_{t+1}^측정**
     (next-state-as-action, joint position 데이터셋의 표준 관례). 손은 actuator 입력이
     그대로 기록돼 있으므로 **a_t = raw `action[7:23]` (명령값)** 을 쓴다 — kp=1 로
     부드러워서 측정값을 목표로 쓰면 파악력이 부족해진다.
2. **롤아웃 컨트롤러**: 팔에 joint PD + gravity comp 를 붙인다.
   `tau = kp(q_des − q) − kd·qd + qfrc_bias`. env 코드를 복제하지 않는 깔끔한 방법은
   env 모듈이 import 한 `opspace` 심볼을 우리 함수로 monkeypatch 하는 것 —
   substep 루프·망치/못 상호작용·성공 판정·info 가 전부 원본 그대로 유지된다
   (`panda_hammer_nail_env.py` 는 `from ..controllers import opspace` 후
   `tau = opspace(...)` 를 호출한다).

리스크: **hammer_nail 의 삽입량이 타격 직전 수직 속도에 비례한다** (§1.3). PD 게인이
무르면 궤적이 뒤처져 접촉 속도가 낮아지고 못이 안 박힌다. 그래서 게인 튜닝이 필수이고,
**게이트는 "joint PD 로 replay 했을 때 원래 100 데모의 성공률"** 이다. 90% 이상 나오면
합격, 낮으면 게인/타깃 정의를 고치거나 A 로 돌아간다.

작업량: replay+로깅 스크립트 1개, joint PD patch 1개, 컨버터에 `--state-source joint`
분기. GPU 1장으로 replay 100 에피소드 (EGL) 는 수십 분 규모.

#### C. joint state + EEF action (중간)

state 만 joint 로 (replay 필요), action 은 A 그대로. 컨트롤러를 새로 안 만들어도 되므로
B 의 리스크 절반이 사라지지만, **critic 이 편집하는 대상이 여전히 EEF** 라서 openarm 과의
핵심 차이는 남는다. B 의 게이트(replay 성공률)만 먼저 확인하는 중간 단계로는 쓸모가 있다.

#### 참고: dexjoco 를 안 쓰는 선택지 — GR1 tabletop

RLDX-1 에 **이미 붙어 있는** sim 벤치마크 중 하나가 openarm 과 거의 동형이다.

- `rldx/configs/data/gr1_config.py`: state/action 모두 **절대 joint position**, 그룹
  `left_arm / left_hand / right_arm / right_hand / waist`, `delta_indices=range(16)`,
  전부 `ABSOLUTE / NON_EEF / DEFAULT`. 손은 Fourier 6-DOF = **RH56-F1(6-DOF)과 같은 규모**
  (Allegro 16-DOF 보다 openarm 에 가깝다).
- 롤아웃 하네스가 이미 있다: `run_scripts/eval/gr1_tabletop/{setup_gr1.sh,eval_gr1.sh}` +
  `rldx/eval/rollout_policy.py`. → **§7 의 롤아웃 클라이언트 작업이 전부 사라진다.**
- 데이터셋도 이 클러스터에 있다: `/rlwrld4/sejune/ft_gr1_merged_100x24_2400` —
  LeRobot v2.0, `GR1ArmsAndWaistFourierHands`, **2400 ep (100 × 24 태스크) / 602,846 frame /
  20 fps**, `observation.state` 44 = arm 7+7 / hand 6+6 / leg 6+6 / neck 3 / waist 3,
  `observation.images.ego_view` 256×256. modality.json 도 joint 그룹으로 이미 갈려 있다.

즉 **joint 공간 + dexterous hand + 기존 롤아웃 하네스 + 기존 데이터셋**이 한 번에 만족된다.
대신 잃는 것:

- 태스크별 BC 성공률이 **알려져 있지 않다** (dexjoco 는 논문 Table 2 로 67.3% 라는 sweet
  spot 을 미리 알고 골랐다). GR1 은 BC 를 돌려보고 성공률을 재야 어느 태스크가
  offline RL 검증에 적합한지 안다.
- 카메라가 ego 1대 (openarm 은 stereo 2대).
- 남의 디렉토리 데이터라 출처/전처리를 확인해야 한다.

**추천**: openarm 동형성이 최우선이면 **GR1 tabletop 을 먼저 조사**하는 것이 총 작업량이
가장 적다 (컨버터·컨트롤러·클라이언트 셋 다 불필요). dexjoco 를 유지하겠다면 **B** 로
가되, **joint PD replay 성공률 게이트를 BC 전에 통과**시킨 다음 학습을 던지는 순서를 권한다.
A 는 "일단 파이프라인 전체를 한 번 관통시켜 본다" 는 목적에는 지금 당장 쓸 수 있는 상태다.

### 4.4 회전 표현: rotvec 을 쓰면 안 된다 (측정치)

dexjoco 자체 컨버터는 `action_rotvec` (22차원 = xyz3 + rotvec3 + hand16) 을 학습 타깃으로
쓴다. **우리 세팅에서 이건 못 쓴다.**

DexJoCo 의 EEF 는 180° 회전 근처에서 동작한다 (state quat w ≈ 0, 예: 첫 프레임
`quat_wxyz = (4e-11, -1.0, 1.7e-8, -0.0025)`). rotvec 은 norm ≤ π 로 정규화되므로 이 자세가
정확히 **±π anti-pode** 에 앉는다. 공개 데이터셋에서 직접 센 값:

| | 전체 프레임 | 연속 프레임 간 3 rad 이상 점프 | 영향받은 에피소드 |
|---|---|---|---|
| hammer_nail | 21,571 | **216 (1.01%)**, 최대 2π 플립 | **66 / 100** |
| water_plant | 27,745 | 212 (0.77%) | 66 / 100 |

같은 구간에서 **quaternion 은 매끄럽다** (연속 프레임 최대 차이 0.023). 즉 불연속은 순수히
표현 문제다. 그런데 32-step **절대** action chunk 회귀에서 청크 안에 플립이 들어가면 그건
학습 불가능한 타깃이고, 에피소드당 평균 2~3번 발생한다.

quaternion 으로 바꾸는 것도 답이 아니다: q / −q 이중덮개 때문에 부호 정규화가 필요하고,
정규화는 **에피소드 내부에서만** 일관되므로 서로 다른 에피소드가 같은 화면에 반대 부호
타깃을 붙일 수 있다. 180° 근처에서는 "고정 반구" 같은 전역 규칙도 연속이 될 수 없다.

→ **rot6d** (회전행렬의 **첫 두 행** flatten. RLDX-1 `EndEffectorPose._matrix_to_rot6d` 와
같은 규약, `pose.py:458`). 연속이고 이중덮개가 없다. 변환 후 실측:

```
state.eef_rotation   step-jump  mean 0.0138  p99 0.0359  max 0.0360   행 norm 1.0000, |행1·행2| 6.7e-8
action.eef_rotation  step-jump  mean 0.0451  p99 0.2496  max 0.2512   행 norm 1.0000, |행1·행2| 6.4e-8
```

rotvec 경로의 max 2.0 (rot6d 환산) → 0.25. 플립이 사라졌고 직교정규성도 정확하다.

### 4.5 최종 레이아웃 (선택지 A)

state 와 action 이 대칭인 25차원:

```
eef_position [ 0: 3]   3    xyz, world frame, 절대
eef_rotation [ 3: 9]   6    rot6d (R 의 첫 두 행)
hand_joints  [ 9:25]  16    Allegro ffj0-3 / mfj0-3 / rfj0-3 / thj0-3
```

- state = **측정된** flange pose + 측정된 손 joint 각
- action = **명령된** mocap 목표 pose + 손 joint 목표각 (delta 아님, 절대값)
- 25 ≪ `RLDXConfig.max_state_dim = max_action_dim = 64` (`configs/model/rldx.py:81`).
  processing 이 64 로 zero-pad 하므로 PT-IMG 체크포인트의 state/action encoder 와
  action decoder 가 **재초기화되지 않는다** (rby1m_wujihand2 가 66 DOF 로 터졌던 문제와
  반대 상황).

---

## 5. RLDX-1 용 데이터 변환

### 5.1 RLDX-1 이 요구하는 것

`rldx/data/dataset/lerobot_episode_loader.py` 기준 — **LeRobot v2.1** 레이아웃:

```
<dataset>/
  data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
  videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
  meta/info.json        codebase_version, fps, chunks_size, data_path, video_path, features
  meta/episodes.jsonl   {"episode_index", "tasks": [...], "length"}  한 줄에 한 에피소드
  meta/tasks.jsonl      {"task_index", "task"}
  meta/modality.json    GR00T 식 그룹 정의 (아래)
  meta/stats.json       정규화 통계. 없으면 DatasetFactory._ensure_stats 가 만든다
```

**HF 의 `DexJoCo/DexJoCo-Datasets-LeRobot` 은 v3.0 이라 그대로 못 쓴다** — 100 에피소드가
`data/chunk-000/file-000.parquet` 하나에 합쳐져 있고 영상도 여러 에피소드가 한 mp4 에 연결,
메타는 `meta/episodes/*.parquet` + `meta/tasks.parquet`. 그래서 **raw repo
(`DexJoCo-Datasets-Raw`) 에서 직접 변환**한다. raw 는 에피소드당 디렉토리 하나
(`replay.zarr` + `videos/*.mp4`) 라 v2.1 로 옮기기가 오히려 쉽다.

### 5.2 우리 `modality.json`

```json
{
  "state": {
    "eef_position": {"start": 0, "end": 3,  "original_key": "observation.state"},
    "eef_rotation": {"start": 3, "end": 9,  "original_key": "observation.state"},
    "hand_joints":  {"start": 9, "end": 25, "original_key": "observation.state"}
  },
  "action": {
    "eef_position": {"start": 0, "end": 3,  "original_key": "action"},
    "eef_rotation": {"start": 3, "end": 9,  "original_key": "action"},
    "hand_joints":  {"start": 9, "end": 25, "original_key": "action"}
  },
  "video": {
    "camera_front": {"original_key": "observation.images.camera_front"},
    "camera_wrist": {"original_key": "observation.images.camera_wrist"}
  },
  "annotation": {"human.task_description": {"original_key": "task_index"}}
}
```

- `start/end` 는 raw parquet 컬럼에서 잘라낼 구간. loader 가
  `df[original_key].map(lambda x: x[start:end])` 로 쓴다.
- `annotation.human.task_description` → `task_index` 는 loader 가
  `tasks.jsonl` 로 문자열을 되찾는 indirection (openarm/rby1 데이터셋과 동일).
- 사본이 `modality/dexjoco_panda_allegro/modality.json` 에 있다 (`configs/exp/*.yaml` 의
  `modality:` 필드가 가리키는 자리).

**concat 순서의 정본은 `modality.json` 이 아니라 등록 config 다** (`rl/data.py` 주석).
우리는 두 순서를 일부러 일치시켰다 (eef_position → eef_rotation → hand_joints).

### 5.3 등록 config

`third_party/RLDX-1/rldx/configs/data/dexjoco_panda_allegro_config.py`

```python
ACTION_HORIZON = 32
GROUPS = ["eef_position", "eef_rotation", "hand_joints"]

video    delta_indices=[0]                modality_keys=["camera_front","camera_wrist"]
state    delta_indices=[0]                modality_keys=GROUPS
action   delta_indices=range(32)          modality_keys=GROUPS
         action_configs = [ABSOLUTE / NON_EEF / DEFAULT] * 3
language delta_indices=[0]                ["annotation.human.task_description"]

register_modality_config(..., EmbodimentTag.GENERAL_EMBODIMENT)
```

- `GENERAL_EMBODIMENT` = projector id 0, PT-IMG 가 사전학습한 warm slot. rby1 finetune 들과
  같은 슬롯.
- 세 그룹 모두 **`ABSOLUTE / NON_EEF / DEFAULT`**. `type`/`format` 은
  `rep == RELATIVE and use_relative_action` 일 때만 효과가 있어서 지금은 무의미한
  필드지만, `ActionType.EEF` 를 선언하면 나중에 relative 로 갈 때
  `stats.py::RelativeActionLoader` 가 EEF 경로에서 `NotImplementedError` 를 던진다
  (`stats.py:200`). NON_EEF 로 두면 relative 실험이 JointPose 경로(지원됨)로 간다.
- `--action-horizon 32` 는 `len(action.delta_indices)` 와 **같아야** 하고, 다르면
  `assembly.py` 가 즉시 assert 로 죽는다.
- 같은 태그에 두 config 를 등록하면 assert 로 죽으므로, 이 파일과
  `rby1_wujihand2_*` / `libero_*` 를 **같은 프로세스에서 같이 import 하면 안 된다**.
  `--modality-config-path` 로만 로드되므로(그 디렉토리의 `__init__.py` 는 비어 있다) 안전.

### 5.4 컨버터가 하는 일

`sim/dexjoco/convert_raw_to_rldx.py`

| 단계 | 내용 |
|---|---|
| 입력 | `replay.zarr/data/{action(T,23 quat), state(T,1,38), action_rotvec, timestamp}` + `videos/{front,wrist}.mp4` |
| shape 정규화 | `(T,1,D) → (T,D)` (레코더가 state 에 singleton 축을 넣는다) |
| 선행 정적 프레임 트림 | dexjoco 와 같은 규칙 (`find_first_non_static_frame`) → 공개 데이터셋과 에피소드 경계 일치 |
| **추가 1프레임 드롭** | 에피소드 첫 action 은 home → teleop 시작점 점프 (한 스텝에 최대 1.1 m). hammer_nail 100개 중 **8개는 첫 action 이 정확히 0** (env 는 0 pose 를 "hold" 로 읽고, quaternion 도 무효) |
| 회전 변환 | quat wxyz → 정규화 → `R.as_matrix()[:2,:].flatten()` = rot6d |
| state | `state[:, :23]` → 25 (privileged 23:38 버림) |
| action | `action[:, :23]` → 25 |
| 영상 | 640×640 → **256×256** 재인코딩 (libx264, yuv420p, crf 20, macro_block_size=1). 프레임 수를 parquet 길이와 **정확히** 맞추고 아니면 예외 |
| RL 컬럼 | `next.success` / `next.done` = 마지막 프레임만 True (raw 데모는 전부 성공), `next.truncated` = 전부 False |
| meta | `info.json` / `episodes.jsonl`(+ `raw_demo`, `raw_start_frame` 로 출처 추적) / `tasks.jsonl` / `modality.json` |

**256×256 인 이유**: 프로세서 기본값이 `image_max_area = 65536 (= 256²)`,
`image_resize_m = 32` (`processing_rldx.py:206`). 256×256 이면
`AspectAreaResizeAndCrop` 이 no-op 이 되어 리사이즈가 두 번 일어나지 않는다.
(참고: openarm/rby1 실데이터는 320×192 = 61440 으로 같은 이유로 고른 값.)

**카메라를 front + wrist 만 쓰는 이유**: raw 데모에는 ego_left/ego_right 스테레오 쌍도 있어서
rby1 egostereo config 와 모양이 맞지만, **policy_mode env 가 실제로 내보내는 키가 태스크마다
다르다** — water_plant 는 `_compute_observation` 이 wrist + front 만 만든다. ego 로 학습하면
서빙이 불가능해진다. front+wrist 는 dexjoco 의 `configs/rand_obj/*.yaml` 매핑과도 일치.

`next.success/next.done` 을 데모에도 심어두면 **데모 자체가 `rl/data.py` 가 읽는 expert
transition** 이 된다 (reward = next.success, mask = 1 − next.done). offline RL 에서
데모 + 롤아웃을 섞어 쓸 수 있다.

### 5.5 검증

`sim/dexjoco/test_format.py` — 합성 데이터셋을 컨버터와 **같은** `write_episode` /
`write_meta` / `_features` / `_modality` 로 쓰고, `generate_stats` + `LeRobotEpisodeLoader`
로 되읽는다. GPU·다운로드·sim venv 없이 도는 포맷 계약 테스트.

```
PYTHONPATH=third_party/RLDX-1:. NO_ALBUMENTATIONS_UPDATE=1 \
    third_party/RLDX-1/.venv/bin/python sim/dexjoco/test_format.py
```

실제 변환 결과도 같은 loader 로 확인했다:

```
episodes 3  lengths [201, 270, 236]  fps 30
cols: state.eef_position/eef_rotation/hand_joints, action.*, video.camera_front/camera_wrist,
      language.annotation.human.task_description
video frame (256,256,3) uint8   lang "Use the hammer to drive the nail into the wooden board."
stats action groups {'eef_position': 3, 'eef_rotation': 6, 'hand_joints': 16}
```

---

## 6. BC 학습

### 6.1 명령

`sbatch/dexjoco/bc_hammer_nail.sbatch` (8 GPU 1노드). 핵심 인자:

```
--base-model-path RLWRLD/RLDX-1-PT-IMG      --backbone-path RLWRLD/RLDX-1-VLM
--dataset-paths /workspace/junmo_cho/dexjoco/lerobot/hammer_nail_rand_obj
--embodiment-tag GENERAL_EMBODIMENT
--modality-config-path rldx/configs/data/dexjoco_panda_allegro_config.py
--action-horizon 32        --rtc-training-max-delay 8
--video-length 1           --n-cog-tokens 64
--global-batch-size 128    --learning-rate 1e-4     --max-steps 10000
--state-dropout-prob 0.3   --random-crop-fraction 0.95
--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
--save-only-model          --save-steps 2500
```

### 6.2 값을 그렇게 고른 이유

| 인자 | 근거 |
|---|---|
| `action_horizon 32` | 데이터 30 fps 기준 1.07 s. dexjoco 의 openpi baseline 이 chunk 30 / replan 0.8 로 돌았으므로 같은 시간 규모. 40 으로 올리면 에피소드가 짧아 effective step 이 더 줄어든다 (RLDX-1 은 에피소드 뒤쪽 `horizon-1` 프레임을 버린다) |
| `max_steps 10000` | effective step = 21,471 − 100×31 = **18,371** (변환 후 실측). 10000 × 128 / 18.4k ≈ **70 epoch** — cjl_0819_v9 rby1 런과 같은 epoch 예산 |
| `save_steps 2500` | 4개 중간 체크포인트. **BC 성공률을 학습량의 함수로 재서 최고점을 critic 단계의 고정 base policy 로 쓴다.** 단일 태스크 100 에피소드는 과적합 구간이 있을 수 있어 필수 |
| `rtc_training_max_delay 8` | 롤아웃에서 replan 경계보다 8 step 먼저 추론을 시작하는 RTC 를 쓸 수 있게. openarm(16/8/2)·rby1(40/15/12) 사이 값 |
| `state_dropout_prob 0.3` | rby1 real 런과 동일. sim 이라 state 노이즈는 적지만 critic 단계에서 state 의존을 줄여두는 편이 안전 |
| `global_batch_size 128`, `lr 1e-4` | 팀 real 런과 동일 |
| `color_jitter` | rand_obj 는 시각 랜덤화가 없으므로 약한 photometric aug 로 보완 |
| `GENERAL_EMBODIMENT` | PT-IMG 가 사전학습한 warm projector slot |

### 6.3 저장 위치 (⚠️ `/rlwrld2` 를 쓰지 않는다)

`/rlwrld2` 는 사용률 **100% / 여유 68 GB** 다 (`checkpoints/` 만 117 GB, RLDX-1 체크포인트
하나가 ~13 GB, `~/.cache/uv` 가 13 GB). 그래서:

| 무엇 | 어디 |
|---|---|
| raw + 변환 데이터셋 | `/workspace/junmo_cho/dexjoco/` |
| sim venv | `/workspace/junmo_cho/dexjoco/venv` (uv cache 도 `/workspace/junmo_cho/uv_cache`) |
| 학습용 HF 캐시 | `/workspace/junmo_cho/hf_cache` (PT-IMG 13 GB) |
| BC 체크포인트 | `/rlwrld-unified-checkpoints/junmo_cho/dexjoco/` |

다른 서버에서 학습한다면 옮겨야 할 것은 **변환된 데이터셋 디렉토리 하나**
(hammer_nail 약 200 MB 예상) + `dexjoco_panda_allegro_config.py` + sbatch 다.

### 6.4 필요한 것

- `HF_TOKEN` — `RLWRLD/RLDX-1-PT-IMG` / `RLDX-1-VLM` 이 private 이고 이 머신 HF 캐시에
  blob 이 없다. `~/.rldx_secrets.sh` 에 `HF_TOKEN` / `WANDB_API_KEY`.
- RLDX-1 체크아웃에 `dexjoco_panda_allegro_config.py` 가 있어야 한다 (지금 커밋 안 된 상태).

---

## 7. 추론 / 롤아웃 (아직 안 만듦 — 설계만)

real 쪽 구성은 **`rl/vla_rldx.py serve` (RLDX-1 + critic guided chunk 선택)** ↔ **로봇 위의
rrc** 다. sim 은 **클라이언트만** 갈아끼운다. RLDX-1 문서가 권하는 2-프로세스 구조
(`docs/evaluation.md`) 와 동일하다: 모델은 training venv 에서 GPU 를 잡고, sim 은 자기 venv
에서 돌고 둘이 ZeroMQ 로 대화한다 — mujoco 스택과 torch 2.7+flash-attn 핀이 충돌하기 때문.

```
  rl/vla_rldx.py serve            ZeroMQ + msgpack      sim/dexjoco/rollout_dexjoco.py
  (RLDX-1 venv, GPU)         <────────────────────>     (/workspace/.../venv, MUJOCO_GL=egl)
  RLDXVLA + EXPOLearner                                 dexjoco env + LeRobot v2.1 기록
  + cog-critic guidance
```

### 7.1 와이어 프로토콜

`rldx/policy/server_client.py`. REQ/REP + msgpack, ndarray 는 `.npy` 바이트로 실린다.
엔드포인트: `ping` / `get_action` / `reset` / `get_modality_config` / `kill`.
클라이언트 쪽 의존성은 **`zmq` + `msgpack` + `numpy` 뿐** — RLDX import 가 필요 없어서
sim venv 에 그대로 넣을 수 있다 (~40줄).

### 7.2 관측 포맷

`RLDXSimPolicyWrapper` (`rldx/policy/rldx_policy.py:253`) 가 받는 flat 키:

```
video.camera_front   uint8   (B, T, H, W, C)     T = video delta_indices 길이 = 1
video.camera_wrist   uint8   (B, T, H, W, C)
state.eef_position   float32 (B, T, 3)
state.eef_rotation   float32 (B, T, 6)
state.hand_joints    float32 (B, T, 16)
annotation.human.task_description  또는  task   → 문자열
```

즉 롤아웃 클라이언트가 할 일:

1. env obs → 640×640 을 **256×256 으로 리사이즈** (학습과 같은 전처리. 여기서 어긋나면
   critic 의 ens.std 가 튄다 — `vla_rldx.py --simulate-rrc` 주석이 real 에서 겪은 그 문제)
2. `state[:23]` = tcp_pose(7) + hand(16) → quat → rot6d → 25차원 → 그룹별로 쪼개 전송
3. 응답 action chunk (25차원 × 32) → `eef_position(3)` + `rot6d(6) → quat wxyz` +
   `hand(16)` = **(23,)** 으로 조립해 `env.step`
4. `replan_steps` 만큼 실행하고 다시 질의 (dexjoco 의 openpi 클라이언트처럼 겹치는 청크를
   시간축 보간 블렌딩할지는 선택 — RTC 를 쓰면 서버가 prefix 를 처리한다)
5. 에피소드를 **§5 와 같은 v2.1 포맷**으로 기록. `next.success` = env `info["succeed"]`,
   `next.done` = terminated. 이때 `write_episode(...)` 를 그대로 재사용한다
   (컨버터에서 이미 분리해 둠)

주의점:

- `ExpoServer` 의 `img_size` 기본값이 `(320,192)` 다 → sim 은 `256 256` 로 줘야 한다.
- rot6d → 회전행렬은 **Gram-Schmidt** 로 직교화해야 한다 (`_rot6d_to_matrix`, `pose.py:426`).
  모델 출력이 정확히 직교정규가 아니다.
- env 의 `_compute_observation` 이 **카메라 4개를 매 step 렌더**한다 (front, ego_left,
  ego_right, wrist). 2개만 쓰므로 monkeypatch 로 렌더를 줄이면 롤아웃이 대략 2배 빨라진다.
- `step()` 의 `time.sleep(1/hz - dt)` 제거 (§3).
- `MUJOCO_GL=egl` 은 GPU 필요. CPU 노드에서는 `osmesa` (동작 확인했지만 소프트웨어 렌더라
  느리다), `glfw` 는 display 없이 core dump.
- 실패 에피소드는 `env_step >= 1000` 까지 간다. 롤아웃 루프에서 `max_episode_steps` 를
  400 정도로 잘라 truncate 로 처리하는 편이 낫다 (데모 최대 길이 522 를 감안).
  이때 `next.truncated` 가 처음으로 1 이 되므로 `rl/data.py` 의 `mask = 1 − done` 가정을
  다시 봐야 한다 (그 파일 검사 8).

### 7.3 BC 성공률

= 롤아웃 에피소드 중 `next.success` 가 있는 비율. 논문 기준선 GR00T N1.5 67.3 ± 4.2
(50 trial × 3 seed). 우리는 seed 를 바꿔 100 에피소드 정도 돌리면 ±5%p 수준.

---

## 8. offline RL / critic

기존 코드 **수정 없이** 재사용한다. 입력은 §7 이 남긴 롤아웃 세션 디렉토리들의 부모 경로.

```
# 1) cognition token feature 캐시 (VLA 백본 1회 통과, frozen 표현)
python -m rl.extract_cogfeat --exp dexjoco_hammer_nail \
    --data <롤아웃 부모> --checkpoints <ckpt> --batch 64 --resume

# 2) IQL critic
python -m rl.offline_iql --exp dexjoco_hammer_nail \
    --data <롤아웃 부모> --checkpoints <ckpt> --features cogfeat.npy \
    --discount 0.995 --bins 128 --expectile 0.7 \
    --steps 30000 --holdout 0.1 --eval-every 5000 --video-eps 30

# 3) guidance 켜고 다시 롤아웃 → BC 성공률과 비교
```

새로 써야 하는 것은 `configs/exp/dexjoco_hammer_nail.yaml` 하나:

```yaml
name: dexjoco_hammer_nail
robot: panda_allegro                  # info.json 의 robot_type 과 교차검증
rldx_data_config: rldx/configs/data/dexjoco_panda_allegro_config.py
modality: modality/dexjoco_panda_allegro/modality.json
base_policy: <BC 체크포인트 상대경로>   # BC 끝나야 채울 수 있다
action_horizon: 32
replan_steps: 16                      # 롤아웃 클라이언트의 execution horizon 과 같아야 한다
inference_latency: 4                  # RTC prefix. action_horizon >= latency + replan 강제
explore_groups: [eef_position, eef_rotation]   # 후보 편집/guidance 를 걸 그룹
```

- 코드가 검사하는 제약: `action_horizon >= inference_latency + replan_steps`
  (32 ≥ 4 + 16 ✓), `replan_steps == 클라이언트 execution horizon`,
  `action_horizon == 모델 config 의 action_horizon`.
- `explore_groups` 를 뭘로 둘지가 실험 변수다. hammer 는 타격 속도가 삽입량을 정하므로
  `eef_position` 이 1순위, 손가락(`hand_joints`) 은 파악 안정성 쪽.
- `gamma_eff = discount ** replan_steps` 이므로 replan 16 이면 0.995¹⁶ ≈ 0.923.
  real(replan 8) 과 실효 할인이 달라진다는 점 주의.
- 검증 도구도 그대로 쓸 수 있다: `python -m rl.vla_rldx verify-cog` 가 학습 때의
  latent/Q 와 서빙 경로의 값을 프레임 단위로 대조한다 (로봇/sim 불필요).

**왜 이 태스크가 offline RL 검증에 맞는가** (다시, offline 관점에서):

- 데이터가 **on-policy 롤아웃 + 성공/실패 라벨** 로 구성된다. IQL 의 expectile 회귀는
  같은 상태에서 좋은 액션과 나쁜 액션이 모두 데이터에 있어야 V 가 의미를 갖는데,
  67% 성공률의 롤아웃이 정확히 그 조건이다.
- 보상이 **sparse terminal** 이라 real(fuji/openarm) 과 같은 형태다. 즉 sim 에서 얻은
  결론이 real 로 이전된다. reward shaping 을 넣으면 그 이전성이 깨지므로 `nail_depth` 는
  진단용으로만 쓴다.
- 짧은 horizon 이라 `discount` / `replan_steps` 스윕을 여러 번 돌릴 수 있다. real 에서는
  한 설정을 재는 데 며칠이 걸린다.
- 실패 모드가 **시각적으로 구분된다** (망치를 못 잡음 / 잡았지만 못을 못 맞춤 / 맞췄지만
  약함). critic 이 무엇을 학습했는지 비디오 오버레이로 읽을 수 있다
  (`offline_iql.py` 가 Q/V/A 커서를 올린 mp4 를 뽑는다).

---

## 9. 현재 상태

| 항목 | 상태 |
|---|---|
| dexjoco raw 다운로드 | ✅ hammer_nail 100ep / 242 MB, water_plant 100ep — `/workspace/junmo_cho/dexjoco/raw/` |
| sim venv | ✅ `/workspace/junmo_cho/dexjoco/venv` (mujoco 3.4.0, zarr 2.18.7, numpy 1.26.4, cv2, imageio, zmq, msgpack) |
| env 헤드리스 동작 | ✅ `MUJOCO_GL=osmesa` 로 reset 확인 (state (38,), 4 카메라 640×640) |
| 컨버터 | ✅ `sim/dexjoco/convert_raw_to_rldx.py` |
| 포맷 계약 테스트 | ✅ `sim/dexjoco/test_format.py` — `LeRobotEpisodeLoader` + `generate_stats` 통과 |
| 등록 config | ✅ `rldx/configs/data/dexjoco_panda_allegro_config.py` (RLDX-1 체크아웃, 미커밋) |
| modality.json 사본 | ✅ `modality/dexjoco_panda_allegro/modality.json` |
| hammer_nail 변환 | ✅ 100 ep / 21,471 frame / 44 MB (len min 148 / mean 214.7 / max 521), `stats.json` |
| 학습 데이터 경로 검증 | ✅ `StandardSingleStepDataset` 이 **18,371 effective step / 100 ep** 으로 인덱싱 (= 21,471 − 100×31, horizon 32). `__getitem__` 은 processor 가 필요해 GPU 서버에서 확인 |
| water_plant 변환 (백업 태스크) | ✅ 100 ep / 27,645 frame / 46 MB (len min 202 / mean 276.4 / max 393) |
| BC sbatch | ✅ `sbatch/dexjoco/bc_hammer_nail.sbatch` — `HF_TOKEN` 필요 |
| 롤아웃 클라이언트 | ⬜ 미착수 (§7 설계만) |
| `configs/exp/dexjoco_hammer_nail.yaml` | ⬜ BC 체크포인트 경로 나온 뒤 |

### 열려 있는 결정

1. **§4.3 state 표현** — A(EEF, 구현됨) / C(joint state + EEF action, replay 필요).
2. **replan_steps / inference_latency** — 16/4 를 제안. real openarm 은 8/2, rby1 은 15/12.
   sim 은 추론 지연이 실제 물리 시간과 무관하므로 "얼마를 쓸 것인가" 가 순수 실험 변수다.
3. **롤아웃 max_episode_steps** — 400 제안. `next.truncated` 가 처음 1 이 되므로 §7.2 주의.
4. **guidance 실험 대상 그룹** (`explore_groups`).
