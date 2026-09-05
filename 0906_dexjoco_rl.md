# 0906 dexjoco sim 온라인 RL — 명령어 전집

복붙용. 배경 설명은 yaml 머리말과 `sim/dexjoco/online_driver.py` 머리말 참조.
모든 명령은 **클러스터의 레포 루트**에서.

---

## 0. 준비 (1회)

```bash
git pull
git submodule update --init third_party/RLDX-1 third_party/expo-ft
wandb login                                                        # ~/.netrc 저장
mkdir -p out
ls checkpoints/dexjoco/            # ★ 신규 3태스크 base_policy 실명 확인 — 다르면 yaml 수정
export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl
```

`MODEL_OUTPUT_DIR` 는 제출 셸마다 export 돼 있어야 한다 (`--export=ALL` 이 실어 나른다).

---

## 1. 실행 명령 — 16개 조합 전부

형식: 태스크 4 x 액션공간 2 (eef | fullact) x critic 2 (분포형 | scalarq).
RUN 은 재시도마다 끝 숫자를 올릴 것 (`_1` → `_2`).

### hammer_nail

```bash
# eef + 분포형(기본)
sbatch --export=ALL,RUN=hammer_eef_1,EXP=dexjoco_hammer_nail_d5r20_online sbatch/dexjoco/online/run.sbatch
# eef + 스칼라 Q (원본 EXPO-FT parity)
sbatch --export=ALL,RUN=hammer_eef_sq_1,EXP=dexjoco_hammer_nail_d5r20_online_scalarq sbatch/dexjoco/online/run.sbatch
# 전체 액션(손가락 포함) + 분포형
sbatch --export=ALL,RUN=hammer_full_1,EXP=dexjoco_hammer_nail_d5r20_online_fullact sbatch/dexjoco/online/run.sbatch
# 전체 액션 + 스칼라 Q
sbatch --export=ALL,RUN=hammer_full_sq_1,EXP=dexjoco_hammer_nail_d5r20_online_fullact_scalarq sbatch/dexjoco/online/run.sbatch
```

### water_plant

```bash
sbatch --export=ALL,RUN=water_eef_1,EXP=dexjoco_water_plant_d5r20_online sbatch/dexjoco/online/run.sbatch
sbatch --export=ALL,RUN=water_eef_sq_1,EXP=dexjoco_water_plant_d5r20_online_scalarq sbatch/dexjoco/online/run.sbatch
sbatch --export=ALL,RUN=water_full_1,EXP=dexjoco_water_plant_d5r20_online_fullact sbatch/dexjoco/online/run.sbatch
sbatch --export=ALL,RUN=water_full_sq_1,EXP=dexjoco_water_plant_d5r20_online_fullact_scalarq sbatch/dexjoco/online/run.sbatch
```

### click_mouse

```bash
sbatch --export=ALL,RUN=click_eef_1,EXP=dexjoco_click_mouse_d5r20_online sbatch/dexjoco/online/run.sbatch
sbatch --export=ALL,RUN=click_eef_sq_1,EXP=dexjoco_click_mouse_d5r20_online_scalarq sbatch/dexjoco/online/run.sbatch
sbatch --export=ALL,RUN=click_full_1,EXP=dexjoco_click_mouse_d5r20_online_fullact sbatch/dexjoco/online/run.sbatch
sbatch --export=ALL,RUN=click_full_sq_1,EXP=dexjoco_click_mouse_d5r20_online_fullact_scalarq sbatch/dexjoco/online/run.sbatch
```

### fold_glasses

```bash
sbatch --export=ALL,RUN=fold_eef_1,EXP=dexjoco_fold_glasses_d5r20_online sbatch/dexjoco/online/run.sbatch
sbatch --export=ALL,RUN=fold_eef_sq_1,EXP=dexjoco_fold_glasses_d5r20_online_scalarq sbatch/dexjoco/online/run.sbatch
sbatch --export=ALL,RUN=fold_full_1,EXP=dexjoco_fold_glasses_d5r20_online_fullact sbatch/dexjoco/online/run.sbatch
sbatch --export=ALL,RUN=fold_full_sq_1,EXP=dexjoco_fold_glasses_d5r20_online_fullact_scalarq sbatch/dexjoco/online/run.sbatch
```

각 yaml 이 자동으로 들고 가는 값:

| task | max_episode_steps | updates/ep | discount | eef critic/edit | fullact critic/edit |
|---|---|---|---|---|---|
| hammer_nail | 400 | 10 | 0.995 | 225d / 180d | 625d / 500d |
| water_plant | 400 | 10 | 0.995 | 225d / 180d | 625d / 500d |
| click_mouse | 600 | 15 | 0.995 | 225d / 180d | 625d / 500d |
| fold_glasses | 800 | 20 | 0.9975 | 225d / 180d | 625d / 500d |

공통: N 8 + edit 8 후보, edit_scale 0.2, REDQ 10앙상블 min-of-2, utd 20,
round0 10ep → 이후 2ep/라운드 → 총 100ep, 시드 = BC 데모 성공 10ep.

---

## 2. 이어받기 (선점/타임아웃/중단 후)

같은 RUN + 같은 EXP 로 재제출만 하면 된다 (DONE 라운드 건너뜀, 반쯤 구른 롤아웃 --resume,
wandb step 이어붙기 자동):

```bash
sbatch --export=ALL,RUN=hammer_eef_1,EXP=dexjoco_hammer_nail_d5r20_online sbatch/dexjoco/online/run.sbatch
```

---

## 3. 변형 노브 — 풀 명령

### edit_scale 낮추기 (round 0 성공 0~1/10 이거나 2~3라운드 무회복일 때)

```bash
# 0.05 판 yaml 생성 (예: hammer eef) 후 새 RUN 으로
sed 's/^name: \(.*\)$/name: \1_es005/; s/^  edit_scale: 0.2 .*/  edit_scale: 0.05/' \
  configs/exp/dexjoco_hammer_nail_d5r20_online.yaml \
  > configs/exp/dexjoco_hammer_nail_d5r20_online_es005.yaml
sbatch --export=ALL,RUN=hammer_eef_es005_1,EXP=dexjoco_hammer_nail_d5r20_online_es005 sbatch/dexjoco/online/run.sbatch
```

### 학습 OOM 대응 두 가지 (아무 조합에나 덧붙임)

```bash
# 1) 서버를 라운드마다 켜고 끄기 — 학습이 GPU 메모리 전체를 쓴다 (라운드당 ~1분 로드)
sbatch --export=ALL,RUN=hammer_eef_1,EXP=dexjoco_hammer_nail_d5r20_online,SERVE_MODE=restart sbatch/dexjoco/online/run.sbatch

# 2) 배치 축소 (실효 128 → 64 = 원본 parity): yaml 의 batch_size 를 32 로
sed -i 's/^  batch_size: 64$/  batch_size: 32/' configs/exp/dexjoco_hammer_nail_d5r20_online.yaml
```

### 기타 env 노브 (전부 --export 에 콤마로 추가)

```bash
SERVE_MODE=resident|restart    # 기본 resident
PORT=22000                     # zmq 포트 (서버 i = PORT+i). 기본 jobid 기반 자동
TASK=hammer_nail               # 기본은 EXP 이름에서 자동 유도 — 보통 만질 일 없음
SIM_PY=/workspace/junmo_cho/dexjoco/venv/bin/python   # dexjoco venv 경로가 다르면
```

---

## 4. 모니터링 — 풀 명령 (RUN=hammer_eef_1 예시, 이름만 바꿔 쓰기)

```bash
squeue -u $USER                                                    # 잡 상태
tail -f out/dexjoco-online_<jobid>.out                             # learner 로그
tail -f runs/hammer_eef_1.driver.log                               # 라운드 진행/곡선 갱신
tail -f runs/hammer_eef_1.server0.log                              # 서버0 ([reload] 확인)
tail -f runs/hammer_eef_1.server1.log                              # 서버1
watch -n30 'ls -la runs/hammer_eef_1/train_success_curve.png runs/hammer_eef_1/plots/ | tail'
cat runs/hammer_eef_1/run_meta.json                                # 무슨 실험이었나
cat runs/hammer_eef_1/r000/READY | python3 -m json.tool | head -20 # 라운드 에피소드별 성공
nvidia-smi                                                          # 메모리 공존 확인 (노드에서)
```

에피소드 영상 (플롯 라벨 epN ↔ episode_00000N.mp4):

```bash
ls runs/hammer_eef_1/r000/dataset/rollout_r000_a/videos/chunk-000/observation.images.camera_front/
```

wandb: https://wandb.ai/junmokane/rd-rl-expo → run 이름 = RUN.
`rollout/success` 는 그 라운드 **학습 스텝 위에 에피소드 경계마다** 찍힌다 (라운드 학습이
끝나야 그 라운드 몫이 다 보임). 차트 x축을 `rollout/sim_seconds` 로 바꾸면 시간축 성공률.

---

## 5. 판정 기준 (fuji 교훈)

- **성공 신호**: 성공률이 초기 하락 후 20~30ep 내 회복 기울기, `train/candidate_q_std` 가
  1e-4 바닥에서 기상, `plots/step*_q_all.png` 에서 성공/실패 분기가 라운드마다 앞당겨짐
- **경보**: `q_max > 1.2` (scalarq 판 — 분포형 128 로 복귀), critic_loss 지속 상승,
  `edit선택` 0.5 고정 (critic 이 edit 을 못 거름), 성공 0 지속 (edit_scale 인하)
- scalarq 는 "분포형만 제거"다 — critic 입력은 여전히 cog feature (완전 원본은 이미지
  ResNet: yaml 에서 critic: 블록 제거 시 legacy 경로가 있으나 미검증)

## 6. 남은 TODO

- 신규 3태스크 `base_policy` 실명 교체 (`ls checkpoints/dexjoco/` 대조) — scalarq/fullact
  변형 yaml 도 같은 줄을 갖고 있으니 같이:
  ```bash
  sed -i 's|dexjoco/dexjoco_water_plant_randobj_.*|dexjoco/<실제 디렉토리명>|' configs/exp/dexjoco_water_plant_d5r20_online*.yaml
  ```
- HL-Gauss bin 해상도 vs 초미세 edit 민감도 오프라인 격자 (`--bins {128,256,512}` +
  `rl/probe_actsens.py`) — fuji 복귀 시점에
