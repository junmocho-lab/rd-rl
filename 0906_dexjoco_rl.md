# 0906 dexjoco sim 온라인 RL — 실행 치트시트

fuji 온라인 RL(실기)이 학습에 실패한 뒤, 분석이 쉬운 sim 에서 EXPO-FT 온라인 RL 이 실제로
도는지 검증하는 실험. 구조 설명은 `configs/exp/dexjoco_hammer_nail_d5r20_online.yaml` 머리말과
`sim/dexjoco/online_driver.py` 머리말에 있다. 이 문서는 **명령어 모음**이다.

한 노드(gpu:2)에서 세 국면이 교대한다:
서버 x2 병렬 롤아웃 → 세션을 라운드 메일박스로 rename → learner 2-rank DDP 학습 →
theta_live.pt 원자 교체(서버가 에피소드 경계에서 핫리로드) → 반복. round 0 = 10ep,
이후 2ep/라운드, 총 100ep. 시드 = 태스크별 BC 데모 성공 10개.

---

## 0. 준비 (클러스터, 레포 루트에서 1회)

```bash
git pull && git submodule update --init third_party/RLDX-1 third_party/expo-ft
wandb login                      # ~/.netrc 저장 (없으면 로그 파일만 남고 wandb 는 조용히 생략)
mkdir -p out
ls checkpoints/dexjoco/          # ★ 신규 태스크 base_policy 실명 확인 (아래 §5)
export MODEL_OUTPUT_DIR=/fsx/rlwrld-unified-checkpoints/$USER/rd-rl   # 제출 플러그인용
```

## 1. 기본 실행

```bash
sbatch --export=ALL,EXP=dexjoco_hammer_nail_d5r20_online sbatch/dexjoco/online/run.sbatch
```

- `RUN` 생략 시 `<EXP>_<월일_시분>` 자동 (예: `dexjoco_hammer_nail_d5r20_online_0906_1430`)
- **이어받기(선점/타임아웃 후)는 기존 RUN 을 명시**: `--export=ALL,RUN=<기존>,EXP=<같은 yaml>`
  — DONE 라운드 건너뛰기 / 반쯤 구른 롤아웃 --resume / wandb step 이어붙기 전부 자동
- `TASK` 는 EXP 이름에서 자동 유도 (`dexjoco_<task>_d5r20_online*`)

## 2. 옵션 매트릭스 — 전부 EXP(yaml) 선택으로 갈린다

### 태스크 x 액션 공간 (yaml 8개)

| | eef 만 (9관절; critic 225d / edit 180d) | 전체 액션 (25관절; 625d / 500d) |
|---|---|---|
| hammer_nail | `dexjoco_hammer_nail_d5r20_online` | `..._online_fullact` |
| water_plant | `dexjoco_water_plant_d5r20_online` | `..._online_fullact` |
| click_mouse | `dexjoco_click_mouse_d5r20_online` | `..._online_fullact` |
| fold_glasses | `dexjoco_fold_glasses_d5r20_online` | `..._online_fullact` |

차이는 `explore_groups` 블록 하나 — critic/edit 차원·target_entropy 는 자동 유도.
fullact 는 d5r20 오프라인 실측(625차원 critic 의 후보 Qstd 0.0001)의 온라인 재검 대조군.

태스크별 자동 반영값 (BC 데모 길이 실측 기반):

| task | max_episode_steps | updates/ep | discount (시작 V) |
|---|---|---|---|
| hammer_nail | 400 | 10 | 0.995 (0.34) |
| water_plant | 400 | 10 | 0.995 (0.25) |
| click_mouse | 600 | 15 | 0.995 (0.20) |
| fold_glasses | 800 | 20 | 0.9975 (0.26) |

### critic: 분포형 vs 스칼라 (원본 EXPO-FT parity)

```bash
# HL-Gauss 분포형 bins 128 (기본 — support [0,1] 이 발산 상한)
EXP=dexjoco_hammer_nail_d5r20_online
# 스칼라 MSE Q (원본 EXPO-FT 와 동일 — REDQ min/보상 [0,1]/LayerNorm 만으로 방어)
EXP=dexjoco_hammer_nail_d5r20_online_scalarq
```

스위치는 yaml 의 `critic.bins: 128 → 0` 한 줄이다. 다른 태스크의 scalarq 판이 필요하면:

```bash
sed 's/^name: \(.*\)$/name: \1_scalarq/; s/^  bins: 128/  bins: 0/' \
  configs/exp/dexjoco_water_plant_d5r20_online.yaml \
  > configs/exp/dexjoco_water_plant_d5r20_online_scalarq.yaml
```

주의: 스칼라는 발산 상한이 없다 — learner 로그의 `q_max > 1.2` 경고가 뜨면 분포형으로 복귀.

### edit_scale (yaml `expo.edit_scale`, 기본 0.2 = 원본값)

- 0.2 는 dexjoco 자연 후보 산포(0.019)의 ~10배 — round 0 성공률이 base(~50%) 아래로
  떨어지는 것은 예상이자 필요 비용 (대비 신호). 판정은 "떨어졌나"가 아니라 **회복하나**.
- round 0 성공이 0~1/10 이면: `edit_scale: 0.05` 로 낮춰 **새 RUN**.
- 2~3라운드에 회복 조짐이 없고 `edit선택` 이 0.5 에서 안 내려가면 critic 이 edit 을
  못 거르는 것 — 같은 처방.

### 서빙 모드 (GPU 메모리)

```bash
SERVE_MODE=resident   # 기본: 서버 상주 + θ 핫리로드 (learner 와 메모리 공존)
SERVE_MODE=restart    # 학습 OOM 시: 라운드마다 서버 on/off — 학습이 GPU 전체를 쓴다
                      # (서버 로드는 클라이언트의 300s 대기와 겹쳐 라운드당 ~1분)
```

learner OOM 의 다른 손잡이: yaml `expo.batch_size: 64 → 32` (실효 128 → 64 = 원본 parity).

## 3. 모니터링

```bash
squeue -u $USER
tail -f out/dexjoco-online_<jobid>.out          # learner (라운드 학습, [N/M] 진행)
tail -f runs/<RUN>.driver.log                   # driver (롤아웃/READY/theta 게시/곡선)
tail -f runs/<RUN>.server0.log                  # 서버0 ([reload] = θ 핫리로드 확인)
```

산출물 지도:

```
runs/<RUN>/
  run_meta.json                 무슨 yaml/태스크/코드로 돌았나 + 재시작 이력
  train_success_curve.png       x=sim 분, 최근 20ep 러닝 성공률 (라운드마다 갱신)
  plots/step*_r*_q.png          라운드 새 에피소드 Q 곡선 (성공 초록/실패 빨강)
  plots/step*_r*_q_all.png      전체 에피소드를 그 시점 critic 으로 재평가 (스텝별 비교용)
  rNNN/dataset/<세션>/videos/   에피소드별 mp4 (plots 라벨의 ep 인덱스와 짝)
  rNNN/READY                    에피소드별 성공/프레임 (json)
  buffer/                       images.mm / cogfeat.npy / actnorm.npy (재생성 가능 캐시)
checkpoints/expo/<RUN>/rNNN/{theta.pt,meta.json,DONE}
```

wandb (`rd-rl-expo/<RUN>`): `rollout/success`(에피소드 1/0 — **그 라운드 학습 스텝 위에
에피소드 경계마다 분산 기록**되므로 라운드 학습이 끝나야 그 라운드 몫이 다 보인다),
`rollout/episode_seconds`, `rollout/sim_seconds·minutes·hours`(누적),
`train/critic_loss·q·q_max·candidate_q_std·select_ratio_with_residual`, `round/*`, `buffer/*`.
차트 x축을 `rollout/sim_seconds` 나 `rollout/online_episode` 로 바꾸면 시간축 성공률 뷰.

## 4. 판정 기준 (fuji 교훈 요약)

- **성공 신호**: 성공률 회복→상승, `candidate_q_std` 가 1e-4 바닥에서 기상,
  `_q_all.png` 에서 성공/실패 곡선의 분기가 라운드마다 앞당겨짐
- **경보**: q_max > 1.2 (스칼라 판), critic_loss 지속 상승, `edit선택` 이 0.5 고정
  (critic 이 edit 을 못 거름), 성공 0 지속 (신호 소멸 — edit_scale 인하)
- LoRA(actor)는 AdamW(b2 0.95, wd 1e-10)+clip 1.0+lr 2.5e-5 — fuji r000 파괴의 재발 방지.
  `train/actor_grad_norm` 이 상시 clip(1.0)에 걸려 있으면 들여다볼 것

## 5. 남은 TODO

- 신규 3태스크 yaml 의 `base_policy` 가 hammer_nail 패턴 **추정치**다 —
  `ls checkpoints/dexjoco/` 실명으로 교체 후 실행 (BC 모델이 없으면 그것부터)
- HL-Gauss bin 해상도 vs 초미세 edit 민감도 — 오프라인 격자 실험 설계는 메모리/이 문서
  이력 참조 (`--bins {128,256,512}` + `rl/probe_actsens.py`)
