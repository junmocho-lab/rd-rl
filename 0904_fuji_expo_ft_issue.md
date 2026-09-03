# 0904 fuji EXPO-FT 온라인 RL — 이슈 로그

첫 온라인 라운드(fuji_online_1, 2026-09-03~04) 브링업에서 겪은 문제들. 증상 → 원인 → 조치 →
교훈 순. 심각한 순서가 아니라 시간 순이다.

성공률 궤적 (맥락):

```
base(참조, edit 없음)   90%   (t0_rc 20ep 중 18)
round 0  θ₀+edit        25%   (20ep) — edit 노출 50% × 에피소드당 ~40쿼리 복리
round 1  r000+edit      62.5% (8ep)  — critic/residual 1라운드 학습 후
round 2  r001+edit      37.5% (8ep)  — 로봇 손 이슈가 세션에 겹쳤을 가능성 (아래 7)
```

---

## 1. cogfeat 4-GPU 병렬 추출이 9 fps 로 기어감 (OMP 과다구독)

- **증상**: H200 4장인데 합계 ~9 fps (예상의 1/20). GPU util 0%, python 프로세스들이
  CPU 1600%+ 점유.
- **원인**: torch 가 프로세스마다 OMP 스레드를 코어 수(128)만큼 생성 → 4 프로세스
  × 128 = 512 스레드가 128코어에서 스핀 경쟁.
- **조치**: `OMP_NUM_THREADS=16` (+MKL/OPENBLAS) 걸고 재시작 → **213 fps (23배)**.
  `rl/extract_cogfeat.py --shard` 도움말에 명시.
- **교훈**: torchrun 은 OMP=1 을 자동으로 잡아주지만 **수동 멀티프로세스 GPU 작업은
  직접 걸어야 한다**. `코어수/프로세스수` 가 기준.

## 2. 추출 중 images.mm 조기 삭제 + 중복 프로세스 기동

- **증상**: merge 전에 `rm images.mm` 실행 + 기존 4개가 도는 채로 shard 4개 재기동.
- **결과**: 신규 4개는 images 가드에서 즉사(무해), 기존 4개는 삭제된 inode 를 fd 로
  계속 읽어 **전부 완주** — 손실 0. merge 의 part 검증(done/T 대조)이 최종 안전망.
- **교훈**: `rm images.mm` 은 **merge 성공 후**. 재기동 전 `jobs`/`ps` 로 기존 프로세스
  확인. shard 가드·merge 검증이 설계대로 사고를 막았다.

## 3. ★ actor LoRA lr 3e-4 가 정책을 파괴 (이번 브링업의 핵심 이슈)

- **증상**: r000 theta 서빙에서 성공률 붕괴, edit 을 꺼도(probe) 실패 지속.
  "피더를 잡으러 갔다가 안 잡고 손이 엉뚱한 곳으로".
- **진단**: r000 LoRA 의 가중치 델타 ‖ΔW‖ 합계 24.8 (60 BC 스텝 만에),
  행동 변화 |Δa| **평균 0.038/dim, 최대 1.16** (base 자연 산포 0.018 의 2배+).
  그런데 actor_loss 는 0.006 으로 바닥.
- **원인(기전)**: 학습 데이터가 자기 성공 롤아웃이라 **배울 신호가 없음** →
  gradient ≈ 노이즈 → **Adam 은 gradient 크기와 무관하게 스텝을 lr 로 정규화** →
  zero-init LoRA 가 노이즈 방향으로 lr×스텝수만큼 순수 표류. EXPO-FT 원본의 3e-4 는
  대규모 연속학습 기준이라 소규모 라운드 체제에 과함.
- **조치**: `vla.lora_lr: 0.0` (actor 동결). r000/theta.pt 에서 lora 키 제거
  (원본은 theta_with_lora.bak.pt). actor 재활성은 **자동/예정이 아니라 별도 결정 사항**
  (아예 안 켜고 test-time 개선만으로 가는 선택지 포함 — VLA 거동 리스크 원천 차단
  vs 개선 상한 제한의 트레이드오프). 켠다면 최소 조건: candidate_q_std > 0.001 +
  성공 에피소드에 edit 실행분 축적, 그때 1e-5 부터.
- **교훈**: loss 가 작다고 무해한 게 아니다 — **Adam + zero-init 어댑터 + 신호 없는
  데이터 = 보장된 표류**. 어댑터 학습은 "배울 것이 생긴 뒤에" 켠다.

## 4. 잘못된 데이터셋을 round 1 로 전송 → 버퍼 롤백

- **증상**: 의도와 다른 14ep 세션을 r001 로 전송, learner 가 물어서 ingest 진행
  (cogfeat 32k/32.6k 시점에 발견).
- **조치**: learner 즉시 kill → 매니페스트(sessions.json)에서 마지막(잘못된) 세션
  제거 + cogfeat done 카운터를 신뢰 구간(24,531)으로 리셋 + actnorm 캐시 삭제 +
  r001 메일박스 삭제. **학습은 시작 전이라 오염 0.** 부작용은 다음 ingest 의
  images.mm 전체 재생성(~5분, 세션 목록 불일치 시 자동 안전 동작)뿐.
- **교훈**: 버퍼는 append-only 라 **꼬리에 붙은 직후가 되돌릴 수 있는 유일한 시점**.
  전송 전 검증 프로토콜(스캔 + next.success 자동 집계)을 거치면 애초에 안 생긴다.
  learner 처리 중 재전송 금지 규칙도 재확인.

## 5. tmux kill 후 구 learner 잔존 → 같은 run id 로 2개 가동

- **증상**: `tmux kill-session` 후 start.sh 재실행 시 `[warn] 이미 도는 learner` —
  구 torchrun(+rank 4)이 살아 있었다.
- **위험**: 같은 run id 두 learner = 메일박스 경쟁 + 구 코드/구 θ(나쁜 LoRA 포함
  메모리 상태)로 라운드를 채갈 수 있음.
- **조치**: `kill -9 <torchrun pid>` 로 트리 정리 후 확인.
- **교훈**: start.sh 의 [warn] 은 무시하지 말 것. 재시작 절차 =
  kill-session → **ps 로 잔존 확인** → start.sh.

## 6. wandb 미부착 + 재시작 시 step 유실

- **증상**: (a) 셸에 WANDB_API_KEY 가 없으면 조용히 로그파일만 남김 — r001 라운드가
  wandb 에 공백. (b) 재시작한 learner 가 step 0 부터 다시 세서, resume 된 run 의
  최대 step(60) 이하 포인트를 wandb 가 조용히 버림.
- **조치(learner/loop.py)**: ① `~/.netrc` 자동 인식 (export 불필요) ② resume 시
  `wandb.run.step` 에서 이어 세기 (`[wandb] 재시작 이어붙기 — step N 부터`).
  r002 부터 정상 작동 확인.
- **교훈**: "조용히 안 붙는" 로깅은 사고 후에야 보인다 — 기동 로그에서
  `[wandb] ... → URL` 을 라운드 시작마다 확인.

## 7. 로봇 손(그리퍼) 하드웨어 이슈 — 소프트웨어 3중 오인 수사

- **증상**: r001 서빙에서 "완전히 이상한 행동" (잡으러 갔다 안 잡고 손이 위로).
- **수사 과정** (전부 무죄로 판명): ① r001 LoRA — 델타 정확히 0 검증 ②
  edit policy — bias 0.0016 으로 r000 보다 온순 ③ LoRA 주입 코드 경로 —
  lora 제거판(theta_nolora)으로도 재현 ④ **r000(전날 검증본)도 이상** ⑤
  **순정 base 서버도 이상** → 소프트웨어 전체 배제.
- **실원인**: 로봇 손 하드웨어 문제. 수리 후 정상.
- **교훈 — 행동 이상 시 격리 순서**: 서버 텔레메트리(qstd/std 배수/지연)가
  정상인데 로봇만 이상하면 **하드웨어(그리퍼·카메라·캘리브레이션)를 먼저** 볼 것.
  소프트웨어 격리는 theta A/B → edit-off probe → base 순. "무조건 되는 base" 가
  안 되는 순간 소프트웨어 수사는 중단.

## 8. (인지 사항) state_dropout 0.3 이 추론에도 적용됨

- hilw1-**dp5** 모델은 `state_dropout_prob: 0.3` 인데, RLDX 의 해당 코드에
  `self.training` 게이트가 없어 **서빙에서도 매 포워드 30% 확률로 관절 상태가
  마스킹**된다 (`rldx/model/core/rldx.py:348` — 바로 아래 additive noise 는
  게이트가 있음). base 가 90% 를 내는 걸 보면 그렇게 훈련돼 강건한 듯하나,
  이상 거동 조사 시 배경 소음이 될 수 있어 기록해 둔다. 후보 N개 확장 시
  후보별로 독립 적용된다는 점도 유의.

---

## 운영 개선으로 이어진 것

- 업로드 전 자동 검증: 해상도 320x192 + modality.json (`scan_sessions`, actor/learner 공용)
- 라운드 프로토콜: 성공 수는 데이터의 next.success 에서 자동 집계, collected-by 는
  서빙 artifacts 추적으로 자동 결정 (수기 실수 원천 차단)
- 라운드 크기 8ep (환경 4종×2) / updates_per_episode 3 유지
- 지표 워치리스트와 튜닝 기준은 METRICS.md (qvgm 보정 포함)
