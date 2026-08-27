#!/usr/bin/env python3
"""오프라인 policy extraction — PA-RL distillation 과 Q-VGM value-gradient matching.

두 방법 모두 **RLDX-1 action expert 의 LoRA 만** 학습한다 (VLM 백본 완전 동결).
RLDX-1 코드는 건드리지 않는다.

--method parl   PA-RL (arXiv 2412.06685)
    critic 이 개선한 액션을 **BC 라벨로** 넣는다. 손실은 RLDX-1 의 원래 flow matching
    그대로 (rldx.py:438) — 학습 RTC prefix 샘플링·action_mask 전부 원본 경로를 탄다.
    타깃은 rl/relabel_parl.py 가 만든 parl_actions.npy (T, action_dim) 다.
    학습 파라미터: action expert LoRA (r=16, alpha=32) 뿐.

--method edit   EXPO-FT 식 residual(edit) policy — **LoRA 를 쓰지 않는다**
    base policy 를 영구히 동결하고 그 출력에 tanh-Gaussian 보정을 얹어 SAC 로 학습한다:
      a = a_base + edit_scale * tanh(...)
      L = E[ alpha * log_prob - Q(s, a_base + edit) ],  alpha 는 목표 엔트로피로 자동 조절
    base 가 고정이므로 **base 청크를 한 번 캐시**하면 학습 루프에 VLA 가 들어가지 않는다
    (0.5M 파라미터 MLP + 캐시 텐서). 30,000 스텝이 수십 초다.
    SAC 업데이트는 rl/expo.py:313 update_residual_actor 와 같은 식이다.
    학습 파라미터: ResidualActor (~0.5M) + Temperature (1개). action expert 는 건드리지 않는다.

--method qvgm   Q-VGM (arXiv 2606.08015)
    critic 이 개선한 액션을 **velocity 타깃으로** 환산해 velocity field 를 직접 가르친다.
      Â_base = x[k] + (1-τ_k)·v_base(x[k],τ_k)            look-forward (Eq. 5)
      Â_Q    = keep-best( J번 ∇_A Q 상승 )                (Eq. 7)
      ĥ_Q    = (Â_Q - Â_base) / (1-τ_k)                    (Eq. 8)
      L      = Σ_k m_k ‖(v_θ - v_base) - sg[ĥ_Q]‖²         (Eq. 9)
    후반 M 스텝만 (m_k). 디노이징 체인을 타고 미분하지 않는다 (x[k] 는 stop-gradient).
    v_base 는 **LoRA adapter 를 끈 같은 가중치** — 사본을 들고 있지 않다.
    학습 파라미터: action expert LoRA 뿐. residual/edit policy 는 쓰지 않는다.

편집 마스크는 두 방법 모두 같다: explore_groups × 청크 스텝 [latency, latency+replan).
qvgm 은 그 밖에서 ĥ_Q = 0 이므로 v_θ = v_base 가 강제된다 (비편집 관절 보존).

usage: POLICY-EXTRACTION.md 참고
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from rl.data import build_flat, build_images, find_sessions, open_images, resolve_modality
from rl.expo import ExpoConfig
from rl.nets import explore_spec
from rl.offline_critic import normalize_all
from rl.vla_rldx import load_state_action_processor, normalize_actions, normalize_states

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--method", required=True, choices=("parl", "qvgm", "edit"))
p.add_argument("--exp", default="openarm_rim")
p.add_argument("--data", type=Path, required=True)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--model-path", default="", help="학습 대상 base policy (기본: exp yaml)")
p.add_argument("--out", type=Path, help="LoRA 체크포인트 출력 (기본: <ckpt>/<exp>-policy/<tag>)")
p.add_argument("--tag", default="")
# --- parl ---
p.add_argument("--targets", default="parl_actions.npy",
               help="[parl] relabel_parl.py 가 만든 raw 액션 npy (work 기준)")
# --- qvgm ---
p.add_argument("--critic", default="",
               help="[qvgm] offline_iql_qvgm.py 체크포인트 (work 기준 상대경로 가능)")
p.add_argument("--flow-steps", type=int, default=10, help="[qvgm] 학습 롤아웃 K (배포는 4)")
p.add_argument("--late-steps", type=int, default=5, help="[qvgm] 정렬할 후반 스텝 수 M (논문 5)")
p.add_argument("--ascent-steps", type=int, default=4, help="[qvgm] Q 상승 횟수 J")
p.add_argument("--ascent-size", type=float, default=0.0, help="[qvgm] 상승 step size α")
p.add_argument("--auto-step", type=float, default=0.05,
               help="[qvgm] 차원당 목표 이동량 D → α = D/(J·median‖g‖). 0 이면 --ascent-size 사용")
p.add_argument("--clip-grad", type=float, default=1.0, help="[qvgm] clip_G — ∇_A Q 원소 클립")
p.add_argument("--groups", default="", help="편집 그룹 (기본: exp yaml 의 explore_groups)")
# --- edit ---
p.add_argument("--base-chunks", type=int, default=8,
               help="[edit] 결정 프레임당 캐시할 base 후보 수. 캐시가 없으면 만든다")
p.add_argument("--cache-batch", type=int, default=64,
               help="[edit] 캐시를 만들 때 VLA 에 한 번에 넣을 결정 프레임 수 (--batch 와 별개. "
                    "--batch 은 MLP 학습용이다). 실측: 이 단계는 **CPU 병목**이라 배치를 키워도 "
                    "거의 안 빨라진다 — rldx PolicyRuntime._prepare_inputs 가 샘플별 파이썬 "
                    "루프로 이미지를 전처리한다 (batch 128/256 모두 ~13 결정/s, GPU 가동률 0%). "
                    "VRAM 만 늘어나니 64~128 이면 충분하다")
p.add_argument("--edit-scale", type=float, default=0.0,
               help="[edit] 편집 크기. 0 이면 exp yaml 의 expo.edit_scale")
p.add_argument("--actor-lr", type=float, default=0.0, help="[edit] 0 이면 yaml 의 actor_lr")
p.add_argument("--entropy-scale", type=float, default=0.0,
               help="[edit] SAC 엔트로피 보너스 계수. **기본 0 = 끈다.** 온라인 SAC 의 탐색용 "
                    "장치인데 오프라인 추출에는 필요 없고, 우리 스케일에서는 해롭다: Q 가 "
                    "[0,1] 로 유계(~0.24)인데 log_prob 은 104차원이라 ~96 이라 엔트로피 항이 "
                    "Q 를 수십~수천 배 압도한다 (실측: alpha 0.094 에서도 38배). 그러면 편집이 "
                    "최대 엔트로피 = 순수 랜덤에 머문다 (|edit| 이 10,000 스텝 내내 0.1176 고정). "
                    "온라인 EXPO 로 갈 때는 다시 켤 것")
# --- 공통 ---
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--lora-rank", type=int, default=16)
p.add_argument("--lora-alpha", type=int, default=32)
p.add_argument("--holdout", default="0.1")
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--save-every", type=int, default=500)
p.add_argument("--no-wandb", action="store_true", help="wandb 를 끈다")
p.add_argument("--wandb-key", default="", help="비우면 WANDB_API_KEY / ~/.netrc 를 쓴다")
p.add_argument("--wandb-project", default="rd-rl-policy")
p.add_argument("--wandb-entity", default="", help="비우면 계정 기본 entity")
p.add_argument("--wandb-run", default="", help="run 이름. 비우면 태그와 같게 둔다")
p.add_argument("--log-every", type=int, default=100, help="wandb 로깅 주기")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda")
a = p.parse_args()
dev = a.device
torch.manual_seed(a.seed)

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / (a.model_path or exp["base_policy"])
groups = [g.strip() for g in a.groups.split(",") if g.strip()] or list(exp["explore_groups"])
if a.method == "edit":
    # critic 종류를 태그에 넣는다 — 안 넣으면 iql 판과 qvgm 판이 같은 디렉토리를 덮어쓴다.
    _ck_kind = "unknown"
    for _c in (Path(a.critic), work / a.critic):
        if _c.is_file():
            _ck_kind = torch.load(_c, map_location="cpu").get("kind", "iql")
            break
    TAG = a.tag or (f"edit-{_ck_kind}-n{a.base_chunks}"
                    f"-es{(a.edit_scale or cfg.edit_scale):g}"
                    f"-lr{(a.actor_lr or cfg.actor_lr):g}-s{a.seed}")
else:
    TAG = a.tag or (f"{a.method}-lora{a.lora_rank}-lr{a.lr:g}"
                    + (f"-K{a.flow_steps}M{a.late_steps}J{a.ascent_steps}"
                       if a.method == "qvgm" else "")
                    + f"-s{a.seed}")
run = a.out or (a.checkpoints / f"{a.exp}-policy" / TAG)
run.mkdir(parents=True, exist_ok=True)


# --- wandb (offline_iql.py 와 같은 규약. 토큰은 절대 config 에 적지 않는다) -----
RUN_CFG = {"tag": TAG, **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "wandb_key": "***" if (a.wandb_key or os.environ.get("WANDB_API_KEY")) else ""}
(run / "config.json").write_text(json.dumps(RUN_CFG, indent=2, default=str, ensure_ascii=False))
wb = None
if not a.no_wandb:
    import wandb
    key = a.wandb_key or os.environ.get("WANDB_API_KEY", "")
    try:
        if key:
            wandb.login(key=key)
        wb = wandb.init(project=a.wandb_project, entity=a.wandb_entity or None,
                        name=a.wandb_run or TAG, dir=str(run), config=RUN_CFG)
        print(f"[wandb] {wb.url}")
    except Exception as e:                       # 로깅 때문에 학습을 날리지 않는다
        wb = None
        print(f"[wandb] 붙지 못했다 → 로깅 없이 진행한다 ({type(e).__name__}: {e})")
else:
    print("[wandb] --no-wandb")

# --- 1. 데이터 --------------------------------------------------------------
mod, _ = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
build_images(sessions, flat, work / "images.mm", mod)
imgs, meta = open_images(work / "images.mm")
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
if not (work / "actnorm.npy").is_file():
    normalize_all(lambda ch, st: normalize_actions(proc, mod.embodiment_tag, mod, ch, st),
                  flat, H, cache=work / "actnorm.npy")
norm = normalize_all(None, flat, H, cache=work / "actnorm.npy")
task = json.loads((sessions[0] / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
A_DIM, FULL = mod.action_dim, (LAT + R) * mod.action_dim

spec = explore_spec(mod.offsets("action"), groups, A_DIM, R, LAT)

frac = float(a.holdout) if a.holdout.replace(".", "", 1).isdigit() else 0.0
if 0 < frac < 1:
    every = max(2, int(round(1 / frac)))
    hold = np.isin(flat.episode, np.unique(flat.episode)[::every])
else:
    hold = np.isin(flat.session,
                   [i for i, n in enumerate(flat.sessions) if a.holdout and a.holdout in n])
# 결정 프레임만 쓴다 (replan 간격) — 실제로 정책이 질의받는 상태들이다
dec = np.concatenate([np.arange(fr[0], fr[-1] + 1, R)
                      for e in np.unique(flat.episode)
                      for fr in [np.flatnonzero(flat.episode == e)]])
train_idx = dec[~hold[dec]]
eval_idx = dec[hold[dec]]

print(f"[방법] {a.method}   exp={a.exp}  replan={R} latency={LAT} horizon={H}")
print(f"[데이터] 결정 프레임 학습 {len(train_idx)} / 평가 {len(eval_idx)}")
print(f"[편집 범위] {groups} → 청크 {FULL}차원 중 {len(spec.index)}개 "
      f"({spec.active_dim}관절 x {R}스텝, prefix {LAT}스텝 제외)")
print(f"[산출물] {run}/")

# --- 2. 정책 ----------------------------------------------------------------
# parl/qvgm 은 action expert LoRA 를 학습하므로 VLA 가 학습 루프에 들어간다.
# edit 은 base 를 영구 동결하고 보정만 학습하므로, base 청크 캐시를 만들 때만 VLA 를 쓴다.
CACHE = work / f"base_chunks_n{a.base_chunks}.npz"
vla = None
if a.method != "edit" or not CACHE.is_file():
    from rl.vla_rldx import RLDXVLA                    # noqa: E402
    vla = RLDXVLA(base, mod, RLDX, exp["rldx_data_config"], device=dev)

if a.method != "edit":
    vla.model.action_model.config.action_model_lora_rank = a.lora_rank
    vla.model.action_model.config.action_model_lora_alpha = a.lora_alpha
    info = vla.setup_training(lr=a.lr, lora=True)
    print(f"[정책] {base.name}  학습 파라미터 {info['trainable_params']/1e6:.2f}M "
          f"({info['trainable_tensors']} 텐서), 백본 학습 텐서 {info['backbone_trainable_tensors']}")

# --- 3a. PA-RL: relabel 된 라벨로 BC ----------------------------------------
if a.method == "parl":
    tgt_path = Path(a.targets) if Path(a.targets).is_file() else work / a.targets
    if not tgt_path.is_file():
        raise SystemExit(f"relabel 결과가 없다: {tgt_path}\n"
                         f"먼저 `python -m rl.relabel_parl ... --no-parquet` 를 돌려라")
    NEW = np.load(tgt_path)
    assert NEW.shape == flat.action.shape, f"{NEW.shape} != {flat.action.shape}"
    chg = np.abs(NEW - flat.action).max(1)
    print(f"[타깃] {tgt_path.name}  바뀐 프레임 {(chg>1e-6).mean():.1%}  "
          f"최대 변화 {chg.max():.4f} rad")

    def chunk_of(idx):
        """(B,H,A) relabel 된 raw 액션 청크. action_chunk 와 같은 클램프 규약."""
        off = np.minimum(idx[:, None] + np.arange(H)[None, :], flat.ep_end[idx][:, None])
        return NEW[off]

# --- 3c. edit: base 청크 캐시 + SAC residual policy -------------------------
elif a.method == "edit":
    from rl.nets import ResidualActor, Temperature      # noqa: E402
    from rl.policy_flow import obs_from_frames

    EDIT_SCALE = a.edit_scale or cfg.edit_scale
    ALR = a.actor_lr or cfg.actor_lr
    ESCALE = a.entropy_scale

    # (a) base 청크 캐시. base 가 영구 동결이라 한 번 만들면 끝이다.
    #     저장 규약은 expo.py:225-231 과 같다: 청크 앞 (LAT+R) 스텝만 쓰고 latency prefix
    #     블록은 로그된 값으로 덮는다 (실행이 이미 확정된 구간).
    if not CACHE.is_file():
        idx_all = np.concatenate([train_idx, eval_idx])
        idx_all.sort()
        out = np.empty((len(idx_all), a.base_chunks, FULL), np.float32)
        print(f"[캐시] base 청크 생성 {len(idx_all)} 결정 x {a.base_chunks} 후보 "
              f"→ {out.nbytes/1e6:.0f}MB")
        t0 = time.time()
        CB = a.cache_batch
        for c in range(0, len(idx_all), CB):
            k = idx_all[c:c + CB]
            with torch.no_grad():
                ch = vla.sample(obs_from_frames(imgs, flat, mod, task, k), a.base_chunks)
            acts = ch[:, :, :LAT + R].reshape(len(k), a.base_chunks, FULL).float()
            logged = torch.from_numpy(np.ascontiguousarray(
                np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev)
            if LAT:
                acts[:, :, :LAT * A_DIM] = logged[:, None, :LAT * A_DIM]
            out[c:c + len(k)] = acts.cpu().numpy()
            if (c // CB) % 10 == 0:
                el = time.time() - t0
                print(f"  {c+len(k)}/{len(idx_all)}  {el:.0f}s  "
                      f"(남은 {(len(idx_all)-c-len(k))*el/max(c+len(k),1)/60:.0f}분)", flush=True)
        # tmp 이름에 pid 를 넣는다 — 두 잡이 동시에 캐시를 만들어도 서로의 tmp 를 덮지 않는다.
        # np.savez 는 **경로 문자열이 .npz 로 끝나지 않으면 .npz 를 덧붙인다** — 그래서
        # 열린 파일 핸들을 넘긴다 (안 그러면 tmp 이름이 어긋나 os.replace 가 실패한다).
        tmp = CACHE.parent / f"{CACHE.stem}.tmp{os.getpid()}.npz"
        with open(tmp, "wb") as fh:
            np.savez(fh, idx=idx_all, chunks=out)
        os.replace(tmp, CACHE)                          # 원자적 교체
        print(f"[캐시] 저장 {CACHE} ({CACHE.stat().st_size/1e6:.0f}MB, {time.time()-t0:.0f}s)")
        del vla
        vla = None
        torch.cuda.empty_cache()
    z = np.load(CACHE)
    CIDX, CCH = z["idx"], z["chunks"]
    print(f"[캐시] {CACHE.name}  결정 {len(CIDX)} x 후보 {CCH.shape[1]} x {CCH.shape[2]}차원")
    POS = {int(v): i for i, v in enumerate(CIDX)}       # 전역 프레임 → 캐시 행
    BASE = torch.from_numpy(CCH).to(dev)                # (n_dec, N, FULL) 74MB

    # (b) critic — offline_iql 판과 qvgm 판 모두 받는다 (ckpt 의 kind 로 분기)
    ck = Path(a.critic)
    if not ck.is_file():
        ck = work / a.critic
    if not ck.is_file():
        raise SystemExit(f"--critic 이 필요하다 (edit). 없다: {a.critic}  (work={work})")
    kind = torch.load(ck, map_location="cpu").get("kind", "iql")
    if kind == "qvgm":
        from rl.critic_io import load_stepwise_critic   # noqa: E402
        C = load_stepwise_critic(ck, work, snorm, dev=dev)
        LATENT = C.meta["latent"] + C.meta["state_dim"]
        qfun = lambda lat, st, act: C.q(lat, act)
        qstdfun = lambda lat, st, act: C.critic(lat, act).sum(-1).std(0)
        latfun = lambda i: C.latent_of(i)
    else:
        from rl.critic_io import load_critic            # noqa: E402
        C = load_critic(ck, work, cfg, mod.n_cams, FULL, snorm.shape[1],
                        features="", imgs=imgs, dev=dev)
        feat = C.meta.get("features") or ""
        LATENT = (cfg.latent_dim_image + snorm.shape[1]) if feat else cfg.latent_dim_image
        qfun = lambda lat, st, act: C.q(lat, st, act).mean(0)   # 앙상블 평균 (EXPO 원본)
        qstdfun = lambda lat, st, act: C.q(lat, st, act).std(0)
        SN = torch.from_numpy(snorm).to(dev)
        latfun = lambda i: C.latent_of(i, SN[torch.as_tensor(i, device=dev)])

    # probe_actopt 와 같은 기준: 같은 (관절, 청크스텝) 에서 t→t+1 변화 = 1프레임 자연 움직임
    _pi = CIDX[:: max(1, len(CIDX) // 512)][:512]
    _d1 = (np.asarray(norm[_pi + 1])[:, :LAT + R].reshape(len(_pi), -1)
           - np.asarray(norm[_pi])[:, :LAT + R].reshape(len(_pi), -1))[:, spec.index.numpy()]
    REF1 = float(np.median(np.linalg.norm(_d1, axis=-1)) / len(spec.index) ** 0.5)
    print(f"[기준] 1프레임 자연 변화 {REF1:.5f}/차원 → edit_scale {EDIT_SCALE} 은 "
          f"{EDIT_SCALE/REF1:.1f} 프레임치가 상한")

    SNT = torch.from_numpy(snorm).to(dev)
    residual = ResidualActor(LATENT, snorm.shape[1], spec, cfg.latent_dim_state,
                             include_state=False, hidden_dims=cfg.hidden_dims).to(dev)
    temp = Temperature(cfg.init_temperature).to(dev)
    opt_r = torch.optim.Adam(residual.parameters(), lr=ALR)
    opt_t = torch.optim.Adam(temp.parameters(), lr=cfg.temp_lr or ALR)
    print(f"[edit] critic={kind}  latent {LATENT}  edit_scale {EDIT_SCALE}  actor_lr {ALR}")
    print(f"[edit] 학습 파라미터 {sum(p.numel() for p in residual.parameters())/1e6:.2f}M "
          f"+ temperature 1개.  action expert 는 건드리지 않는다")
    # 목표 엔트로피를 **스케일 공간**으로 옮긴다.
    # ResidualActor.sample 은 EXPO-FT 규약대로 log_prob 에 -out_dim*log(edit_scale) 보정을
    # 넣는다 (nets.py:409). 그래서 보고되는 엔트로피는 tanh 공간보다 out_dim*log(edit_scale)
    # 만큼 낮다. 그런데 ExploreSpec.target_entropy 는 tanh 공간 기준(-out_dim/2)이다.
    #
    # edit_scale=0.2, out_dim=104 에서 실측한 결과:
    #   달성 가능한 최대 엔트로피 = out_dim*log(2*edit_scale) = -95.3
    #   목표 -out_dim/2          = -52.0   ← 최대값보다 높다 = 영원히 도달 못 한다
    #   → alpha 가 무한히 커진다 (실측 6.3 에서 계속 증가). actor 손실의 엔트로피 항이
    #     alpha*|log_prob| ~ 605 로 Q(0.24) 를 2,520배 압도해 Q 를 아예 무시한다.
    #     결과: 5,000 스텝 학습해도 이득 -0.0007 (편집이 아무 일도 안 한다)
    #
    # 같은 공간으로 옮기면 -out_dim/2 + out_dim*log(edit_scale) = -219.4 이고, 실측 -96.3 은
    # 그보다 높으므로 alpha 가 0 으로 내려가 Q 항이 실제로 작동한다.
    # ※ rl/expo.py 의 온라인 루프도 같은 규약을 쓰므로 같은 문제가 있을 수 있다 —
    #   ExploreSpec 은 건드리지 않고 여기서만 보정한다 (온라인 경로는 따로 확인할 것).
    TARGET_ENT = spec.target_entropy + spec.out_dim * math.log(EDIT_SCALE)
    ENT_SHIFT = spec.out_dim * math.log(EDIT_SCALE)     # 로그용: tanh 공간 환산
    if ESCALE > 0:
        print(f"[edit] 엔트로피 보너스 켬 (scale {ESCALE}). 목표 {TARGET_ENT:.1f} = "
              f"-{spec.out_dim}/2 + {spec.out_dim}*log({EDIT_SCALE}) (스케일 공간), "
              f"달성 가능 최대 {spec.out_dim * math.log(2 * EDIT_SCALE):.1f}")
    else:
        print("[edit] 엔트로피 보너스 끔 (--entropy-scale 0) → 손실 = -Q. "
              "오프라인 추출에는 탐색이 필요 없고, Q[0,1] 대 log_prob~96 스케일이 어긋난다")

    def batch_of(sel, gen=None):
        """(lat, state, base) — 캐시에서 후보 하나를 랜덤으로 고른다."""
        rows = torch.as_tensor([POS[int(v)] for v in sel], device=dev)
        pick = torch.randint(BASE.shape[1], (len(sel),), device=dev, generator=gen)
        return latfun(sel), SNT[torch.as_tensor(sel, device=dev)], BASE[rows, pick]

# --- 3b. Q-VGM: velocity 타깃 -----------------------------------------------
else:
    from rl.critic_io import load_stepwise_critic       # noqa: E402
    from rl.policy_flow import FlowPolicy, chunk_mask, obs_from_frames

    ck = Path(a.critic)
    if not ck.is_file():
        ck = work / a.critic
    if not ck.is_file():
        raise SystemExit(f"--critic 이 필요하다 (qvgm). 없다: {a.critic}  (work={work})")
    C = load_stepwise_critic(ck, work, snorm, dev=dev)
    flow = FlowPolicy(vla, flow_steps=a.flow_steps)
    flow.eval_mode()                                   # dropout off — v_base 가 결정적이어야 한다
    K, M = a.flow_steps, min(a.late_steps, a.flow_steps)
    LATE = set(range(K - M, K))
    # DiT 는 (horizon=16, max_action_dim=64) 로 적분한다. 마스크를 그 좌표로 옮겨 심는다.
    MASK_CHUNK = chunk_mask(spec, flow.horizon, flow.dim, A_DIM, dev)
    print(f"[qvgm] K={K} 스텝 롤아웃, 후반 M={M} 스텝 정렬 (τ >= {(K-M)/K:.2f}), "
          f"J={a.ascent_steps} 상승, clip_G={a.clip_grad}")
    print(f"[qvgm] DiT 공간 ({flow.horizon}, {flow.dim}) 중 편집 {int(MASK_CHUNK.sum())}개 "
          f"= spec.index {len(spec.index)}개  (일치해야 함)")
    assert int(MASK_CHUNK.sum()) == len(spec.index)

    def q_of(idx_t, chunk):
        """(B,) 스칼라 Q. critic 은 실제 관절 A_DIM 의 앞 LAT+R 스텝만 본다."""
        lat = C.latent_of(idx_t)
        return C.q(lat, chunk[:, :LAT + R, :A_DIM].reshape(len(chunk), -1))

    def pad_chunk(arr):
        """(B, H, A_DIM) 정규화 청크 → DiT 공간 (B, horizon, max_action_dim). 나머지는 0."""
        z = torch.zeros(len(arr), flow.horizon, flow.dim, device=dev)
        h = min(arr.shape[1], flow.horizon)
        z[:, :h, :A_DIM] = torch.from_numpy(np.ascontiguousarray(arr[:, :h])).to(dev)
        return z

    def ascend(idx_t, A0, alpha):
        """J번 ∇_A Q 상승 + keep-best (Eq. 7). 편집 마스크 적용. (B,H,A), (B,) 최종 Q."""
        best_A, best_q = A0, q_of(idx_t, A0)
        Acur = A0
        for _ in range(a.ascent_steps):
            Acur = Acur.detach().requires_grad_(True)
            g, = torch.autograd.grad(q_of(idx_t, Acur).sum(), Acur)
            g = g.clamp(-a.clip_grad, a.clip_grad) * MASK_CHUNK
            Acur = (Acur + alpha * g).clamp(-1.0, 1.0).detach()
            qc = q_of(idx_t, Acur)
            take = (qc > best_q)[:, None, None]
            best_A = torch.where(take, Acur, best_A)
            best_q = torch.maximum(best_q, qc)
        return best_A, best_q

    # --- α 캘리브레이션 (probe_actopt --auto-step 과 같은 규칙) ---
    ALPHA = a.ascent_size
    if a.auto_step > 0:
        k = eval_idx[:: max(1, len(eval_idx) // 32)][:32]
        A0 = pad_chunk(np.asarray(norm[k])).requires_grad_(True)
        g, = torch.autograd.grad(q_of(k, A0).sum(), A0)
        gg = (g * MASK_CHUNK)[MASK_CHUNK.bool().expand_as(g)].view(len(k), -1)
        gm = float((gg.norm(dim=-1) / len(spec.index) ** 0.5).median())
        ALPHA = a.auto_step / max(a.ascent_steps * gm, 1e-12)
        print(f"[auto-step] median‖g‖/√d = {gm:.3e} → α = {ALPHA:.4g} "
              f"(목표 이동 {a.auto_step}/차원)")

# --- 4. 학습 ----------------------------------------------------------------
hist = []
t0 = time.time()
for step in range(1, a.steps + 1):
    sel = train_idx[np.random.default_rng(a.seed + step).integers(len(train_idx), size=a.batch)]
    if a.method == "edit":
        lat, st_, bse = batch_of(sel)
        edit, logp = residual.sample(lat.detach(), st_, bse, EDIT_SCALE)
        q = qfun(lat.detach(), st_, (bse + edit).clamp(-1.0, 1.0))
        ent = float(-logp.mean().detach())
        if ESCALE > 0:
            alpha = temp().detach()
            loss = (ESCALE * logp * alpha - q).mean()
        else:
            loss = -q.mean()                              # 순수 Q 최대화 (오프라인 추출)
        opt_r.zero_grad(set_to_none=True)
        loss.backward()
        opt_r.step()
        if ESCALE > 0:
            t_loss = temp() * (ent - TARGET_ENT)          # 목표 엔트로피로 alpha 자동 조절
            opt_t.zero_grad(set_to_none=True)
            t_loss.backward()
            opt_t.step()
        m = {"actor_loss": float(loss.detach()), "q": float(q.mean().detach()),
             "entropy": ent, "entropy_tanh": ent - ENT_SHIFT,
             "temperature": float(temp().detach()),
             "edit_norm": float(edit.detach()[:, spec.index].norm(dim=-1).mean()
                                / len(spec.index) ** 0.5)}
    elif a.method == "parl":
        obs = {"video": {name: np.asarray(imgs[sel])[:, c][:, None]
                         for c, (name, _) in enumerate(mod.video)},
               "state": {name: flat.state[sel][:, None, s:e]
                         for name, s, e in mod.offsets("state")},
               "language": {mod.task_key: [[task]] * len(sel)}}
        m = vla.train_step(obs, chunk_of(sel))
    else:
        obs = obs_from_frames(imgs, flat, mod, task, sel)
        ctx = flow.context(obs)
        xs, taus = flow.rollout(ctx)                   # detach 된 현재 정책 롤아웃
        loss = 0.0
        dq_sum, ood_sum, n_late = 0.0, 0.0, 0
        for k in range(K):
            if k not in LATE:
                continue
            x, tau = xs[k], taus[k]
            rem = 1.0 - float(k) / K                   # 1 - τ_k
            with torch.no_grad():
                v_b = flow.velocity(ctx, x, tau, base=True)
                A0 = (x + rem * v_b).clamp(-1.0, 1.0)  # look-forward anchor (Eq. 5)
                q0 = q_of(sel, A0)
            A_q, q1 = ascend(sel, A0, ALPHA)
            h_q = ((A_q - A0) / rem).detach()           # Eq. 8
            v_th = flow.velocity(ctx, x, tau)           # gradient 는 여기로만
            loss = loss + (((v_th - v_b.detach()) - h_q) ** 2).mean()
            dq_sum += float((q1 - q0).mean())
            n_late += 1
        vla.opt.zero_grad(set_to_none=True)
        loss.backward()
        vla.opt.step()
        m = {"actor_loss": float(loss.detach()), "dQ": dq_sum / max(n_late, 1)}

    hist.append(m)
    if wb is not None and (step % a.log_every == 0 or step == a.steps):
        wb.log({f"train/{k}": v for k, v in m.items()}, step=step)
    if step % 20 == 0:
        el = time.time() - t0
        if a.method == "qvgm":
            extra = f"  ΔQ {np.mean([h['dQ'] for h in hist[-20:]]):+.4f}"
        elif a.method == "edit":
            extra = (f"  Q {np.mean([h['q'] for h in hist[-20:]]):+.4f}"
                     f"  ent {np.mean([h['entropy'] for h in hist[-20:]]):+.1f}"
                     f"/{TARGET_ENT:+.0f}"
                     f"  alpha {hist[-1]['temperature']:.3f}"
                     f"  |edit| {np.mean([h['edit_norm'] for h in hist[-20:]]):.4f}")
        else:
            extra = ""
        print(f"  step {step:5d}  loss {np.mean([h['actor_loss'] for h in hist[-20:]]):.5f}{extra}"
              f"  {el/step:.2f}s/step  남은 {(a.steps-step)*el/step/60:.0f}분", flush=True)

    # --- 평가 (edit): Q(base+edit) 가 Q(base) / Q(로그) 를 넘는가 ---
    if a.method == "edit" and (step % a.eval_every == 0 or step == a.steps):
        with torch.no_grad():
            rows = []
            for c in range(0, len(eval_idx), 256):
                k = eval_idx[c:c + 256]
                lat, st_, bse = batch_of(k)
                e, _ = residual.sample(lat, st_, bse, EDIT_SCALE)
                lg = torch.from_numpy(np.ascontiguousarray(
                    np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev)
                ae = (bse + e).clamp(-1, 1)
                rows.append(torch.stack([
                    qfun(lat, st_, ae), qfun(lat, st_, bse), qfun(lat, st_, lg),
                    # 앙상블 불일치 — 편집이 critic 이 아는 영역을 벗어나면 커진다
                    qstdfun(lat, st_, ae), qstdfun(lat, st_, bse),
                    # 난수 편집을 같은 크기로 준 대조군 (Q 상승이 편집 방향 때문인지 확인)
                    qfun(lat, st_, (bse + spec.scatter(
                        (torch.rand(len(k), spec.out_dim, device=dev) * 2 - 1) * EDIT_SCALE)
                        ).clamp(-1, 1)),
                ], -1).cpu().numpy())
            r = np.concatenate(rows)
        # 편집량을 "1프레임 자연 변화" 단위로. 정규화 공간 수치는 해석이 안 된다.
        en = np.mean([h["edit_norm"] for h in hist[-20:]])
        print(f"  [eval] step {step:5d}  Q(base+edit) {r[:,0].mean():+.4f}  "
              f"Q(base) {r[:,1].mean():+.4f}  Q(로그) {r[:,2].mean():+.4f}  "
              f"Q(난수편집) {r[:,5].mean():+.4f}  이득(vs base) {(r[:,0]-r[:,1]).mean():+.4f}")
        print(f"          앙상블std {r[:,4].mean():.4f}->{r[:,3].mean():.4f} "
              f"({r[:,3].mean()/max(r[:,4].mean(),1e-9):.2f}배)  "
              f"편집량 {en:.4f} = {en/REF1:.1f} 프레임치 "
              f"(상한 {EDIT_SCALE/REF1:.1f})  "
              f"이득/std {(r[:,0]-r[:,1]).mean()/max(r[:,3].mean(),1e-9):.2f}")
        if wb is not None:
            wb.log({"eval/q_edited": float(r[:, 0].mean()),
                    "eval/q_base": float(r[:, 1].mean()),
                    "eval/q_logged": float(r[:, 2].mean()),
                    "eval/q_random_edit": float(r[:, 5].mean()),
                    "eval/q_gain": float((r[:, 0] - r[:, 1]).mean()),
                    "eval/ens_std_base": float(r[:, 4].mean()),
                    "eval/ens_std_edited": float(r[:, 3].mean()),
                    "eval/edit_frames": float(en / REF1)}, step=step)

    # --- 평가: 정책 자기 샘플의 Q 가 오르는가 (qvgm 만; parl 은 별도 도구로) ---
    if a.method == "qvgm" and (step % a.eval_every == 0 or step == a.steps):
        with torch.no_grad():
            qs = []
            for c in range(0, min(len(eval_idx), 64), a.batch):
                k = eval_idx[c:c + a.batch]
                ctx_e = flow.context(obs_from_frames(imgs, flat, mod, task, k))
                xe, te = flow.rollout(ctx_e)
                Ae = (xe[-1] + (1.0 / K) * flow.velocity(ctx_e, xe[-1], te[-1])).clamp(-1, 1)
                q_pi = q_of(k, Ae)
                q_dat = q_of(k, pad_chunk(np.asarray(norm[k])))
                qs.append(torch.stack([q_pi, q_dat], -1).cpu().numpy())
            qs = np.concatenate(qs)
        print(f"  [eval] step {step:5d}  Q(정책) {qs[:,0].mean():+.4f}  "
              f"Q(로그) {qs[:,1].mean():+.4f}  차이 {(qs[:,0]-qs[:,1]).mean():+.4f}")
        if wb is not None:
            wb.log({"eval/q_policy": float(qs[:, 0].mean()),
                    "eval/q_logged": float(qs[:, 1].mean()),
                    "eval/q_gain": float((qs[:, 0] - qs[:, 1]).mean())}, step=step)

    if step % a.save_every == 0 or step == a.steps:
        if a.method == "edit":
            # ExpoServer._load 가 읽는 키 이름을 그대로 쓴다 (residual / temp).
            sd = {"residual": residual.state_dict(), "temp": temp.state_dict()}
            ck = run / f"edit_{step:06d}.pt"
            tmp = ck.with_suffix(".pt.tmp")
            torch.save({**sd, "method": a.method, "step": step, "exp": a.exp,
                        "base_policy": str(base), "groups": groups, "tag": TAG,
                        "edit_scale": EDIT_SCALE, "critic": a.critic,
                        "critic_kind": kind, "latent": LATENT}, tmp)
            os.replace(tmp, ck)
            lnk, ltmp = run / "edit_latest.pt", run / "edit_latest.pt.tmp"
            ltmp.unlink(missing_ok=True)
            ltmp.symlink_to(ck.name)
            os.replace(ltmp, lnk)
            print(f"  [저장] {ck.name} ({ck.stat().st_size/1e6:.1f}MB)")
            continue
        sd = {k: v.detach().cpu() for k, v in vla.model.state_dict().items()
              if "lora_" in k}                          # 학습 가능한 것만 = LoRA adapter
        ck = run / f"lora_{step:06d}.pt"
        tmp = ck.with_suffix(".pt.tmp")
        torch.save({"lora": sd, "method": a.method, "step": step, "exp": a.exp,
                    "base_policy": str(base), "lora_rank": a.lora_rank,
                    "lora_alpha": a.lora_alpha, "groups": groups, "tag": TAG,
                    "flow_steps": a.flow_steps, "late_steps": a.late_steps,
                    "ascent_steps": a.ascent_steps,
                    "ascent_size": (ALPHA if a.method == "qvgm" else None)}, tmp)
        os.replace(tmp, ck)
        lnk, ltmp = run / "lora_latest.pt", run / "lora_latest.pt.tmp"
        ltmp.unlink(missing_ok=True)
        ltmp.symlink_to(ck.name)
        os.replace(ltmp, lnk)
        print(f"  [저장] {ck.name} ({ck.stat().st_size/1e6:.1f}MB, {len(sd)} 텐서)")

if wb is not None:
    wb.finish()
print(f"\n[완료] {run}")
