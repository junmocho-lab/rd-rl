#!/usr/bin/env python3
"""critic 오프라인 학습 — IQL 판 (rl/offline_critic_0.py 의 SARSA 를 IQL 로 바꾼 것).

SARSA 와 다른 곳은 **타깃 하나**다:

    SARSA (offline_critic_0.py)   tq = r + γ^R·mask· min_2 Q_target(s', a'_로그된액션)
    IQL   (이 파일)               tq = r + γ^R·mask· V(s')
                                  V  ← expectile_τ( min_2 Q_target(s, a_data) − V(s) )

즉 "다음 상태에서 로그된 액션의 Q" 대신 "다음 상태의 가치 V" 를 쓰고, 그 V 를 데이터 액션의 Q
분포의 **상위 expectile** 로 맞춘다. τ>0.5 면 V 가 평균보다 위를 보므로 데이터 안에서(=외삽 없이)
정책 개선이 들어간다. PA-RL 의 parl_iql 과 같은 구성이고, 거기서도 actor 는 학습하지 않는다
(`parl_iql.py:171` train_actor=False) — 정책은 롤아웃 때 후보 argmax 로 만든다.

우리 세팅에서 기대하는 부수효과: SARSA 타깃의 `min_2` 가 매 backup 마다 음의 편향을 넣어
Q 가 음수 고정점(측정 -0.013)에 갇혔는데, IQL 타깃에는 그 min 이 없다 (min 은 V 회귀의 타깃에만
남는다). 그래서 Q 스케일이 정상화되는지가 첫 확인 대상이다.

캐시(images.mm / actnorm.npy)는 offline_critic_0.py 와 공유하고, **산출물은 실험(태그)마다
디렉토리 하나**로 분리한다:

    <checkpoints>/<exp>-critic/
      images.mm, images.json, actnorm.npy      ← 실험끼리 공유 (재디코딩 안 하려고)
      <tag>/                                   ← 실험 하나 = 디렉토리 하나
        config.json                            인자/설정 스냅샷
        critic_<step>.pt                       enc/critic/target/value (+ bins/q_range)
        critic_latest.pt                       → 가장 최근 것을 가리키는 심링크
        plots/<step>_qv.png                    홀드아웃 에피소드의 Q·V 궤적
        videos/<step>_ep<번호>_<succ|fail>.mp4  카메라 + Q·V 커서 오버레이

태그 기본값은 설정에서 자동 생성된다 (예: iql-scalar-t07-s0, iql-dist64-t07-s0) — 스칼라 판과
distributional 판을 나란히 비교할 때 서로 덮어쓰지 않는다.

wandb 는 **loss 만** 올린다 (loss_q / loss_v / loss). AUC·플롯·비디오는 로컬에만 남는다.
토큰은 `WANDB_API_KEY` 환경변수로 주는 것을 권한다 — `--wandb-key` 는 `ps` 에 그대로 보인다.

usage:
  source configs/paths.sh
  wandb login              # 토큰은 ~/.netrc 에. 이 파일은 .gitignore 대상이 아니다
                           # 또는: export WANDB_API_KEY=<토큰>  /  --no-wandb

  IQL from critic in EXPO-FT
  PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" NO_ALBUMENTATIONS_UPDATE=1 third_party/RLDX-1/.venv/bin/python -u -m rl.offline_iql \
  --exp fuji --data rl-dataset/fuji-rl-dataset --checkpoints checkpoints \
  --steps 40000 --holdout 0.2 --eval-every 2500 --expectile 0.7 --discount 0.999

  IQL with distributional critic
  PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" NO_ALBUMENTATIONS_UPDATE=1 third_party/RLDX-1/.venv/bin/python -u -m rl.offline_iql \
  --exp fuji --data rl-dataset/fuji-rl-dataset --checkpoints checkpoints \
  --discount 0.999 --bins 128 --expectile 0.7 \
  --steps 40000 --holdout 0.2 --eval-every 2500
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import cv2
import imageio.v2 as iio
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

from rl.data import build_flat, build_images, find_sessions, nstep, open_images, resolve_modality
from rl.expo import ExpoConfig
from dataclasses import fields as _dc_fields
import torch.nn as nn

from rl.nets import BatchEncoder, CriticEnsemble, xavier_
from rl.offline_critic import normalize_all
from rl.vla_rldx import RLDXVLA, load_state_action_processor, normalize_states

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="fuji")
p.add_argument("--data", type=Path, required=True)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval-every", type=int, default=2500,
               help="이 주기마다 홀드아웃 평가 + Q·V 플롯 + 체크포인트를 한꺼번에 저장한다")
p.add_argument("--holdout", default="0.2", help="세션 이름 문자열 또는 에피소드 비율(0<x<1)")
p.add_argument("--expectile", type=float, default=0.7, help="τ. 0.5 면 SARSA 에 가까워진다")
p.add_argument("--bins", type=int, default=0,
               help="0 이면 스칼라 MSE. >0 이면 distributional critic (HL-Gauss, PA-RL 기본 128)")
p.add_argument("--q-range", default="0,1",
               help="distributional 일 때 support. 우리 보상은 성공 종료 1프레임뿐이라 리턴 ∈ [0,1]")
p.add_argument("--discount", type=float, default=0.0,
               help="프레임당 할인. 0 이면 exp yaml 값. 결정당 실효는 discount^replan 이다 — "
                    "heuristic 1-1/L 은 L 을 **프레임** 길이로 넣으면 된다 (fuji 1200 → 0.99917)")
p.add_argument("--num-qs", type=int, default=0,
               help="Q 앙상블 크기. 0 이면 exp yaml 값(10). PA-RL 은 distributional 이면 10, "
                    "스칼라면 2 를 쓴다. yaml 과 다르게 주면 ExpoServer 가 못 읽는다")
p.add_argument("--v-min", default="all", choices=("all", "sub"),
               help="V 타깃의 Q 앙상블 축약. all = 전체 min (PA-RL: iql.py 의 jnp.min(q,axis=0)), "
                    "sub = 무작위 num_min_qs 개 min (REDQ 식)")
p.add_argument("--images", default="gpu", choices=("gpu", "mmap"),
               help="gpu: images.mm 을 통째로 VRAM 에 올린다 (fuji 30.7GB). NFS 랜덤읽기가 "
                    "스텝 시간의 96%% 라 이게 가장 큰 개선. 여유가 없으면 mmap 으로 떨어진다")
p.add_argument("--tag", default="",
               help="실험 디렉토리 이름. 비우면 설정에서 자동 생성 (판/τ/seed 가 달라도 안 겹치게)")
# --- wandb (loss 만 올린다) --------------------------------------------------
p.add_argument("--no-wandb", action="store_true", help="wandb 를 끈다")
p.add_argument("--wandb-key", default="",
               help="API 토큰. **`ps` 에 그대로 보인다** — 가능하면 WANDB_API_KEY 환경변수를 쓸 것")
p.add_argument("--wandb-project", default="rd-rl-critic")
p.add_argument("--wandb-entity", default="", help="비우면 계정 기본 entity")
p.add_argument("--wandb-run", default="", help="run 이름. 비우면 태그와 같게 둔다")
p.add_argument("--log-every", type=int, default=100, help="loss 를 wandb 에 올리는 주기")
# --- 평가 비디오 -------------------------------------------------------------
p.add_argument("--video-every", type=int, default=2500,
               help="N 스텝마다 비디오. 0 = 평가할 때마다 (--eval-every 주기). 음수 = 안 만든다. "
                    "비디오는 평가 블록 안에서 굽히므로 --eval-every 의 배수여야 실제로 걸린다")
p.add_argument("--keep-last", type=int, default=1,
               help="남길 체크포인트 개수. 0 이면 전부 보관. 하나가 294MB 라 40k/2500 = 16개 = 4.7GB")
p.add_argument("--video-stride", type=int, default=4,
               help="비디오에 넣을 프레임 간격. fuji 는 에피소드가 ~1200 프레임이라 4면 300컷")
p.add_argument("--video-fps", type=int, default=30)
p.add_argument("--video-eps", type=int, default=0,
               help="비디오를 만들 홀드아웃 에피소드 수. 0 이면 전부")
a = p.parse_args()

torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
gen = torch.Generator().manual_seed(a.seed)
assert torch.cuda.is_available(), "GPU 전용이다"
dev = "cuda"

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
if a.discount:
    cfg.discount = a.discount                        # 지평 스윕용 (yaml 을 고치지 않는다)
if a.num_qs:
    cfg.num_qs = a.num_qs
R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
GEFF = cfg.discount ** R
base = a.checkpoints / exp["base_policy"]
work = a.checkpoints / f"{a.exp}-critic"             # images.mm / actnorm.npy 는 공유
# 산출물만 태그로 분리한다 — 스칼라 판과 distributional 판을 같이 비교하려면 필수
TAG = a.tag or (f"iql-{'dist' + str(a.bins) if a.bins else 'scalar'}"
                f"-t{str(a.expectile).replace('.', '')}"
                f"-g{f'{cfg.discount:g}'.replace('.', '')}"
                f"-q{cfg.num_qs}{a.v_min}-s{a.seed}")
run = work / TAG                                     # 실험 하나 = 디렉토리 하나
plots, vids = run / "plots", run / "videos"
for d in (run, plots, vids):
    d.mkdir(parents=True, exist_ok=True)
# 체크포인트는 스텝마다 별도 파일로 남기고, critic_latest.pt 심링크로 최신을 가리킨다

# --- 1. 데이터 (offline_critic_0.py 와 동일) ---------------------------------
mod, src = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
build_images(sessions, flat, work / "images.mm", mod)
imgs, meta = open_images(work / "images.mm")
# 이미지를 VRAM 에 상주시킨다. 실측: NFS 랜덤 읽기가 배치당 0.7~1.9s (96 MB/s) 인데 스텝의
# GPU 계산은 81ms 뿐이라 가동률이 4% 였다. 순차 업로드는 364~469 MB/s 로 30.7GB 가 1.4분.
GI = None
if a.images == "gpu":
    need = int(np.prod(meta["shape"]))
    freeb = torch.cuda.mem_get_info()[0]
    if freeb < need * 1.15:
        print(f"[이미지] VRAM 부족 (여유 {freeb/1e9:.1f}GB < 필요 {need*1.15/1e9:.1f}GB) → mmap")
    else:
        t0 = time.time()
        GI = torch.empty(tuple(meta["shape"]), dtype=torch.uint8, device=dev)
        for c in range(0, meta["shape"][0], 2048):    # 순차로 읽어 올린다
            GI[c:c + 2048] = torch.from_numpy(np.array(imgs[c:c + 2048])).to(dev)
        print(f"[이미지] VRAM 상주 {need/1e9:.1f}GB  {time.time()-t0:.0f}s  "
              f"(남은 여유 {torch.cuda.mem_get_info()[0]/1e9:.1f}GB)")
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
if not (work / "actnorm.npy").is_file():
    vla = RLDXVLA(base, mod, RLDX, exp["rldx_data_config"], device=dev)
    normalize_all(vla, flat, H, cache=work / "actnorm.npy")
    del vla
    torch.cuda.empty_cache()
norm = normalize_all(None, flat, H, cache=work / "actnorm.npy")

frac = float(a.holdout) if a.holdout.replace(".", "", 1).isdigit() else 0.0
if 0 < frac < 1:
    every = max(2, int(round(1 / frac)))
    hold = np.isin(flat.episode, np.unique(flat.episode)[::every])
    how = f"에피소드 {every}개마다 1개"
else:
    sel = [i for i, n in enumerate(flat.sessions) if a.holdout and a.holdout in n]
    hold = np.isin(flat.session, sel)
    how = f"세션 이름에 '{a.holdout}' 포함 = {[flat.sessions[i] for i in sel]}"
train = np.flatnonzero(~hold[:len(flat) - R])
eps = [(e, np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode[hold])]
eps = [(e, fr, bool(flat.is_success[fr[-1]])) for e, fr in eps]
n_ok = sum(o for _, _, o in eps)
print(f"[exp] {a.exp} replan={R} latency={LAT} horizon={H} expectile={a.expectile}")
print(f"[할인] 프레임당 {cfg.discount} → 결정당 {GEFF:.5f}  지평 {1/(1-GEFF):.0f} 결정 "
      f"= {1/(1-GEFF)*R:.0f} 프레임  (min 편향 증폭 {1/(1-GEFF):.0f}배)")
print(f"[critic] 앙상블 {cfg.num_qs}, V 타깃 min = {a.v_min}")
print(f"[산출물] {run}/  (critic_<step>.pt · critic_latest.pt · plots/ · videos/)")
print(f"[데이터] 세션 {len(sessions)} / 프레임 {len(flat)} / 학습 {len(train)} / "
      f"state {flat.state.shape[1]}→{snorm.shape[1]}차원")
print(f"[평가셋] {how} → 에피소드 {len(eps)} (성공 {n_ok} / 실패 {len(eps) - n_ok})")
if not n_ok or n_ok == len(eps):
    raise SystemExit("평가셋에 성공/실패가 한쪽뿐이라 AUC 를 못 낸다 — --holdout 을 바꿀 것")

# 실험 스냅샷. 나중에 어떤 설정이었는지 디렉토리만 보고 알 수 있게 남긴다.
# **토큰은 절대 적지 않는다** — 있었는지 여부만 남긴다.
RUN_CFG = {"tag": TAG, **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "wandb_key": "***" if (a.wandb_key or os.environ.get("WANDB_API_KEY")) else "",
           "replan_steps": R, "inference_latency": LAT, "action_horizon": H,
           "gamma_eff": GEFF, "n_train": int(len(train)), "n_eval_eps": len(eps),
           "expo": {f.name: getattr(cfg, f.name) for f in _dc_fields(ExpoConfig)}}
(run / "config.json").write_text(json.dumps(RUN_CFG, indent=2, default=str, ensure_ascii=False))

# --- wandb: loss 만 올린다 ----------------------------------------------------
wb = None
if not a.no_wandb:
    import wandb
    key = a.wandb_key or os.environ.get("WANDB_API_KEY", "")
    try:
        if key:
            wandb.login(key=key)
        # 키가 없어도 `wandb login` 으로 ~/.netrc 에 저장해 뒀으면 그대로 붙는다
        wb = wandb.init(project=a.wandb_project, entity=a.wandb_entity or None,
                        name=a.wandb_run or TAG, dir=str(run), config=RUN_CFG)
        print(f"[wandb] {wb.url}")
    except Exception as e:                       # 로깅 때문에 40k 스텝을 날리지 않는다
        wb = None
        print(f"[wandb] 붙지 못했다 → 로깅 없이 진행한다 ({type(e).__name__}: {e})")
else:
    print("[wandb] --no-wandb")

# 액션/상태도 GPU 상주 (각각 83MB / 7.5MB). 남는 CPU 작업은 nstep 뿐이다.
NORM = torch.from_numpy(np.ascontiguousarray(np.asarray(norm[:, :LAT + R]))).to(dev)
SNORM = torch.from_numpy(snorm).to(dev)

def obs(i):
    """(B, H, W, 3*n_cams) uint8. **카메라 concat 을 GPU 에서** 한다 — CPU concat 은 340ms 였다."""
    if GI is not None:
        x = GI[torch.as_tensor(i, device=dev)]
    else:
        x = torch.from_numpy(np.ascontiguousarray(np.asarray(imgs[i]))).to(dev)
    return torch.cat([x[:, c] for c in range(x.shape[1])], -1)

act = lambda i: NORM[torch.as_tensor(i, device=dev)].reshape(len(i), -1)
st = lambda i: SNORM[torch.as_tensor(i, device=dev)]

# --- 1b. 평가 비디오 --------------------------------------------------------
# 홀드아웃 에피소드마다 "카메라 스트립 + Q·V 궤적(현재 프레임 커서)" mp4 를 굽는다.
# 플롯은 에피소드당 **한 번만** matplotlib 으로 굽고, 프레임마다는 cv2 로 커서만 그린다
# (프레임마다 다시 그리면 300컷에 10초씩 걸린다).
_m16 = lambda x: max(16, int(round(x / 16)) * 16)   # h264 가 좋아하는 16 배수


def _raw_cams(idx) -> np.ndarray:
    """(B, n_cams, H, W, 3) uint8 — 카메라 concat 전의 원본."""
    idx = np.asarray(idx)
    if GI is not None:
        return GI[torch.as_tensor(idx, device=dev)].cpu().numpy()
    return np.asarray(imgs[idx])


def _plot_base(qv, vv, e, ok, step):
    """정적 Q·V 플롯을 RGB 배열로 굽고 프레임→픽셀 사상을 함께 돌려준다."""
    n = len(qv)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axs[0].plot(np.arange(n), qv, lw=1.4, color="tab:blue")
    axs[1].plot(np.arange(n), vv, lw=1.4, color="tab:orange")
    for ax, name in zip(axs, ("Q (min of ensemble)", "V")):
        ax.axhline(0, color="gray", lw=0.5)
        ax.axhline(1, color="gray", lw=0.5)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
    axs[1].set_xlabel("frame in episode")
    axs[0].set_title(f"ep {e}  {'SUCCESS' if ok else 'FAIL'}  step {step}  tau={a.expectile}")
    fig.tight_layout()
    fig.canvas.draw()
    base = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    hp = base.shape[0]
    x = np.arange(n)
    xs = axs[0].transData.transform(np.column_stack([x, np.zeros(n)]))[:, 0]
    qy = hp - axs[0].transData.transform(np.column_stack([x, qv]))[:, 1]
    vy = hp - axs[1].transData.transform(np.column_stack([x, vv]))[:, 1]
    top = hp - axs[0].get_window_extent().y1
    bot = hp - axs[1].get_window_extent().y0
    plt.close(fig)
    return base, xs, qy, vy, float(top), float(bot)


def write_videos(step, qc, vc):
    sel = eps if a.video_eps <= 0 else eps[:a.video_eps]
    t0 = time.time()
    for e, fr, ok in sel:
        n = len(fr)
        ks = np.arange(0, n, max(1, a.video_stride))
        base, xs, qy, vy, top, bot = _plot_base(qc[e], vc[e], e, ok, step)
        probe = _raw_cams(fr[ks[:1]])                       # (1, n_cams, H, W, 3)
        _, ncam, ih, iw, _ = probe.shape
        sw = iw * ncam                                      # 카메라 스트립 폭
        sc = sw / base.shape[1]
        base_r = cv2.resize(base, (sw, int(round(base.shape[0] * sc))))
        xs_r, qy_r, vy_r = xs * sc, qy * sc, vy * sc
        top_r, bot_r = int(top * sc), int(bot * sc)
        cw, ch = sw, ih + base_r.shape[0]
        tw, th = _m16(cw), _m16(ch)
        need = (tw, th) != (cw, ch)
        out = vids / f"{step:06d}_ep{e:04d}_{'succ' if ok else 'fail'}.mp4"
        with iio.get_writer(out, fps=a.video_fps, codec="libx264", quality=7,
                            macro_block_size=None) as w:
            for b0 in range(0, len(ks), 64):                # 이미지는 64프레임씩 읽는다
                blk = ks[b0:b0 + 64]
                raw = _raw_cams(fr[blk])
                for j, k in enumerate(blk):
                    strip = np.concatenate(list(raw[j]), axis=1)     # (H, W*n_cams, 3)
                    pf = base_r.copy()
                    px = int(round(xs_r[k]))
                    cv2.line(pf, (px, top_r), (px, bot_r), (40, 40, 40), 2)
                    cv2.circle(pf, (px, int(round(qy_r[k]))), 5, (214, 39, 40), -1)
                    cv2.circle(pf, (px, int(round(vy_r[k]))), 5, (214, 39, 40), -1)
                    comp = np.vstack([strip, pf])
                    w.append_data(cv2.resize(comp, (tw, th)) if need else comp)
    print(f"  [video] {len(sel)}개 → {vids}/  ({time.time()-t0:.0f}s)")


# --- 2. 학습 (IQL) -----------------------------------------------------------
enc = BatchEncoder(3 * mod.n_cams, cfg.latent_dim_image, cfg.encoder_stage_sizes,
                   cfg.encoder_num_filters).to(dev)
critic = CriticEnsemble(cfg.latent_dim_image, snorm.shape[1], (LAT + R) * mod.action_dim,
                        cfg.num_qs, cfg.latent_dim_state, cfg.include_state, cfg.hidden_dims,
                        cfg.critic_layer_norm).to(dev)
# --- distributional critic (HL-Gauss) 옵션 ---------------------------------
# 스칼라 회귀를 분류로 바꾼다: support [lo,hi] 를 bins 개로 자르고 bin logits 를 내게 한 뒤,
# 타깃 스칼라를 그 자리 가우시안으로 번진 soft label 로 CE 회귀한다 (PA-RL 기본).
#   · Q 는 support 를 벗어날 수 없다 → 발산 구조적 불가 (우리 음수 고정점 -0.013 이 표현 불가)
#   · V 는 스칼라로 둔다 (PA-RL 도 ValueCritic 은 스칼라, CE 는 critic 에만)
if a.bins:
    lo_q, hi_q = (float(x) for x in a.q_range.split(","))
    edges = torch.linspace(lo_q, hi_q, a.bins + 1, device=dev)
    centers = ((edges[:-1] + edges[1:]) / 2)
    sigma = 0.75 * (hi_q - lo_q) / a.bins            # PA-RL hl_gauss_transform 기본값
    for m in critic.qs:                              # 헤드만 bins 출력으로 교체
        m.head = xavier_(nn.Linear(m.body.out_dim, a.bins)).to(dev)

    def q_of(x):                                     # (..., bins) logits → 스칼라 기대값
        return (x.softmax(-1) * centers).sum(-1)

    def q_loss(x, y, w):                             # CE against HL-Gauss soft label
        z = (edges - y.clamp(lo_q, hi_q)[:, None]) / (sigma * 2 ** 0.5)
        cdf = 0.5 * (1 + torch.erf(z))
        pr = (cdf[:, 1:] - cdf[:, :-1])
        pr = pr / pr.sum(-1, keepdim=True).clamp_min(1e-8)
        return (-(pr * x.log_softmax(-1)).sum(-1) * w).mean()
    print(f"[critic] distributional: support [{lo_q},{hi_q}] bins {a.bins} sigma {sigma:.4f}")
else:
    q_of = lambda x: x
    q_loss = lambda x, y, w: (((x - y) ** 2) * w).mean()
    print("[critic] 스칼라 MSE")

target = copy.deepcopy(critic).requires_grad_(False)
# PA-RL 의 forward_target_critic 은 target_params 로 **인코더까지** 타깃을 쓴다
# (iql.py:173, grad_params=self.state.target_params). V 회귀 타깃이 그만큼 안정된다.
tenc = copy.deepcopy(enc).requires_grad_(False)
# V = **액션 입력 폭이 0인 critic** (앙상블 1). 구조/초기화를 Q 와 똑같이 두려고 재사용한다.
value = CriticEnsemble(cfg.latent_dim_image, snorm.shape[1], 0, 1, cfg.latent_dim_state,
                       cfg.include_state, cfg.hidden_dims, cfg.critic_layer_norm).to(dev)
# PA-RL 의 kernel_scale_final=1e-2 (+ orthogonal scale 1e-2) 에 대응 — 마지막 층을 작게 두어
# 출력이 0 근처에서 시작하게 한다. 없으면 V 가 4.07 에서 시작해 Q 타깃(r + γ^R·V)이 support
# 밖으로 나가 한동안 1.0 으로 클램프된다 (스모크 실측).
with torch.no_grad():
    for m in list(critic.qs) + list(value.qs):
        m.head.weight.mul_(1e-2)
        m.head.bias.zero_()
opt = torch.optim.Adam(list(critic.parameters()) + list(value.parameters())
                       + list(enc.parameters()), lr=cfg.critic_lr)
none_act = torch.zeros(cfg.batch_size, 0, device=dev)     # V 호출용 빈 액션
print(f"[학습] {a.steps} 스텝, batch {cfg.batch_size}, V 는 액션폭 0 critic")

for step in range(1, a.steps + 1):
    i = train[rng.integers(0, len(train), cfg.batch_size)]
    n = nstep(flat, i, R, cfg.discount)
    j = n["next_idx"]
    lat = enc(obs(i), stop_gradient=cfg.freeze_critic_encoder)
    with torch.no_grad():
        # Q 타깃: V(s') — 정책 액션이 필요 없다 (여기가 SARSA 와의 차이)
        nv = value(enc(obs(j), stop_gradient=True), st(j), none_act)[0]
        tq = (torch.from_numpy(n["reward"]).to(dev)
              + (cfg.discount ** R) * torch.from_numpy(n["mask"]).to(dev) * nv)
        # V 타깃: target critic 의 Q(s, a_data). 인코더도 **target 사본**(tenc) 을 쓴다 —
        # PA-RL 의 forward_target_critic 이 params 전체를 target_params 로 바꿔 부른다.
        mem = None if a.v_min == "all" else critic.subsample(cfg.num_min_qs, gen)
        qt = q_of(target(tenc(obs(i), stop_gradient=True), st(i), act(i),
                         members=mem)).min(dim=0).values
    valid = torch.from_numpy(n["valid"]).to(dev)
    ql = critic(lat, st(i), act(i))                   # bins 면 logits, 아니면 스칼라
    q = q_of(ql)
    v = value(lat, st(i), none_act)[0]
    loss_q = q_loss(ql, tq, valid)
    d = qt - v                                            # expectile 회귀
    loss_v = (torch.where(d > 0, a.expectile, 1 - a.expectile) * d ** 2).mean()
    loss = loss_q + loss_v
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():                            # polyak: critic + 인코더 둘 다
        for tp, pp in zip(list(target.parameters()) + list(tenc.parameters()),
                          list(critic.parameters()) + list(enc.parameters())):
            tp.mul_(1 - cfg.tau).add_(pp, alpha=cfg.tau)
    if step % 100 == 0 or step == a.steps:
        print(f"  step {step:5d}  q_loss {float(loss_q):.5f}  v_loss {float(loss_v):.5f}  "
              f"q {float(q.mean()):+.4f}  v {float(v.mean()):+.4f}")
    if wb is not None and (step % a.log_every == 0 or step == a.steps):
        wb.log({"loss/q": float(loss_q), "loss/v": float(loss_v),
                "loss/total": float(loss)}, step=step)

    # --- 평가 + 저장 --------------------------------------------------------
    if step % a.eval_every == 0 or step == a.steps:
        with torch.no_grad():
            def qv_at(idx, bs=256):
                """Q 와 V 를 같은 인코딩에서 한 번에 뽑는다 (비디오가 둘 다 쓴다)."""
                qq, vv = [], []
                for k in np.array_split(idx, max(1, len(idx) // bs)):
                    h = enc(obs(k), stop_gradient=True)
                    qq.append(q_of(critic(h, st(k), act(k))).min(0).values.float().cpu().numpy())
                    vv.append(value(h, st(k), torch.zeros(len(k), 0, device=dev))[0]
                              .float().cpu().numpy())
                return np.concatenate(qq), np.concatenate(vv)
            curves = {e: qv_at(fr) for e, fr, _ in eps}
        qc = {e: c[0] for e, c in curves.items()}
        vc = {e: c[1] for e, c in curves.items()}
        fin = np.array([qc[e][-1] for e, _, _ in eps])
        okm = np.array([o for _, _, o in eps])
        sq, fq = fin[okm], fin[~okm]
        auc = float((sq[:, None] > fq[None, :]).mean())
        print(f"  [eval] step {step:5d}  AUC {auc:.3f}  Q(성공끝) {sq.mean():+.3f}  "
              f"Q(실패끝) {fq.mean():+.3f}  Q범위 [{fin.min():+.3f},{fin.max():+.3f}]")
        fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for e, fr, o in eps:
            col = "tab:green" if o else "tab:red"
            axs[0].plot(np.arange(len(fr)), qc[e], lw=1, alpha=0.5, color=col)
            axs[1].plot(np.arange(len(fr)), vc[e], lw=1, alpha=0.5, color=col)
        for ax, name in zip(axs, ("Q (min of ensemble)", "V")):
            ax.axhline(0, color="gray", lw=0.5); ax.axhline(1, color="gray", lw=0.5)
            ax.set_ylabel(name); ax.grid(alpha=0.25)
        axs[1].set_xlabel("frame in episode")
        axs[0].set_title(f"IQL tau={a.expectile}{f' dist{a.bins}' if a.bins else ''}  "
                         f"step {step}  AUC {auc:.3f}  green=success")
        fig.tight_layout(); fig.savefig(plots / f"{step:06d}_qv.png", dpi=110); plt.close(fig)
        ck = run / f"critic_{step:06d}.pt"
        tmp = ck.with_suffix(".pt.tmp")
        torch.save({"enc": enc.state_dict(), "critic": critic.state_dict(),
                    "target": target.state_dict(), "value": value.state_dict(),
                    "tenc": tenc.state_dict(),
                    "step": step, "exp": a.exp, "seed": a.seed, "expectile": a.expectile,
                    "discount": cfg.discount, "num_qs": cfg.num_qs, "v_min": a.v_min,
                    "latency": LAT, "replan": R, "action_dim": mod.action_dim,
                    "state_dim": snorm.shape[1], "bins": a.bins, "tag": TAG,
                    "q_range": a.q_range if a.bins else None}, tmp)
        os.replace(tmp, ck)
        # 최신 포인터. 고정 이름을 원하는 다운스트림(probe_pairs 등)이 쓸 수 있게 둔다.
        lnk, ltmp = run / "critic_latest.pt", run / "critic_latest.pt.tmp"
        ltmp.unlink(missing_ok=True)
        ltmp.symlink_to(ck.name)                     # 상대 링크 — 디렉토리를 옮겨도 안 깨진다
        os.replace(ltmp, lnk)
        print(f"  [저장] {ck.name} ({ck.stat().st_size / 1e6:.0f}MB) → critic_latest.pt")
        if a.keep_last > 0:
            old_ck = sorted(run.glob("critic_[0-9]*.pt"))[:-a.keep_last]
            for f in old_ck:
                f.unlink()
            if old_ck:
                print(f"  [정리] 오래된 체크포인트 {len(old_ck)}개 삭제 (--keep-last {a.keep_last})")

        # 이 블록 자체가 --eval-every 주기로만 돈다. video_every=0 이면 매 평가마다,
        # N>0 이면 N 의 배수인 평가에서 굽는다 (N 이 eval_every 의 배수가 아니면 안 걸린다).
        if a.video_every >= 0 and (step % (a.video_every or a.eval_every) == 0
                                   or step == a.steps):
            write_videos(step, qc, vc)

if wb is not None:
    wb.finish()
print(f"[완료] {a.steps} 스텝 / 산출물 {run}")
