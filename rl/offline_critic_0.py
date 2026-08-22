#!/usr/bin/env python3
"""critic 오프라인 학습 최소판 — 데이터 로드 / TD 학습 / 저장, 세 단계뿐.

진단·게이트·플롯은 rl/offline_critic.py 에 있다. 여기는 "학습해서 저장" 만 한다.

--checkpoints 는 체크포인트 루트 하나다. base 정책을 여기서 찾고(exp yaml 의
base_policy 가 이 아래 상대경로), 산출물도 여기 <exp>-critic/ 에 쓴다:
    images.mm(.json)  디코딩된 프레임      actnorm.npy  정규화된 액션 청크 (둘 다 캐시)
    critic.pt         enc/critic/target
    eval/<step>_q.png                held-out 에피소드별 Q 곡선 (시간축 = 프레임 그대로)
    eval/<step>_ep<번호>_<성패>.mp4   카메라 + Q 곡선 (성공 3 / 실패 3, H.264)

usage:
  source configs/kakao_path.sh
  PYTHONPATH="$L_PYTHONPATH" $L_PY -u -m rl.offline_critic_0 --exp openarm_rim --data $L_DS/0815_openarm_rh56f1_inference --checkpoints $L_CKPT
  PYTHONPATH="$L_PYTHONPATH" $L_PY -u -m rl.offline_critic_0 --exp fuji --data $L_DS/fuji-rl-dataset --checkpoints $L_CKPT/temp --holdout 0.2

**-u 를 붙일 것.** 리다이렉트하면 stdout 이 블록 버퍼링되어 돌고 있는데도 얼어붙어 보인다.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import imageio
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402  (Agg 를 먼저 잡아야 한다)

from rl.data import (build_flat, build_images, find_sessions, nstep, open_images,
                     resolve_modality)
from rl.expo import ExpoConfig
from rl.nets import BatchEncoder, CriticEnsemble
from rl.offline_critic import normalize_all
from rl.vla_rldx import RLDXVLA, load_state_action_processor, normalize_states

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="openarm_rim")
p.add_argument("--data", type=Path, required=True)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval-every", type=int, default=1000)
p.add_argument("--holdout", default="eval",
               help="세션 이름에 이 문자열이 있으면 평가 전용. 숫자(예: 0.2)를 주면 그 비율의 "
                    "에피소드를 균등 간격으로 뺀다 (이름에 eval 이 없는 데이터용)")
a = p.parse_args()

# 재현성: 안 걸면 critic 초기값이 실행마다 달라져 같은 스텝에서 Q 가 크게 다르다.
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
gen = torch.Generator().manual_seed(a.seed)
assert torch.cuda.is_available(), "GPU 전용이다"
dev = "cuda"

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
base = a.checkpoints / exp["base_policy"]
work = a.checkpoints / f"{a.exp}-critic"
work.mkdir(parents=True, exist_ok=True)

# --- 1. 데이터 ---------------------------------------------------------------
mod, src = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
print(f"[exp] {a.exp} replan={R} latency={LAT} horizon={H}\n[modality] {src}")
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
build_images(sessions, flat, work / "images.mm", mod)
imgs, _ = open_images(work / "images.mm")  # (N, n_cam, 192, 320, 3) e.g. (64619, 2, 192, 320, 3)

# critic 의 액션/상태는 롤아웃 때와 같은 모델 공간이어야 한다 (rl/vla_rldx.normalize_states).
# 상태는 processor 만으로 되고(가중치 불필요) 벡터 연산 한 번이라 캐시하지 않는다.
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
if not (work / "actnorm.npy").is_file():             # 액션 청크는 비싸다 → VLA + 캐시
    vla = RLDXVLA(base, mod, RLDX, exp["rldx_data_config"], device=dev)
    normalize_all(vla, flat, H, cache=work / "actnorm.npy")
    del vla
    torch.cuda.empty_cache()
norm = normalize_all(None, flat, H, cache=work / "actnorm.npy")
# held-out 분리 — AUC 를 학습 프레임에서 재면 암기력을 재는 것이 된다
# 0<x<1 만 비율로 본다 — 세션 이름이 타임스탬프(170641)처럼 숫자뿐일 수 있다
frac = float(a.holdout) if a.holdout.replace(".", "", 1).isdigit() else 0.0
if 0 < frac < 1:                                     # 에피소드 비율로 뺀다
    every = max(2, int(round(1 / frac)))
    hold = np.isin(flat.episode, np.unique(flat.episode)[::every])
    how = f"에피소드 {every}개마다 1개"
else:                                                # 세션 이름 부분 매칭
    sel = [i for i, n in enumerate(flat.sessions) if a.holdout and a.holdout in n]
    hold = np.isin(flat.session, sel)
    how = f"세션 이름에 '{a.holdout}' 포함 = {[flat.sessions[i] for i in sel]}"
train = np.flatnonzero(~hold[:len(flat) - R])        # 뒤 R 프레임은 n-step 창이 안 찬다
eps = [(e, np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode[hold])]
eps = [(e, fr, bool(flat.is_success[fr[-1]])) for e, fr in eps]
vids = [x for x in eps if x[2]][:3] + [x for x in eps if not x[2]][:3]   # 비디오: 성공/실패 3개씩
n_ok = sum(o for _, _, o in eps)
print(f"[데이터] 세션 {len(sessions)} / 프레임 {len(flat)} / 학습 {len(train)} / "
      f"state {flat.state.shape[1]}→{snorm.shape[1]}차원")
print(f"[평가셋] {how} → 에피소드 {len(eps)} (성공 {n_ok} / 실패 {len(eps) - n_ok})")
if not n_ok or n_ok == len(eps):                     # 500스텝 뒤에 죽지 말고 지금 죽는다
    raise SystemExit("평가셋에 성공/실패가 한쪽뿐이라 AUC 를 못 낸다 — --holdout 을 바꿀 것 "
                     "(세션 이름 문자열, 또는 0.2 처럼 비율)")

def obs(i):                                          # (B, H, W, 3*n_cams) — 카메라를 채널로
    x = np.asarray(imgs[i])                          # memmap fancy-index 는 한 번만
    return torch.from_numpy(np.concatenate([x[:, c] for c in range(x.shape[1])], -1)).to(dev)

# critic 이 보는 액션 = 청크 [0, LAT+R) — prefix(이미 커밋된 LAT 스텝) + 새로 커밋하는
# R 스텝. 결정 이후 실제로 실행되는 전부이고, 보상 창 [t, t+R) 을 일으킨 액션이 다 들어온다.
act = lambda i: torch.from_numpy(norm[i][:, :LAT + R].reshape(len(i), -1)).to(dev)
st = lambda i: torch.from_numpy(snorm[i]).to(dev)

# --- 2. 학습 (behavior policy 의 Q^pi 를 TD 로) ------------------------------
enc = BatchEncoder(3 * mod.n_cams, cfg.latent_dim_image, cfg.encoder_stage_sizes,
                   cfg.encoder_num_filters).to(dev)
critic = CriticEnsemble(cfg.latent_dim_image, snorm.shape[1], (LAT + R) * mod.action_dim,
                        cfg.num_qs,
                        cfg.latent_dim_state, cfg.include_state, cfg.hidden_dims,
                        cfg.critic_layer_norm).to(dev)
target = copy.deepcopy(critic).requires_grad_(False)
opt = torch.optim.Adam(list(critic.parameters()) + list(enc.parameters()), lr=cfg.critic_lr)
print(f"[학습] {a.steps} 스텝, batch {cfg.batch_size}")


# --- 평가: held-out AUC + 에피소드별 Q 곡선/비디오 ---------------------------
ev = work / "eval"
ev.mkdir(exist_ok=True)

@torch.no_grad()
def q_at(idx, bs=256):                               # 앙상블 10개의 min (argmax 때와 같은 비관)
    return np.concatenate([critic(enc(obs(j), stop_gradient=True), st(j), act(j))
                           .min(0).values.float().cpu().numpy()
                           for j in np.array_split(idx, max(1, len(idx) // bs))])

def video(path, fr, q, title, fps=20, ph=170, hd=24):
    """카메라 프레임 + Q 곡선. 시간축은 프레임 그대로 (정규화 없음).

    축·격자·기준선은 matplotlib 로 **에피소드당 한 번** 렌더하고, 프레임마다 지나온
    구간과 커서만 그 위에 덧그린다 (프레임마다 savefig 하면 수천 장이라 느리다).

    코덱은 libx264 (imageio-ffmpeg 의 정적 ffmpeg). cv2 는 쓰지 않는다 — 번들 ffmpeg 에
    H.264 인코더가 없어 avc1/H264 가 0바이트를 만들고, 유일하게 되는 mp4v(MPEG-4 Part 2)
    는 브라우저·IDE 에서 재생되지 않는다.
    """
    x0 = np.asarray(imgs[fr[0]])                     # (n_cams, H, W, 3)
    Hc, W = x0.shape[1], x0.shape[2] * x0.shape[0]
    fig = plt.figure(figsize=(W / 100, ph / 100), dpi=100)
    ax = fig.add_axes([0.055, 0.21, 0.935, 0.75])
    ax.plot(q, color="0.8", lw=1.2)                                    # 전체 곡선 (미리보기)
    ax.axhline(0.0, color="0.35", lw=0.8, ls="--")                     # 실패 기준
    ax.axhline(1.0, color="tab:blue", lw=0.8, ls="--")                 # 성공 상한
    ax.set_xlim(0, max(1, len(q) - 1))
    ax.set_ylim(min(-0.25, float(q.min()) - 0.05), max(1.1, float(q.max()) + 0.05))
    ax.set_yticks([0.0, 0.5, 1.0]); ax.grid(alpha=0.25)
    ax.set_xlabel("frame in episode", fontsize=8); ax.tick_params(labelsize=8)
    fig.canvas.draw()
    base = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()        # RGB (ph, W, 3)
    xy = ax.transData.transform(np.c_[np.arange(len(q)), q])           # 데이터 → 픽셀
    pts = np.stack([xy[:, 0], ph - xy[:, 1]], axis=1).astype(np.int32)
    plt.close(fig)

    vw = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                            macro_block_size=1, pixelformat="yuv420p")
    for t in range(len(fr)):
        pan = base.copy()                                              # 색은 RGB 순으로 준다
        cv2.polylines(pan, [pts[:t + 1]], False, (0, 140, 0), 2)       # 지나온 구간
        cv2.circle(pan, tuple(pts[t]), 4, (200, 0, 0), -1)             # 현재 지점
        head = np.full((hd, W, 3), 255, np.uint8)
        cv2.putText(head, f"{title}  t={t}/{len(fr) - 1}  Q={q[t]:+.3f}", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cams = np.concatenate(list(np.asarray(imgs[fr[t]])), axis=1)    # 카메라 가로 concat
        vw.append_data(np.concatenate([head, cams, pan], axis=0))
    vw.close()


def evaluate(step):
    q = {e: q_at(fr) for e, fr, _ in eps}
    fin = np.array([q[e][-1] for e, _, _ in eps])                # 에피소드 마지막 프레임의 Q
    okm = np.array([o for _, _, o in eps])
    sq, fq = fin[okm], fin[~okm]
    auc = float((sq[:, None] > fq[None, :]).mean()) if len(sq) and len(fq) else float("nan")
    print(f"  [eval] step {step:5d}  AUC {auc:.3f}  Q(성공끝) {sq.mean():+.3f}  "
          f"Q(실패끝) {fq.mean():+.3f}  Q범위 [{fin.min():+.3f},{fin.max():+.3f}]")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for e, fr, o in eps:
        ax.plot(np.arange(len(fr)), q[e], lw=1, alpha=0.5,
                color="tab:green" if o else "tab:red")
    ax.axhline(0, color="gray", lw=0.5); ax.axhline(1, color="gray", lw=0.5)
    ax.set_xlabel("frame in episode (정규화 없음)"); ax.set_ylabel("Q (min of ensemble)")
    ax.set_title(f"step {step}  AUC {auc:.3f}   green=success  red=failure")
    fig.tight_layout(); fig.savefig(ev / f"{step:06d}_q.png", dpi=110); plt.close(fig)
    for e, fr, o in vids:
        tag = f"ep{e:04d}_{'succ' if o else 'fail'}"
        video(ev / f"{step:06d}_{tag}.mp4", fr, q[e], f"step {step}  {tag}")

for step in range(1, a.steps + 1):
    i = train[rng.integers(0, len(train), cfg.batch_size)]
    n = nstep(flat, i, R, cfg.discount)
    j = n["next_idx"]
    with torch.no_grad():                            # next_action 은 로그된 액션 (SARSA)
        nq = target(enc(obs(j), stop_gradient=True), st(j), act(j),
                    members=critic.subsample(cfg.num_min_qs, gen)).min(dim=0).values
        tq = (torch.from_numpy(n["reward"]).to(dev)
              + (cfg.discount ** R) * torch.from_numpy(n["mask"]).to(dev) * nq)
    q = critic(enc(obs(i), stop_gradient=cfg.freeze_critic_encoder), st(i), act(i))
    loss = (((q - tq) ** 2) * torch.from_numpy(n["valid"]).to(dev)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():                            # target polyak
        for tp, pp in zip(target.parameters(), critic.parameters()):
            tp.mul_(1 - cfg.tau).add_(pp, alpha=cfg.tau)
    if step % 100 == 0 or step == a.steps:
        print(f"  step {step:5d}  loss {float(loss):.5f}  q {float(q.mean()):+.4f}")
    if step % a.eval_every == 0 or step == a.steps:
        evaluate(step)

# --- 3. 저장 -----------------------------------------------------------------
torch.save({"enc": enc.state_dict(), "critic": critic.state_dict(),
            "target": target.state_dict(), "step": a.steps, "exp": a.exp, "seed": a.seed},
           work / "critic.pt")
print(f"[저장] {work / 'critic.pt'}")
