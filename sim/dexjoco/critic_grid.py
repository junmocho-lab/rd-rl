#!/usr/bin/env python3
"""홀드아웃 에피소드들을 **한 화면에 세로로 쌓아** 비교한다 (행마다 카메라 + Q 곡선).

에피소드를 따로따로 보면 Q 곡선의 높이 차이를 눈으로 못 비교한다. 여기서는 행마다
같은 y축(0~1)을 쓰고 각자의 길이대로 곡선을 그려서, "빨리 끝나는 에피소드가 처음부터
높은 Q 를 받는가" 를 한눈에 보게 한다.

  파란 실선  critic Q(s_t, 로그된 A_t)
  회색 점선  참값 γ^(T-t)          (성공만 학습했으면 이것이 Q 의 목표다)
  빨간 커서  현재 프레임

에피소드마다 길이가 다르므로 짧은 것은 끝나면 마지막 프레임에서 멈춘다 (커서도 정지).

  python sim/dexjoco/critic_grid.py --set holdout --success-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="dexjoco_hammer_nail_d2r8_s0")
p.add_argument("--data", type=Path, default=REPO / "rl-dataset/dexjoco/hammer_nail_d2r8_s0")
p.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
p.add_argument("--critic", default="fixed_successonly/critic_latest.pt")
p.add_argument("--out", type=Path,
               default=Path("/workspace/junmo_cho/dexjoco/critic_grid/holdout_success.mp4"))
p.add_argument("--holdout", default="0.1")
p.add_argument("--set", default="holdout", choices=("all", "train", "holdout"))
p.add_argument("--success-only", action="store_true")
p.add_argument("--max-rows", type=int, default=8,
               help="한 화면에 쌓을 행 수 상한. 길이 분포에 걸쳐 고르게 뽑는다 — "
                    "빠른 성공과 느린 성공이 같이 있어야 'Q 가 완주 속도를 반영하는가' 를 본다")
p.add_argument("--cam", type=int, default=128, help="행당 카메라 픽셀")
p.add_argument("--fps", type=int, default=30)
p.add_argument("--device", default="cuda")
a = p.parse_args()
dev = a.device

from rl.critic_io import load_stepwise_critic  # noqa: E402
from rl.data import build_flat, find_sessions, resolve_modality  # noqa: E402
from rl.expo import ExpoConfig  # noqa: E402
from rl.vla_rldx import load_state_action_processor, normalize_states  # noqa: E402

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
R, LAT, G = exp["replan_steps"], exp["inference_latency"], float(cfg.discount)
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / exp["base_policy"]
sessions = find_sessions(a.data)
mod, _ = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
flat = build_flat(sessions, mod)
norm = np.load(work / "actnorm.npy", mmap_mode="r")
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)

ck = Path(a.critic)
if not ck.is_file():
    ck = work / a.critic
SC = load_stepwise_critic(ck, work, snorm, dev=dev)

eps = [(e, np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode)]
eps = [(e, fr, bool(flat.is_success[fr[-1]])) for e, fr in eps]
frac = float(a.holdout)
every = max(2, int(round(1 / frac)))
hold_ids = set(np.unique(flat.episode)[::every].tolist())
pick = [x for x in eps if a.set == "all" or (a.set == "holdout") == (x[0] in hold_ids)]
if a.success_only:
    pick = [x for x in pick if x[2]]
pick = sorted(pick, key=lambda x: len(x[1]))          # 짧은 것(=빠른 성공)부터 위로
if len(pick) > a.max_rows:                           # 길이 분포에 걸쳐 고르게
    idx = np.linspace(0, len(pick) - 1, a.max_rows).round().astype(int)
    pick = [pick[i] for i in sorted(set(idx.tolist()))]
print(f"[홀드아웃] {sorted(hold_ids)}  (학습에 쓰지 않은 에피소드)")
print(f"[대상] {[(int(e), len(fr)) for e, fr, _ in pick]}")

sess = sessions[0]
# flat.episode 는 parquet 의 episode_index 를 그대로 쓴다 (연속이 아닐 수 있다 —
# 부분집합 세션은 원본 번호를 유지한 채 심링크만 건다). 그러므로 flat 번호가 곧 원본 번호다.
# 1000개마다 chunk 가 갈리므로 비디오 경로의 chunk 는 번호에서 계산한다.
def vpath(i, cam="camera_front"):
    return sess / f"videos/chunk-{i//1000:03d}/observation.images.{cam}/episode_{i:06d}.mp4"


rows = []
for e, fr, o in pick:
    with torch.no_grad():
        st = torch.from_numpy(snorm[fr]).to(dev)
        act = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[fr])[:, :LAT + R].reshape(len(fr), -1))).to(dev)
        q = SC.q_all(SC.latent_of(fr, st), act).min(0).values.sum(-1).float().cpu().numpy()
    truth = (G ** (len(fr) - np.arange(len(fr)))) if o else np.zeros(len(fr))
    src = int(e)
    frames = imageio.mimread(str(vpath(src)), memtest=False)
    s = max(1, frames[0].shape[0] // a.cam)
    frames = [f[..., :3][::s, ::s][:a.cam, :a.cam] for f in frames]
    rows.append(dict(src=src, q=q, truth=truth, frames=frames, n=len(fr), ok=o))

T = max(r["n"] for r in rows)
PW, PH = 640, a.cam
bases, spans = [], []
for r in rows:
    fig, ax = plt.subplots(figsize=(PW / 100, PH / 100), dpi=100)
    ax.plot(r["truth"], color="0.55", lw=1.4, ls="--")
    ax.plot(r["q"], color="tab:blue", lw=1.8)
    ax.set_xlim(0, T - 1)                    # **모든 행이 같은 x 축** — 길이 비교가 되게
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.25)
    ax.text(0.012, 0.93, f"ep{r['src']}  {r['n']}f   Q0={r['q'][0]:.3f}  "
                         f"true0={r['truth'][0]:.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", ec="none", alpha=.75))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.99, bottom=0.13)
    fig.canvas.draw()
    bases.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    bb = ax.get_position()
    spans.append((int(bb.x0 * PW), int(bb.x1 * PW)))
    plt.close(fig)

a.out.parent.mkdir(parents=True, exist_ok=True)
w = imageio.get_writer(str(a.out), fps=a.fps, codec="libx264", pixelformat="yuv420p",
                       macro_block_size=1, output_params=["-crf", "20"])
for t in range(T):
    strip = []
    for r, bimg, (x0, x1) in zip(rows, bases, spans):
        k = min(t, r["n"] - 1)
        panel = bimg.copy()
        cx = int(x0 + (x1 - x0) * k / max(T - 1, 1))
        panel[:, max(cx - 1, 0):cx + 2] = np.array([220, 40, 40], np.uint8)
        cam = r["frames"][min(k, len(r["frames"]) - 1)]
        if cam.shape[0] != PH:
            cam = np.pad(cam, ((0, PH - cam.shape[0]), (0, 0), (0, 0)), constant_values=255)
        strip.append(np.hstack([cam, panel]))
    w.append_data(np.vstack(strip).astype(np.uint8))
w.close()
print(f"\n[비디오] {a.out}  ({T}프레임, {len(rows)}행)")
print("  파란 실선 = critic Q,  회색 점선 = 참값 γ^(T-t),  빨간 세로선 = 현재 프레임")
print("  x 축은 모든 행이 동일하다 — 짧은 에피소드는 먼저 끝나고 그 자리에서 멈춘다")
