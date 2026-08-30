#!/usr/bin/env python3
"""qvgm critic 의 Q 를 에피소드 영상 위에 얹어 본다 (offline_iql 의 videos/ 와 같은 취지).

offline_iql_qvgm.py 는 플롯만 만들고 비디오를 안 만든다. 눈으로 확인해야 알 수 있는 것들이
있어서 (Q 가 언제 오르는지, 타격 순간에 반응하는지, 실패에서 어떻게 되는지) 따로 만든다.

왼쪽 = front 카메라, 오른쪽 = Q(s_t, 로그된 A_t) 곡선 + 현재 위치 커서.
성공만으로 학습한 critic 이면 Q 의 참값이 γ^(T-t) 이므로 **끝으로 갈수록 1 에 수렴**해야 하고,
빨리 끝나는 에피소드일수록 처음부터 높아야 한다.

  python sim/dexjoco/critic_video.py --critic fixed_successonly/critic_latest.pt --n 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="dexjoco_hammer_nail_d2r8_s0")
p.add_argument("--data", type=Path, default=REPO / "rl-dataset/dexjoco/hammer_nail_d2r8_s0")
p.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
p.add_argument("--critic", default="fixed_successonly/critic_latest.pt")
p.add_argument("--out", type=Path, default=Path("/workspace/junmo_cho/dexjoco/critic_video"))
p.add_argument("--n", type=int, default=6, help="가장 빠른 성공 n/2 + 가장 느린/실패 n/2")
p.add_argument("--holdout", default="0.1",
               help="학습 때와 같은 규칙으로 홀드아웃을 재현한다 (에피소드 round(1/x) 개마다 1개)")
p.add_argument("--set", default="mixed", choices=("mixed", "all", "train", "holdout"),
               help="mixed=빠른성공+느린실패 섞어서, 나머지는 해당 집합 전부")
p.add_argument("--success-only", action="store_true", help="성공 에피소드만")
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
frac = float(a.holdout) if a.holdout.replace(".", "", 1).isdigit() else 0.0
every = max(2, int(round(1 / frac))) if 0 < frac < 1 else 0
hold_ids = set(np.unique(flat.episode)[::every].tolist()) if every else set()
if a.set == "mixed":
    ok = sorted([x for x in eps if x[2]], key=lambda x: len(x[1]))
    bad = sorted([x for x in eps if not x[2]], key=lambda x: -len(x[1]))
    pick = ok[: a.n // 2] + (bad[: a.n - a.n // 2] if bad else ok[-(a.n - a.n // 2):])
else:
    pick = [x for x in eps
            if a.set == "all"
            or (a.set == "holdout") == (x[0] in hold_ids)]
    if a.success_only:
        pick = [x for x in pick if x[2]]
    pick = sorted(pick, key=lambda x: len(x[1]))
print(f"[홀드아웃] 규칙 = 에피소드 {every}개마다 1개 → {sorted(hold_ids)}")
print(f"[대상] {a.set}"
      f"{' 성공만' if a.success_only else ''}: "
      f"{[(int(e), len(fr), '성공' if o else '실패') for e, fr, o in pick]}")

a.out.mkdir(parents=True, exist_ok=True)
sess = sessions[0]
ep_meta = {j["episode_index"]: j for j in
           (json.loads(l) for l in (sess / "meta/episodes.jsonl").read_text().splitlines() if l.strip())}
# flat 의 에피소드 번호 -> 원본 파일 번호
# flat.episode 는 parquet 의 episode_index 를 그대로 쓴다 (연속이 아닐 수 있다 —
# 부분집합 세션은 원본 번호를 유지한 채 심링크만 건다). 그러므로 flat 번호가 곧 원본 번호다.
# 1000개마다 chunk 가 갈리므로 비디오 경로의 chunk 는 번호에서 계산한다.
def vpath(i, cam="camera_front"):
    return sess / f"videos/chunk-{i//1000:03d}/observation.images.{cam}/episode_{i:06d}.mp4"


for e, fr, o in pick:
    with torch.no_grad():
        st = torch.from_numpy(snorm[fr]).to(dev)
        act = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[fr])[:, :LAT + R].reshape(len(fr), -1))).to(dev)
        q = SC.q_all(SC.latent_of(fr, st), act).min(0).values.sum(-1).float().cpu().numpy()
    truth = G ** (len(fr) - np.arange(len(fr))) if o else np.zeros(len(fr))
    src = int(e)
    frames = imageio.mimread(str(vpath(src)), memtest=False)

    fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=110)
    ax.plot(q, color="tab:blue", lw=1.6, label="Q(s,A) critic")
    ax.plot(truth, color="tab:gray", lw=1.2, ls="--", label="true gamma^(T-t)")
    ax.axhline(0, color="k", lw=.4); ax.axhline(1, color="k", lw=.4)
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, max(len(q) - 1, 1))
    ax.set_title(f"ep{src}  {'SUCCESS' if o else 'FAIL'}  {len(fr)}f", fontsize=9)
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.canvas.draw()
    base_img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    x0, x1 = 52, base_img.shape[1] - 12          # 축 영역 대략값 (커서용)

    dst = a.out / f"ep{src:03d}_{'succ' if o else 'fail'}_{len(fr)}f_q.mp4"
    w = imageio.get_writer(str(dst), fps=a.fps, codec="libx264",
                           pixelformat="yuv420p", macro_block_size=1,
                           output_params=["-crf", "20"])
    H = max(base_img.shape[0], frames[0].shape[0])
    for t in range(min(len(frames), len(q))):
        panel = base_img.copy()
        cx = int(x0 + (x1 - x0) * t / max(len(q) - 1, 1))
        panel[:, max(cx - 1, 0):cx + 2] = np.array([220, 40, 40], np.uint8)
        cam = frames[t][..., :3]
        pad = lambda im: np.pad(im, ((0, H - im.shape[0]), (0, 0), (0, 0)), constant_values=255)
        w.append_data(np.hstack([pad(cam), pad(panel)]).astype(np.uint8))
    w.close()
    print(f"  {dst}  (Q 처음 {q[0]:.3f} 끝 {q[-1]:.3f} / 참값 끝 {truth[-1]:.3f})")

print(f"\n[비디오] {a.out}")
