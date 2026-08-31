#!/usr/bin/env python3
"""홀드아웃 에피소드를 **한 화면에 세로로 쌓은** 비디오로 만든다 (행마다 카메라 + Q 곡선).

에피소드를 따로따로 보면 Q 곡선의 높이 차이를 눈으로 못 비교한다. 여기서는 행마다
같은 x·y 축을 쓰고 각자의 길이대로 곡선을 그려서, "빨리 끝나는 에피소드가 처음부터
높은 Q 를 받는가" 를 한눈에 보게 한다. --critic-steps 로 체크포인트를 여러 개 주면
한 행 안에 색으로 겹쳐 그린다 — 학습이 진행되며 Q 가 어떻게 변했는지가 보인다.

  색 있는 실선  critic Q(s_t, 로그된 A_t)   (체크포인트 스텝마다 다른 색)
  회색 점선     참값 γ^(T-t)                (성공만 학습했으면 이것이 Q 의 목표다)
  빨간 커서     현재 프레임                  (에피소드가 끝나면 그 자리에 멈춘다)

sim/dexjoco/critic_grid.py 에서 갈라져 나왔다. fuji 는 실기 태스크라 sim/ 아래가 아니다.
그 판에서 고친 것:

  1. 카메라 이름이 "camera_front" 로 하드코딩돼 있었다. rby1m_rh56f1 에는 그 카메라가
     없다 (camera_ego_left / camera_ego_right / wrist_left) — 잡 713/714 의 마지막
     단계가 여기서 FileNotFoundError 로 죽었다. 이제 modality 에서 고른다.
  2. `sess = sessions[0]` 로 **첫 세션에만** 비디오를 찾았다. 세션이 여러 개인 루트
     (fuji 는 7개)에서는 전역 에피소드 번호를 첫 세션 경로에 그대로 붙여
     "20 에피소드짜리 세션에서 episode_000260" 을 찾는다. Flat 이 그 매핑을 이미
     들고 있다 (rl/data.py:261):
         세션 = flat.session[t],  세션 안 번호 = flat.episode[t] - flat.ep_offset[세션]
  3. 카메라를 **크롭**했다 (`[:128, :128]`). fuji 는 192x320 이라 왼쪽 위 귀퉁이만
     남았다. 이제 종횡비를 지켜 리사이즈한다.
  4. 행마다 x 축 눈금 라벨이 아래 행 그림과 겹쳤다. 이제 맨 아래 행만 라벨을 단다.

사용:
  PY=third_party/RLDX-1/.venv/bin/python
  PYTHONPATH=third_party/RLDX-1:. $PY utils/critic_grid.py \
      --exp fuji_d3r8 --tag all \
      --critic-steps 1000,5000,20000,50000,100000,200000 --success-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 라벨에 한글이 있다. 없으면 두부(□)로 나오므로 CJK 폰트를 명시한다.
for _f in ("Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="fuji_d3r8")
p.add_argument("--data", type=Path, default=None, help="비우면 exp yaml 의 dataset")
p.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
p.add_argument("--tag", default="all", help="<exp>-critic/<tag>/ (success | all)")
p.add_argument("--critic-steps", default="1000,5000,20000,50000,100000,200000",
               help="쉼표 구분. critic_<step:06d>.pt 를 찾는다. 'latest' 도 쓸 수 있다")
p.add_argument("--out-dir", type=Path, default=None,
               help="비우면 <exp>-critic/<tag>/plots/grid/")
p.add_argument("--holdout", default="0.1")
p.add_argument("--set", default="holdout", choices=("all", "train", "holdout"))
p.add_argument("--success-only", action="store_true",
               help="그림 행을 성공만으로 (AUC 는 성공/실패 전체로 계산한다)")
p.add_argument("--max-rows", type=int, default=8,
               help="한 화면에 쌓을 행 수 상한. 길이 분포에 걸쳐 고르게 뽑는다 — "
                    "빠른 성공과 느린 성공이 같이 있어야 'Q 가 완주 속도를 반영하는가' 를 본다")
p.add_argument("--cam-name", default="", help="modality 의 카메라명. 비우면 첫 번째")
p.add_argument("--row-h", type=int, default=150, help="행 높이(px). 카메라는 종횡비를 지켜 맞춘다")
p.add_argument("--plot-w", type=int, default=900, help="Q 곡선 폭(px)")
p.add_argument("--fps", type=int, default=30)
p.add_argument("--combined-only", action="store_true",
               help="체크포인트를 겹친 비디오 하나만 만든다 (기본은 스텝별 파일도 함께)")
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
data = a.data or (REPO / exp["dataset"])
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / exp["base_policy"]
run = work / a.tag
out_dir = a.out_dir or (run / "plots/grid")
out_dir.mkdir(parents=True, exist_ok=True)

sessions = find_sessions(data)
mod, _ = resolve_modality(data, None, RLDX, exp["rldx_data_config"], base)
flat = build_flat(sessions, mod)
norm = np.load(work / "actnorm.npy", mmap_mode="r")
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)

cams = [c for c, _ in mod.video]
cam_name = a.cam_name or cams[0]
if cam_name not in cams:
    raise SystemExit(f"카메라 '{cam_name}' 가 modality 에 없다. 가능: {cams}")
cam_key = dict(mod.video)[cam_name]
print(f"[데이터] 세션 {len(sessions)} / 에피소드 {len(flat.ep_length)} / 프레임 {len(flat)}")
print(f"[카메라] {cam_name} -> {cam_key}   (가능: {cams})")


def vpath(gid: int) -> Path:
    """전역 에피소드 번호 -> 그 세션의 mp4 경로 (rl/data.py:261 의 매핑)."""
    fr0 = int(np.flatnonzero(flat.episode == gid)[0])
    si = int(flat.session[fr0])
    local = gid - flat.ep_offset[si]
    return (sessions[si] / f"videos/chunk-{local // 1000:03d}"
            / cam_key / f"episode_{local:06d}.mp4")


# ── 홀드아웃 분할 (offline_iql_qvgm.py 와 같은 규칙) ─────────────────────────
eps_all = [(e, np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode)]
eps_all = [(e, fr, bool(flat.is_success[fr[-1]])) for e, fr in eps_all]
frac = float(a.holdout)
every = max(2, int(round(1 / frac)))
hold_ids = set(np.unique(flat.episode)[::every].tolist())
sel = [x for x in eps_all if a.set == "all" or (a.set == "holdout") == (x[0] in hold_ids)]
# AUC 는 성공/실패가 모두 있어야 계산된다 — 그림 행이 성공만이어도 AUC 는 전체로 잰다.
pick = [x for x in sel if x[2]] if a.success_only else list(sel)
pick = sorted(pick, key=lambda x: len(x[1]))          # 짧은 것(=빠른 성공)부터 위로
if len(pick) > a.max_rows:                            # 길이 분포에 걸쳐 고르게
    idx = np.linspace(0, len(pick) - 1, a.max_rows).round().astype(int)
    pick = [pick[i] for i in sorted(set(idx.tolist()))]
n_ok = sum(o for _, _, o in sel)
print(f"[{a.set}] {len(sel)} 에피소드 (성공 {n_ok} / 실패 {len(sel) - n_ok})")
print(f"[대상 행] {[(int(e), len(fr)) for e, fr, _ in pick]}")


def q_of(SC, fr: np.ndarray):
    """Q(s_t, 로그된 A_t) 와 앙상블 std 를 에피소드 전 프레임에 대해."""
    with torch.no_grad():
        st = torch.from_numpy(snorm[fr]).to(dev)
        act = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[fr])[:, :LAT + R].reshape(len(fr), -1))).to(dev)
        qa = SC.q_all(SC.latent_of(fr, st), act)        # (num_qs, B, n_steps)
        q = qa.min(0).values.sum(-1).float().cpu().numpy()
        std = qa.sum(-1).std(0).float().cpu().numpy()   # 앙상블 불일치 (OOD 신호)
    return q, std


# ── 체크포인트별로 Q 를 모은다 ────────────────────────────────────────────────
steps = [s.strip() for s in a.critic_steps.split(",") if s.strip()]
curves: dict[str, dict] = {}
for s in steps:
    ck = run / ("critic_latest.pt" if s == "latest" else f"critic_{int(s):06d}.pt")
    if not ck.is_file():
        print(f"[건너뜀] {ck.name} 가 없다")
        continue
    SC = load_stepwise_critic(ck, work, snorm, dev=dev)
    qs = {int(e): q_of(SC, fr)[0] for e, fr, _ in pick}
    fin, okm, stds = [], [], []
    for e, fr, o in sel:                                # AUC 는 홀드아웃 전체로
        q, sd = q_of(SC, fr)
        fin.append(q[-1]); okm.append(o); stds.append(float(sd.mean()))
    fin, okm = np.asarray(fin), np.asarray(okm, bool)
    auc = (float((fin[okm][:, None] > fin[~okm][None, :]).mean())
           if okm.any() and (~okm).any() else float("nan"))
    curves[s] = dict(q=qs, auc=auc,
                     q_ok=float(fin[okm].mean()) if okm.any() else float("nan"),
                     q_ng=float(fin[~okm].mean()) if (~okm).any() else float("nan"),
                     ens_std=float(np.mean(stds)))
    del SC
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

if not curves:
    raise SystemExit(f"{run} 에서 체크포인트를 하나도 못 찾았다")

print(f"\n{'step':>8s} {'AUC':>7s} {'Q(성공끝)':>10s} {'Q(실패끝)':>10s} {'간격':>7s} {'앙상블std':>10s}")
print("-" * 60)
for s, c in curves.items():
    print(f"{s:>8s} {c['auc']:7.3f} {c['q_ok']:10.3f} {c['q_ng']:10.3f} "
          f"{c['q_ok'] - c['q_ng']:7.3f} {c['ens_std']:10.5f}")
print("-" * 60)
print("앙상블std = 두 Q 의 불일치. 0 에 가까우면 relabel 에서 후보를 줄 세울 변별력이 없다.")

# ── 비디오 ──────────────────────────────────────────────────────────────────
import imageio.v2 as imageio  # noqa: E402
from PIL import Image  # noqa: E402

T = max(len(fr) for _, fr, _ in pick)
ROW_H, PW = a.row_h, a.plot_w
cmap = plt.get_cmap("turbo")
colors = {s: cmap(0.08 + 0.84 * i / max(len(curves) - 1, 1)) for i, s in enumerate(curves)}

# 카메라: 크롭하지 말고 종횡비를 지켜 행 높이에 맞춘다.
frames_by_ep, CAM_W = {}, None
for e, fr, _ in pick:
    vp = vpath(int(e))
    if not vp.is_file():
        raise SystemExit(f"비디오가 없다: {vp}")
    fs = imageio.mimread(str(vp), memtest=False)
    h, w = fs[0].shape[:2]
    CAM_W = CAM_W or int(round(ROW_H * w / h))
    frames_by_ep[int(e)] = [
        np.asarray(Image.fromarray(f[..., :3]).resize((CAM_W, ROW_H), Image.BILINEAR))
        for f in fs]
print(f"[카메라] 원본 {w}x{h} -> 행 {CAM_W}x{ROW_H} (종횡비 유지)")

def render(sub: dict, out: Path, title: str) -> None:
    """sub 에 담긴 체크포인트들의 Q 를 겹쳐 그린 비디오 하나를 만든다."""
    # 각 행의 곡선 패널을 미리 렌더한다 (프레임마다 다시 그리면 느리다 — 커서만 덧그린다).
    bases, spans = [], []
    for r, (e, fr, o) in enumerate(pick):
        last = r == len(pick) - 1
        truth = (G ** (len(fr) - np.arange(len(fr)))) if o else np.zeros(len(fr))
        fig, ax = plt.subplots(figsize=(PW / 100, ROW_H / 100), dpi=100)
        ax.plot(truth, color="0.5", lw=1.3, ls="--",
                label="참값 γ^(T-t)" if r == 0 else None)
        for st_, c in sub.items():
            ax.plot(c["q"][int(e)], lw=1.4, color=colors[st_], label=st_ if r == 0 else None)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, .5, 1])
        ax.tick_params(labelsize=7)
        if not last:
            ax.set_xticklabels([])            # 아래 행 그림과 라벨이 겹치지 않게
        ax.grid(alpha=.25)
        ax.text(0.008, 0.94, f"ep{int(e)}  {len(fr)}f  {'성공' if o else '실패'}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(fc="white", ec="none", alpha=.8))
        if r == 0:
            ax.legend(fontsize=6.5, ncol=len(sub) + 1, loc="upper right", framealpha=.85)
            ax.text(0.5, 0.94, title, transform=ax.transAxes, fontsize=8,
                    ha="center", va="top",
                    bbox=dict(fc="white", ec="none", alpha=.8))
        fig.subplots_adjust(left=0.05, right=0.995,
                            top=0.97, bottom=0.26 if last else 0.06)
        fig.canvas.draw()
        bases.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        bb = ax.get_position()
        spans.append((int(bb.x0 * PW), int(bb.x1 * PW)))
        plt.close(fig)

    w_ = imageio.get_writer(str(out), fps=a.fps, codec="libx264", pixelformat="yuv420p",
                            macro_block_size=1, output_params=["-crf", "20"])
    for t in range(T):
        strip = []
        for (e, fr, _), bimg, (x0, x1) in zip(pick, bases, spans):
            k = min(t, len(fr) - 1)
            panel = bimg.copy()
            cx = int(x0 + (x1 - x0) * k / max(T - 1, 1))
            panel[:, max(cx - 1, 0):cx + 2] = np.array([220, 40, 40], np.uint8)
            cam = frames_by_ep[int(e)][min(k, len(frames_by_ep[int(e)]) - 1)]
            row = np.hstack([cam, panel])
            row[-1, :] = 210                                # 행 구분선
            strip.append(row)
        w_.append_data(np.vstack(strip).astype(np.uint8))
    w_.close()
    print(f"[비디오] {out.name}  ({T}프레임, {len(pick)}행, "
          f"{CAM_W + PW}x{ROW_H * len(pick)})")


print()
# 겹친 판 — 스텝별 변화를 한 화면에서 비교한다
render(curves, out_dir / f"{a.tag}_{'-'.join(curves)}.mp4",
       f"{a.exp} / {a.tag}   체크포인트 겹침")
# 스텝별 판 — 곡선이 하나뿐이라 진동·계단이 훨씬 잘 보인다
if not a.combined_only:
    for st_, c in curves.items():
        render({st_: c}, out_dir / f"{a.tag}_step{st_}.mp4",
               f"{a.exp} / {a.tag}   step {st_}   "
               f"AUC {c['auc']:.3f}  간격 {c['q_ok'] - c['q_ng']:.3f}  "
               f"앙상블std {c['ens_std']:.5f}")

print("\n  실선 = critic Q (색 = 체크포인트 스텝),  회색 점선 = 참값 γ^(T-t),"
      "  빨간 세로선 = 현재 프레임")
print("  x 축은 모든 행이 동일하다 — 짧은 에피소드는 먼저 끝나고 그 자리에서 멈춘다")
