#!/usr/bin/env python3
"""critic 이 액션을 실제로 조정하는지 — PA-RL 식 액션 최적화로 진단.

로그된 액션에서 출발해 ∇_a Q 로 상승시킨 뒤, **얼마나 움직였는지** 를 프레임별로 본다.
편집 범위는 explore_groups(fuji: right_arm_joints)로 제한한다 — EXPO 롤아웃이 그 구간만
건드리므로 진단도 같은 구간에서 해야 의미가 있다.

기대:
  성공 에피소드 → 이동거리 작다 (로그된 액션이 이미 좋았다)
  실패 에피소드 → 실패 직전에 이동거리가 커진다 (critic 이 다른 액션을 원한다)

PA-RL 대응 (jaxrl_m/agents/continuous/action_optimization.py):
  · 상승은 `a ← a + step_size · ∇_a Q̄`,  Q̄ = 앙상블 **mean** (optimize_critic_ensemble_min=False)
  · 액션공간으로 clip (우리는 processor 의 clip_outliers 와 같은 ±1)
  · num_steps=10, step_size=3e-4 가 기본값이지만 **Q·액션 스케일이 달라 그대로면 거의 안 움직인다**
    → g 노름과 이동거리를 같이 찍으니 step_size 를 보고 조절할 것
  · PA-RL 은 base policy 후보 M=32 → top-K=10 에서 출발한다. 여기서는 **로그된 액션에서** 출발해
    "critic 이 이 액션을 어떻게 바꾸고 싶은가" 만 분리해 본다 (VLA 호출이 없다)

usage:
  PYTHONPATH="$L_PYTHONPATH" $L_PY -u -m rl.probe_actopt \\
      --exp fuji --data <데이터셋> --checkpoints <ckpt> \\
      --critic critic_iql-dist128-t07-g0999-q10all-s0.pt \\
      --model-path rldx-img-curated/rldx_img_curated-0810-0818-r05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

from rl.data import build_flat, find_sessions, open_images, resolve_modality
from rl.expo import ExpoConfig
from rl.nets import BatchEncoder, CriticEnsemble, explore_spec, xavier_
from rl.vla_rldx import load_state_action_processor, normalize_states

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="fuji")
p.add_argument("--data", type=Path, required=True)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--critic", required=True,
               help="체크포인트 경로. work 디렉토리 기준 상대경로도 된다 "
                    "(예: iql-dist128-t07-g0999-q10all-s0/critic_latest.pt)")
p.add_argument("--model-path", default="", help="processor 를 읽을 체크포인트 (기본: exp yaml)")
p.add_argument("--groups", default="", help="편집할 action 그룹 (기본: exp yaml 의 explore_groups)")
p.add_argument("--num-steps", type=int, default=10)
p.add_argument("--step-size", type=float, default=3e-4, help="PA-RL 기본값 3e-4")
p.add_argument("--auto-step", type=float, default=0.0,
               help="차원당 목표 이동거리 D. 주면 표본에서 ‖g‖ 를 재서 "
                    "step_size = D/(num_steps·median‖g‖) 로 잡는다. PA-RL 의 3e-4 는 우리 "
                    "Q·액션 스케일에서 이동이 1e-9 라 사실상 아무 일도 안 한다")
p.add_argument("--holdout", default="0.2")
p.add_argument("--stride", type=int, default=4, help="에피소드에서 몇 프레임마다 볼지")
p.add_argument("--anno", type=Path, help="probe_pairs 의 anno.csv — 실패 시점을 그림에 표시")
p.add_argument("--video-eps", type=int, default=6,
               help="비디오로 만들 에피소드 수 (성공/실패 절반씩). 0 이면 안 만든다")
p.add_argument("--video-stride", type=int, default=2, help="비디오 프레임 간격")
p.add_argument("--device", default="cuda")
a = p.parse_args()
dev = a.device

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / (a.model_path or exp["base_policy"])
groups = [g.strip() for g in a.groups.split(",") if g.strip()] or list(exp["explore_groups"])

# --- 데이터 (학습과 같은 경로) ----------------------------------------------
mod, _ = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
imgs, meta = open_images(work / "images.mm")
norm = np.load(work / "actnorm.npy", mmap_mode="r")
snorm = normalize_states(load_state_action_processor(base, RLDX, exp["rldx_data_config"]),
                         mod.embodiment_tag, mod, flat.state)
FULL, A_DIM = (LAT + R) * mod.action_dim, mod.action_dim

# --- critic ------------------------------------------------------------------
ck = Path(a.critic)
if not ck.is_file():
    ck = work / a.critic
if not ck.is_file():
    raise SystemExit(f"체크포인트가 없다: {a.critic}  (work={work})")
sd = torch.load(ck, map_location=dev)
BINS = sd.get("bins") or 0
enc = BatchEncoder(3 * mod.n_cams, cfg.latent_dim_image, cfg.encoder_stage_sizes,
                   cfg.encoder_num_filters).to(dev).eval()
critic = CriticEnsemble(cfg.latent_dim_image, snorm.shape[1], FULL, sd.get("num_qs", cfg.num_qs),
                        cfg.latent_dim_state, cfg.include_state, cfg.hidden_dims,
                        cfg.critic_layer_norm).to(dev).eval()
if BINS:
    for m in critic.qs:
        m.head = xavier_(nn.Linear(m.body.out_dim, BINS)).to(dev)
    lo_q, hi_q = (float(x) for x in (sd.get("q_range") or "0,1").split(","))
    edges = torch.linspace(lo_q, hi_q, BINS + 1, device=dev)
    centers = (edges[:-1] + edges[1:]) / 2
    q_of = lambda x: (x.softmax(-1) * centers).sum(-1)
else:
    q_of = lambda x: x
value = CriticEnsemble(cfg.latent_dim_image, snorm.shape[1], 0, 1, cfg.latent_dim_state,
                       cfg.include_state, cfg.hidden_dims, cfg.critic_layer_norm).to(dev).eval()
enc.load_state_dict(sd["enc"]); critic.load_state_dict(sd["critic"])
if "value" in sd:
    value.load_state_dict(sd["value"])
else:
    value = None
print(f"[critic] {ck}\n          step {sd.get('step')}  γ={sd.get('discount')}  "
      f"τ={sd.get('expectile')}  bins={BINS}  num_qs={sd.get('num_qs')}")

# --- 편집 마스크: explore_groups 의 실행 구간만 -------------------------------
spec = explore_spec(mod.offsets("action"), groups, A_DIM, R, LAT)
MASK = torch.zeros(FULL, device=dev)
MASK[spec.index] = 1.0
NIDX = len(spec.index)
print(f"[편집 범위] {groups} → 액션 {FULL}차원 중 {NIDX}개 "
      f"({spec.active_dim}관절 x {R}스텝, prefix {LAT}스텝 제외)")
print(f"[상승] num_steps {a.num_steps}, step_size {a.step_size}, 앙상블 mean 으로 상승 (PA-RL)")

# --- 평가 에피소드 (학습과 같은 holdout) --------------------------------------
frac = float(a.holdout) if a.holdout.replace(".", "", 1).isdigit() else 0.0
if 0 < frac < 1:
    hold = np.isin(flat.episode, np.unique(flat.episode)[::max(2, int(round(1 / frac)))])
else:
    hold = np.isin(flat.session, [i for i, n in enumerate(flat.sessions) if a.holdout in n])
eps = [(e, np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode[hold])]
eps = [(e, fr, bool(flat.is_success[fr[-1]])) for e, fr in eps]
print(f"[평가셋] 에피소드 {len(eps)} (성공 {sum(o for _,_,o in eps)})")

def obs_of(idx):
    x = np.asarray(imgs[idx])
    return torch.from_numpy(np.ascontiguousarray(
        np.concatenate([x[:, c] for c in range(x.shape[1])], -1))).to(dev)

def ascend(idx, bs=48):
    """로그된 액션에서 ∇_a Q 상승.
    반환 열: 0 q_log(min) 1 q_opt(min) 2 dq_mean 3 d_rms 4 g_rms 5 V(s) 6 d_l2"""
    out = []
    for c in range(0, len(idx), bs):
        k = idx[c:c + bs]
        with torch.no_grad():
            lat = enc(obs_of(k), stop_gradient=True)
            st = torch.from_numpy(snorm[k]).to(dev)
            a0 = torch.from_numpy(np.ascontiguousarray(
                np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev)
            q0 = q_of(critic(lat, st, a0))
        act = a0.clone()
        g_last = None
        for _ in range(a.num_steps):
            act = act.detach().requires_grad_(True)
            qm = q_of(critic(lat, st, act)).mean(0).sum()      # PA-RL: 앙상블 mean
            g, = torch.autograd.grad(qm, act)
            g_last = g
            act = (act + a.step_size * g * MASK).clamp(-1.0, 1.0)
        with torch.no_grad():
            q1 = q_of(critic(lat, st, act.detach()))
            d = (act.detach() - a0)[:, spec.index]
            gg = (g_last * MASK)[:, spec.index]
            v = (value(lat, st, torch.zeros(len(k), 0, device=dev))[0].float().cpu().numpy()
                 if value is not None else np.zeros(len(k), np.float32))
            out.append(np.stack([
                q0.min(0).values.float().cpu().numpy(),
                q1.min(0).values.float().cpu().numpy(),
                (q1.mean(0) - q0.mean(0)).float().cpu().numpy(),
                (d.norm(dim=-1) / NIDX ** 0.5).float().cpu().numpy(),
                (gg.norm(dim=-1) / NIDX ** 0.5).float().cpu().numpy(),
                v,
                d.norm(dim=-1).float().cpu().numpy()], 1))
    return np.concatenate(out)

# --- step_size 캘리브레이션 -------------------------------------------------
if a.auto_step:
    probe_idx = np.concatenate([fr[::max(1, len(fr) // 24)] for _, fr, _ in eps])[:256]
    gs = []
    for c in range(0, len(probe_idx), 48):
        k = probe_idx[c:c + 48]
        with torch.no_grad():
            lat = enc(obs_of(k), stop_gradient=True)
            st = torch.from_numpy(snorm[k]).to(dev)
        act0 = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev).requires_grad_(True)
        qm = q_of(critic(lat, st, act0)).mean(0).sum()
        g, = torch.autograd.grad(qm, act0)
        gs.append(((g * MASK)[:, spec.index].norm(dim=-1) / NIDX ** 0.5).cpu().numpy())
    gmed = float(np.median(np.concatenate(gs)))
    a.step_size = a.auto_step / (a.num_steps * max(gmed, 1e-12))
    print(f"[캘리브레이션] ‖g‖/차원 중앙값 {gmed:.6f} → 목표 이동 {a.auto_step} 이면 "
          f"step_size {a.step_size:.4g} (PA-RL 3e-4 의 {a.step_size/3e-4:.0f}배)")

anno = {}
if a.anno and a.anno.is_file():
    import csv
    with a.anno.open() as fh:
        for r in csv.DictReader(fh):
            if (r.get("fail_sec") or "").strip():
                si = [i for i, n in enumerate(flat.sessions) if n == r["session"]]
                if si:
                    gep = int(r["episode"]) + flat.ep_offset[si[0]]
                    anno[gep] = int(round(float(r["fail_sec"]) * float(r["fps"])))

res, curves = [], []
for e, fr, ok in eps:
    ii = fr[::a.stride]
    m = ascend(ii)
    curves.append((e, ii - fr[0], m, ok, len(fr)))
    res.append((ok, m))
    print(f"  ep{e:04d} {'성공' if ok else '실패'} {len(fr):5d}프레임  "
          f"Q {m[:,0].mean():+.4f}→{m[:,1].mean():+.4f}  이동 {m[:,3].mean():.4f}  "
          f"‖g‖ {m[:,4].mean():.4f}")

S = np.concatenate([m for ok, m in res if ok]); F = np.concatenate([m for ok, m in res if not ok])
print(f"\n{'':14s} {'Q(a_log)':>9} {'Q(a_opt)':>9} {'ΔQ(mean)':>9} {'이동/차원':>9} {'‖g‖/차원':>9}")
for tag, M in (("성공 에피소드", S), ("실패 에피소드", F)):
    print(f"{tag:14s} {M[:,0].mean():+9.4f} {M[:,1].mean():+9.4f} {M[:,2].mean():+9.5f} "
          f"{M[:,3].mean():9.5f} {M[:,4].mean():9.4f}")
print(f"\n{'종료전 프레임':>12} {'성공 이동':>10} {'실패 이동':>10} {'성공 ΔQ':>10} {'실패 ΔQ':>10}")
for lo, hi in ((0,100),(100,200),(200,400),(400,800),(800,2000)):
    sv, fv = [], []
    for e, x, m, ok, L in curves:
        back = (L - 1) - x
        sel = (back >= lo) & (back < hi)
        if sel.any():
            (sv if ok else fv).append(m[sel])
    if not sv or not fv: continue
    sv, fv = np.concatenate(sv), np.concatenate(fv)
    print(f"{f'{lo}-{hi}':>12} {sv[:,3].mean():10.5f} {fv[:,3].mean():10.5f} "
          f"{sv[:,2].mean():+10.5f} {fv[:,2].mean():+10.5f}")

ev = ck.parent / "plots"                             # 새 레이아웃: <tag>/plots/
ev.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for e, x, m, ok, L in curves:
    c = "tab:green" if ok else "tab:red"
    ax[0].plot(x, m[:, 0], color=c, lw=1, alpha=0.5)
    ax[1].plot(x, m[:, 3], color=c, lw=1, alpha=0.5)
    if e in anno:
        ax[1].axvline(anno[e], color=c, ls=":", lw=0.8, alpha=0.7)
ax[0].set_ylabel("Q (min of ensemble)"); ax[0].grid(alpha=0.25)
ax[0].set_title(f"{sd.get('tag') or ck.parent.name}  step {sd.get('step')}   green=success red=failure"
                f"   (dotted = labeled failure moment)")
ax[1].set_ylabel(f"|a_opt - a_log| per dim  ({'+'.join(groups)}, {NIDX} dims)")
ax[1].set_xlabel("frame in episode"); ax[1].grid(alpha=0.25)
fig.tight_layout()
out = ev / f"actopt_s{a.step_size:g}_n{a.num_steps}.png"
fig.savefig(out, dpi=110); plt.close(fig)
print(f"\n[그림] {out}")


# --- 에피소드별 비디오: 카메라 + (V, Q_log, Q_opt) + 액션 이동거리 ------------
FPS = json.loads((sessions[0] / "meta/info.json").read_text())["fps"]

def make_video(path, fr, m, title, ph=150, hd=22):
    """matplotlib 로 축을 한 번 렌더하고, 프레임마다 커서만 덧그린다 (offline_critic_0 와 같은 방식).
    코덱은 libx264 — cv2 번들 ffmpeg 에는 H.264 인코더가 없다."""
    x0 = np.asarray(imgs[fr[0]])
    Hc, W = x0.shape[1], x0.shape[2] * x0.shape[0]
    xs = np.arange(len(fr))
    panels = []
    for ylabel, series, ylim in (
            ("value", [("V(s)", m[:, 5], "0.5"), ("Q(a_log)", m[:, 0], "tab:blue"),
                       ("Q(a_opt)", m[:, 1], "tab:orange")], (-0.05, 1.05)),
            (f"|a_opt-a_log| L2 ({NIDX}d)", [("dist", m[:, 6], "tab:purple")], None)):
        fig = plt.figure(figsize=(W / 100, ph / 100), dpi=100)
        ax = fig.add_axes([0.075, 0.20, 0.915, 0.76])
        for lab, y, c in series:
            ax.plot(xs, y, color=c, lw=1.2, label=lab)
        ax.set_xlim(0, max(1, len(fr) - 1))
        if ylim:
            ax.set_ylim(*ylim); ax.axhline(0, color="0.8", lw=0.6); ax.axhline(1, color="0.8", lw=0.6)
        ax.set_ylabel(ylabel, fontsize=7); ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
        ax.legend(fontsize=6, loc="upper left", ncol=len(series), framealpha=0.6)
        fig.canvas.draw()
        base = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        # 커서용 x 픽셀 좌표
        px = ax.transData.transform(np.c_[xs, np.zeros_like(xs)])[:, 0].astype(np.int32)
        plt.close(fig)
        panels.append((base, px))

    vw = imageio.get_writer(str(path), fps=max(1, int(FPS / a.video_stride)), codec="libx264",
                            quality=8, macro_block_size=1, pixelformat="yuv420p")
    for t in range(len(fr)):
        cams = np.concatenate(list(np.asarray(imgs[fr[t]])), axis=1)
        head = np.full((hd, W, 3), 255, np.uint8)
        cv2.putText(head, f"{title}  t={t*a.video_stride}  V={m[t,5]:+.3f}  "
                          f"Q_log={m[t,0]:+.3f}  Q_opt={m[t,1]:+.3f}  d={m[t,6]:.3f}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        rows = [head, cams]
        for base, px in panels:
            pan = base.copy()
            cv2.line(pan, (int(px[t]), 0), (int(px[t]), ph - 1), (220, 0, 0), 1)
            rows.append(pan)
        vw.append_data(np.concatenate(rows, axis=0))
    vw.close()

if a.video_eps:
    sel = ([x for x in eps if x[2]][:max(1, a.video_eps // 2)]
           + [x for x in eps if not x[2]][:max(1, a.video_eps // 2)])
    print(f"\n[비디오] {len(sel)} 에피소드, stride {a.video_stride}")
    for e, fr, ok in sel:
        ii = fr[::a.video_stride]
        m = ascend(ii)
        tag = f"ep{e:04d}_{'succ' if ok else 'fail'}"
        out = ev / f"actopt_{tag}.mp4"
        make_video(out, ii, m, tag)
        print(f"  {out.name}  {len(ii)} 프레임  {out.stat().st_size/1e6:.1f} MB")
