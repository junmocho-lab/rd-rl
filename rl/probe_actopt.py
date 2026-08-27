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
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

from rl.data import build_flat, find_sessions, open_images, resolve_modality
from rl.expo import ExpoConfig
from rl.critic_io import load_critic
from rl.nets import explore_spec
from rl.vla_rldx import (denormalize_actions, load_state_action_processor,
                         normalize_states)

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
p.add_argument("--features", default="",
               help="frozen VLM feature 로 학습한 critic 이면 그 npy 이름 (예: cogfeat.npy). "
                    "ckpt 에 기록돼 있으면 생략해도 된다")
p.add_argument("--ascend", default="mean", choices=("mean", "min"),
               help="상승·표시에 쓸 앙상블 축약. PA-RL 은 config 기본값이 "
                    "optimize_critic_ensemble_min=False 라 실전 경로가 mean 이고 후보 선택은 "
                    "하드코딩 mean 이다. min 은 보수적(REDQ 식) 대안 — 상승과 플롯이 같은 값을 "
                    "쓰도록 이 선택이 둘 다에 적용된다")
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
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
FULL, A_DIM = (LAT + R) * mod.action_dim, mod.action_dim

# --- critic ------------------------------------------------------------------
ck = Path(a.critic)
if not ck.is_file():
    ck = work / a.critic
if not ck.is_file():
    raise SystemExit(f"체크포인트가 없다: {a.critic}  (work={work})")
C = load_critic(ck, work, cfg, mod.n_cams, FULL, snorm.shape[1],
                features=a.features, imgs=imgs, dev=dev)

# --- 편집 마스크: explore_groups 의 실행 구간만 -------------------------------
spec = explore_spec(mod.offsets("action"), groups, A_DIM, R, LAT)
MASK = torch.zeros(FULL, device=dev)
MASK[spec.index] = 1.0
NIDX = len(spec.index)
JCOL = torch.as_tensor(sorted({int(i) % A_DIM for i in spec.index}), device=dev)   # explore 관절
# 상승 목적함수와 화면에 찍는 값을 **같은** 축약으로 묶는다. 예전에는 mean 으로 올리고 min 을
# 그려서, 보이는 곡선이 실제로 올리고 있는 양이 아니었다.
RED = (lambda q: q.min(0).values) if a.ascend == "min" else (lambda q: q.mean(0))
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

# --- 기준 스케일 --------------------------------------------------------------
# 정규화 공간의 L2 자체로는 "큰 변화" 인지 알 수 없다 (q01/q99 로 관절 스케일이 뭉개져 있다).
# 데이터가 스스로 주는 자연 단위를 쓴다: 같은 (관절, 청크스텝) 에서 t → t+1 의 변화는
# 정확히 그 관절이 **1프레임 동안 실제로 움직인 양**이다. 편집량을 이것으로 나누면
# "이 편집은 평소 N 프레임치 움직임이다" 가 되어 바로 해석된다.
_pi = np.concatenate([fr[:-1:max(1, len(fr) // 40)] for _, fr, _ in eps])[:2048]
_d1 = (np.asarray(norm[_pi + 1])[:, :LAT + R].reshape(len(_pi), -1)
       - np.asarray(norm[_pi])[:, :LAT + R].reshape(len(_pi), -1))[:, spec.index]
REF1 = float(np.median(np.linalg.norm(_d1, axis=-1)) / NIDX ** 0.5)
print(f"[기준] 1프레임 자연 변화 (중앙) = {REF1:.5f}/차원 — 편집량을 이걸로 나눠 프레임 환산")


# --- OOD 눈금: 앙상블 불일치의 양 끝점을 실측한다 ---------------------------
# ens.std 자체는 단위가 없어 "얼마면 심한가" 를 말할 수 없다. 두 극단을 재서 0~1 로 환산한다.
#   바닥 STD_DATA : 로그된 액션 (확실히 분포 안)
#   중간 STD_SHUF : 다른 프레임의 로그된 액션을 explore 차원에만 붙여넣기 — 액션 자체는
#                   실제 로봇 액션인데 맥락이 틀린 경우. "그럴듯하지만 여기 것이 아님" 의 눈금
#   천장 STD_RAND : explore 차원을 U(-1,1) 난수로 (완전 OOD)
def _ens_std(k, mk=None):
    with torch.no_grad():
        st_ = torch.from_numpy(snorm[k]).to(dev)
        lat_ = C.latent_of(k, st_)
        a_ = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev)
        if mk is not None:
            a_ = a_.clone()
            a_[:, spec.index] = mk
        return C.q(lat_, st_, a_).std(0).float().cpu().numpy()


_ck = np.concatenate([fr[::max(1, len(fr) // 16)] for _, fr, _ in eps])[:768]
_rng = np.random.default_rng(0)
_sh = _rng.permutation(_ck)                                   # 다른 프레임의 액션
_shv = torch.from_numpy(np.ascontiguousarray(
    np.asarray(norm[_sh])[:, :LAT + R].reshape(len(_sh), -1))).to(dev)[:, spec.index]
_g = torch.Generator(device=dev).manual_seed(0)
_rd = (torch.rand(len(_ck), NIDX, device=dev, generator=_g) * 2 - 1)
STD_DATA = float(np.median(_ens_std(_ck)))
STD_SHUF = float(np.median(_ens_std(_ck, _shv)))
STD_RAND = float(np.median(_ens_std(_ck, _rd)))
SPAN = max(STD_RAND - STD_DATA, 1e-9)
OOD_SHUF = (STD_SHUF - STD_DATA) / SPAN
print(f"[OOD 눈금] 표본 {len(_ck)} 프레임의 앙상블 std 중앙값")
print(f"  로그 액션 (분포 안)        {STD_DATA:.5f}   -> OOD 0.00")
print(f"  다른 프레임 액션 (맥락 틀림) {STD_SHUF:.5f}   -> OOD {(STD_SHUF-STD_DATA)/SPAN:.2f}")
print(f"  난수 액션 (완전 OOD)       {STD_RAND:.5f}   -> OOD 1.00")


def _jerk(x):
    """(B, LAT+R, A) 청크의 실행 구간에서 explore 관절의 2차 차분 RMS.
    critic 을 무제한 상승시키면 청크가 고주파로 튀는데(제어 불가) 그걸 잡는 지표다."""
    z = x[:, LAT:LAT + R][:, :, JCOL]
    return (z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]).pow(2).mean((1, 2)).sqrt()


def ascend(idx, bs=48):
    """로그된 액션에서 ∇_a Q 상승.

    반환 열: 0 q_log(min) 1 q_opt(min) 2 dq_mean 3 d_rms 4 g_rms 5 V(s) 6 d_l2
             7 d/REF1 (프레임 환산)  8 앙상블 std(a_log)  9 앙상블 std(a_opt)
             10 jerk(a_log)  11 jerk(a_opt)  12 OOD 점수 (0=분포 안, 1=난수 수준)"""
    out = []
    for c in range(0, len(idx), bs):
        k = idx[c:c + bs]
        with torch.no_grad():
            st = torch.from_numpy(snorm[k]).to(dev)
            lat = C.latent_of(k, st)
            a0 = torch.from_numpy(np.ascontiguousarray(
                np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev)
            q0 = C.q(lat, st, a0)
        act = a0.clone()
        g_last = None
        for _ in range(a.num_steps):
            act = act.detach().requires_grad_(True)
            qm = RED(C.q(lat, st, act)).sum()         # PA-RL 실전 경로는 mean (--ascend)
            g, = torch.autograd.grad(qm, act)
            g_last = g
            act = (act + a.step_size * g * MASK).clamp(-1.0, 1.0)
        with torch.no_grad():
            a1 = act.detach()
            q1 = C.q(lat, st, a1)
            d = (a1 - a0)[:, spec.index]
            gg = (g_last * MASK)[:, spec.index]
            v = C.v(lat, st).float().cpu().numpy()
            drms = (d.norm(dim=-1) / NIDX ** 0.5).float()
            nb = len(k)
            out.append(np.stack([
                RED(q0).float().cpu().numpy(),
                RED(q1).float().cpu().numpy(),
                (RED(q1) - RED(q0)).float().cpu().numpy(),
                drms.cpu().numpy(),
                (gg.norm(dim=-1) / NIDX ** 0.5).float().cpu().numpy(),
                v,
                d.norm(dim=-1).float().cpu().numpy(),
                (drms / REF1).cpu().numpy(),                     # 프레임 환산
                q0.std(0).float().cpu().numpy(),                 # 앙상블 불일치 (외삽 신호)
                q1.std(0).float().cpu().numpy(),
                _jerk(a0.view(nb, LAT + R, A_DIM)).float().cpu().numpy(),
                _jerk(a1.view(nb, LAT + R, A_DIM)).float().cpu().numpy(),
                # 편집이 만든 불일치 증가분을 "분포 안 -> 난수" 전체 폭으로 나눈 0~1 점수
                ((q1.std(0) - q0.std(0)).float().cpu().numpy() / SPAN)], 1))
    return np.concatenate(out)

# --- step_size 캘리브레이션 -------------------------------------------------
if a.auto_step:
    probe_idx = np.concatenate([fr[::max(1, len(fr) // 24)] for _, fr, _ in eps])[:256]
    gs = []
    for c in range(0, len(probe_idx), 48):
        k = probe_idx[c:c + 48]
        with torch.no_grad():
            st = torch.from_numpy(snorm[k]).to(dev)
            lat = C.latent_of(k, st)
        act0 = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev).requires_grad_(True)
        qm = RED(C.q(lat, st, act0)).sum()
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

# --- 편집이 "말이 되는 액션" 인지 판정 ---------------------------------------
# L2 하나로는 알 수 없다. 세 축을 같이 본다.
print(f"\n{'':16s} {'OOD점수':>8} {'프레임환산':>10} {'ΔQ':>9} {'ΔQ/std':>8} {'jerk비':>7}")
for tag, M in (("성공 에피소드", S), ("실패 에피소드", F)):
    print(f"{tag:16s} {M[:,12].mean():8.3f} {M[:,7].mean():10.2f} {M[:,2].mean():+9.5f} "
          f"{(M[:,2]/np.maximum(M[:,9],1e-9)).mean():8.2f} "
          f"{(M[:,11]/np.maximum(M[:,10],1e-9)).mean():7.2f}")
ALL = np.concatenate([S, F])
ood = ALL[:, 12]
print(f"\n[판정] OOD 점수            중앙 {np.median(ood):.3f}  p95 {np.percentile(ood,95):.3f}  "
      f"최대 {ood.max():.3f}")
print(f"        (0 = 로그 액션 수준, {OOD_SHUF:.2f} = 다른 프레임 액션 붙여넣기 수준, 1 = 난수 수준)")
print(f"        붙여넣기 수준을 넘은 프레임 {100*(ood > OOD_SHUF).mean():.1f}%")
print(f"        ΔQ / ens.std(a_opt)  중앙 {np.median(ALL[:,2]/np.maximum(ALL[:,9],1e-9)):.2f}   "
      f"(1 미만이면 개선이 앙상블 노이즈 안)")
print(f"        프레임 환산 편집량   중앙 {np.median(ALL[:,7]):.2f}  "
      f"p95 {np.percentile(ALL[:,7],95):.2f} 프레임치")
print(f"        청크 jerk 비율      중앙 {np.median(ALL[:,11]/np.maximum(ALL[:,10],1e-9)):.2f}  "
      f"p95 {np.percentile(ALL[:,11]/np.maximum(ALL[:,10],1e-9),95):.2f}   (1 이면 매끄러움 유지)")

# 관절별 편집량을 라디안으로. 정규화 공간 수치는 관절 스케일이 뭉개져 있어 해석 불가.
pj = np.concatenate([fr[::max(1, len(fr) // 12)] for _, fr, _ in eps])[:512]
with torch.no_grad():
    stj = torch.from_numpy(snorm[pj]).to(dev)
    latj = C.latent_of(pj, stj)
    a0j = torch.from_numpy(np.ascontiguousarray(
        np.asarray(norm[pj])[:, :LAT + R].reshape(len(pj), -1))).to(dev)
aj = a0j.clone()
for _ in range(a.num_steps):
    aj = aj.detach().requires_grad_(True)
    g, = torch.autograd.grad(RED(C.q(latj, stj, aj)).sum(), aj)
    aj = (aj + a.step_size * g * MASK).clamp(-1.0, 1.0)
raw0 = denormalize_actions(proc, mod.embodiment_tag, mod,
                           a0j.view(len(pj), LAT + R, A_DIM).cpu().numpy(), flat.state[pj])
raw1 = denormalize_actions(proc, mod.embodiment_tag, mod,
                           aj.detach().view(len(pj), LAT + R, A_DIM).cpu().numpy(), flat.state[pj])
dr = np.abs(raw1 - raw0)[:, LAT:LAT + R]                 # (B, R, A) 라디안
nxt = np.minimum(pj + 1, flat.ep_end[pj])                # 에피소드 경계를 넘지 않는다
nat = np.abs(flat.action[nxt] - flat.action[pj])         # 1프레임 자연 변화 (라디안)
print(f"\n{'관절':22s} {'편집 p95':>10} {'편집 최대':>10} {'1프레임 자연변화 중앙':>22} {'배수':>7}")
for name, s0, e0 in mod.offsets("action"):
    if name not in groups:
        continue
    for j in range(s0, e0):
        n_ = float(np.median(nat[:, j]))
        print(f"  {name}[{j-s0}]{'':<10} {np.percentile(dr[:,:,j],95):10.5f} "
              f"{dr[:,:,j].max():10.5f} {n_:22.5f} "
              f"{np.percentile(dr[:,:,j],95)/max(n_,1e-9):7.1f}")
print("  (단위 라디안. 배수 = 편집 p95 / 1프레임 자연변화 — 1 이하면 사실상 노이즈 수준)")

ev = ck.parent / "plots"                             # 새 레이아웃: <tag>/plots/
ev.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for e, x, m, ok, L in curves:
    c = "tab:green" if ok else "tab:red"
    ax[0].plot(x, m[:, 0], color=c, lw=1, alpha=0.5)
    ax[1].plot(x, m[:, 12], color=c, lw=1, alpha=0.5)     # OOD 점수
    if e in anno:
        ax[1].axvline(anno[e], color=c, ls=":", lw=0.8, alpha=0.7)
ax[0].set_ylabel(f"Q ({a.ascend} of ensemble)"); ax[0].grid(alpha=0.25)
ax[0].set_title(f"{C.meta.get('tag') or ck.parent.name}  step {C.meta.get('step')}   green=success red=failure"
                f"   (dotted = labeled failure moment)")
ax[1].axhline(OOD_SHUF, color="tab:red", lw=0.8, ls="--")
ax[1].axhline(1, color="0.4", lw=0.8, ls=":")
ax[1].set_ylabel(f"OOD score (0=logged, {OOD_SHUF:.2f}=shuffled, 1=random)\n"
                 f"({'+'.join(groups)}, {NIDX} dims)")
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
            ("value", [("V(s)", m[:, 5], "0.5"), (f"Q(a_log) {a.ascend}", m[:, 0], "tab:blue"),
                       (f"Q(a_opt) {a.ascend}", m[:, 1], "tab:orange")], (-0.05, 1.05)),
            # 앙상블 불일치를 0~1 OOD 점수로 환산한 것만 그린다. 기준선이 절대 의미를 준다:
            #   0 = 로그된 액션과 같은 수준 (분포 안)
            #   SHUF = 다른 프레임의 실제 액션을 붙여넣은 수준 (그럴듯하지만 여기 것이 아님)
            #   1 = 난수 액션 수준 (완전 외삽)
            ("OOD score (ens. disagreement)",
             [("a_opt", m[:, 12], "tab:brown")], (-0.05, 1.05))):
        fig = plt.figure(figsize=(W / 100, ph / 100), dpi=100)
        ax = fig.add_axes([0.075, 0.20, 0.915, 0.76])
        for lab, y, c in series:
            ax.plot(xs, y, color=c, lw=1.2, label=lab)
        ax.set_xlim(0, max(1, len(fr) - 1))
        ax.set_ylim(*ylim)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.axhline(1, color="0.8", lw=0.6)
        if "OOD" in ylabel:                                    # 붙여넣기 수준 눈금
            ax.axhline(OOD_SHUF, color="tab:red", lw=0.8, ls="--")
            ax.text(0.995, OOD_SHUF, " shuffled-action level", color="tab:red", fontsize=6,
                    ha="right", va="bottom", transform=ax.get_yaxis_transform())
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
