#!/usr/bin/env python3
"""**개입 여지가 에피소드 구간마다 다른가** — 상태별 Q 비중의 근거를 찾는다.

동기 (fuji 관찰): 초반에 피더를 잡는 구간은 BC 가 이미 잘하므로 guidance 가 필요 없고,
꺼내서 다른 선반에 넣는 구간에서는 guidance 를 해야 성공률이 오른다. 그렇다면 Q 의
비중(온도)이 상태마다 달라야 하는데 **무엇으로 그 구간을 감지하는가** 가 문제다.

앙상블 std 는 죽은 신호였다 (실측: 완전 무작위 액션에서도 1.4~2.2배). 그것은 "critic 이
이 상태를 아는가"(불확실성)를 재는데, 필요한 것은 "개입하면 달라지는가"(이득)다.

재는 것 — 전부 로그된 액션 기준이라 VLA 호출이 없다:
  V(s)       상태 가치. 궤도에 올랐는가
  Q - V      로그 액션의 advantage
  |dQ|       로그 액션을 흔들었을 때 Q 변화 = 액션 민감도
  gradQ      critic 이 액션을 바꾸고 싶은 정도

에피소드 진행률 구간별로, 성공/실패를 갈라서 본다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from rl.critic_io import load_critic, load_stepwise_critic
from rl.data import build_flat, find_sessions, open_images, resolve_modality
from rl.expo import ExpoConfig
from rl.nets import explore_spec
from rl.vla_rldx import load_state_action_processor, normalize_states

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--critic", required=True, help="work 기준 상대경로도 된다")
p.add_argument("--model-path", default="")
p.add_argument("--features", default="", help="cogfeat.npy 등. 비면 픽셀 인코더")
p.add_argument("--groups", default="", help="기본: exp yaml 의 explore_groups")
p.add_argument("--all-dims", action="store_true",
               help="explore_groups 가 아니라 **액션 전 차원**을 흔든다. explore 범위가"
                    " 좁아서 둔한 것인지, critic 자체가 둔한 것인지 가른다")
p.add_argument("--holdout", default="0.1")
p.add_argument("--frames", type=int, default=2048)
p.add_argument("--sigmas", default="0.018,0.05,0.1,0.2,0.5")
p.add_argument("--device", default="cuda")
a = p.parse_args()
dev = a.device

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
R, LAT = exp["replan_steps"], exp["inference_latency"]
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / (a.model_path or exp["base_policy"])
groups = [g.strip() for g in a.groups.split(",") if g.strip()] or list(exp["explore_groups"])

mod, _ = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
imgs, _ = open_images(work / "images.mm")
norm = np.load(work / "actnorm.npy", mmap_mode="r")
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
FULL, A_DIM = (LAT + R) * mod.action_dim, mod.action_dim

ck = Path(a.critic)
if not ck.is_file():
    ck = work / a.critic
if not ck.is_file():
    raise SystemExit(f"체크포인트가 없다: {a.critic}  (work={work})")
# offline_iql (CriticEnsemble) 과 offline_iql_qvgm (StepwiseEnsemble) 둘 다 받는다.
# 후자는 로더도 다르고 q() 시그니처도 다르므로 (state 를 latent 안에 이미 넣는다)
# 얇은 어댑터로 감싸 아래 코드가 한 가지 형태만 보게 한다.
QVGM = torch.load(ck, map_location="cpu").get("kind") == "qvgm"
_SC = None
if QVGM:
    _SC = load_stepwise_critic(ck, work, snorm, dev=dev)

    class _QvgmAdapter:
        def latent_of(self, i, st):
            return _SC.latent_of(i, st)

        def q(self, lat, st, act):
            # (num_qs, B, n_steps) -> 청크 위치별 Q 를 더해 (num_qs, B).
            # 앙상블 축을 남겨야 호출측이 min/std 를 직접 잡는다.
            return _SC.q_all(lat, act).sum(-1)

    C = _QvgmAdapter()
else:
    C = load_critic(ck, work, cfg, mod.n_cams, FULL, snorm.shape[1],
                    features=a.features, imgs=imgs, dev=dev)

if a.all_dims:
    # prefix(LAT) 는 실행이 확정된 구간이라 제외하고, 실행 구간의 **모든** 관절을 흔든다.
    idx = np.array([t * A_DIM + j for t in range(LAT, LAT + R) for j in range(A_DIM)])
    label = f"실행 구간 전 관절 ({A_DIM}개)"
else:
    idx = np.asarray(explore_spec(mod.offsets("action"), groups, A_DIM, R, LAT).index)
    label = f"explore_groups {groups}"
NIDX = len(idx)
IDX = torch.as_tensor(idx, device=dev)
print(f"[편집 범위] {label} → 액션 {FULL}차원 중 {NIDX}개 (prefix {LAT}스텝 제외)")

frac = float(a.holdout) if a.holdout.replace(".", "", 1).isdigit() else 0.0
hold = (np.isin(flat.episode, np.unique(flat.episode)[::max(2, int(round(1 / frac)))])
        if 0 < frac < 1 else
        np.isin(flat.session, [i for i, n in enumerate(flat.sessions) if a.holdout in n]))
eps = [(e, np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode[hold])]
eps = [(e, fr, bool(flat.is_success[fr[-1]])) for e, fr in eps]
print(f"[평가셋] 홀드아웃 에피소드 {len(eps)} (성공 {sum(o for _, _, o in eps)})")

# 성공/실패 Q 격차 — 모든 ΔQ 를 이 눈금으로 읽는다.
def q_at(k, act=None):
    with torch.no_grad():
        st = torch.from_numpy(snorm[k]).to(dev)
        lat = C.latent_of(k, st)
        aa = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev) if act is None else act
        q = C.q(lat, st, aa)
        return q.min(0).values.float().cpu().numpy(), q.std(0).float().cpu().numpy()

fin = np.array([q_at(fr[-1:])[0][0] for _, fr, _ in eps])
okm = np.array([o for _, _, o in eps])
GAP = float(fin[okm].mean() - fin[~okm].mean())
print(f"[눈금] 홀드아웃 마지막 프레임 Q: 성공 {fin[okm].mean():+.4f} / "
      f"실패 {fin[~okm].mean():+.4f} → **격차 {GAP:.4f}**  (AUC "
      f"{float((fin[okm][:, None] > fin[~okm][None, :]).mean()):.3f})")

# ── 진행률 구간별 집계 ────────────────────────────────────────────────────────
SIG, BINS = 0.05, 5
rows = {}
for e, fr, ok in eps:
    T = len(fr)
    for b in range(BINS):
        k = fr[int(T*b/BINS): max(int(T*(b+1)/BINS), int(T*b/BINS)+1)]
        if len(k) == 0:
            continue
        k = k[:: max(1, len(k)//24)][:24]
        st = torch.from_numpy(snorm[k]).to(dev)
        lat = C.latent_of(k, st)
        a0 = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[k])[:, :LAT+R].reshape(len(k), -1))).to(dev)
        with torch.no_grad():
            q0 = C.q(lat, st, a0).min(0).values
            v = _SC.v(lat) if QVGM else torch.zeros_like(q0)
            g = torch.zeros_like(a0)
            g[:, IDX] = torch.randn(len(k), NIDX, device=dev) * SIG
            dq = (C.q(lat, st, (a0+g).clamp(-1, 1)).min(0).values - q0).abs()
        ag = a0.clone().requires_grad_(True)
        gq, = torch.autograd.grad(C.q(lat, st, ag).mean(0).sum(), ag)
        gn = gq[:, IDX].norm(dim=-1) / NIDX**0.5
        r = rows.setdefault((ok, b), {"q": [], "v": [], "dq": [], "g": []})
        for key, val in (("q", q0), ("v", v), ("dq", dq), ("g", gn)):
            r[key].append(val.detach().float().cpu().numpy())

print(f"\n=== 에피소드 진행률 구간별 (노이즈 {SIG}, 성공/실패 격차 {GAP:.4f} 로 환산) ===")
print(f"{'':>6} {'구간':>10} {'V(s)':>8} {'Q(로그)':>9} {'Q-V':>8} {'|dQ|/격차':>10} {'gradQ':>11}")
for ok in (True, False):
    for b in range(BINS):
        r = rows.get((ok, b))
        if not r:
            continue
        q, v = np.concatenate(r["q"]), np.concatenate(r["v"])
        dq, g = np.concatenate(r["dq"]), np.concatenate(r["g"])
        print(f"{'성공' if ok else '실패':>6} {f'{b/BINS:.1f}~{(b+1)/BINS:.1f}':>10} "
              f"{v.mean():8.4f} {q.mean():9.4f} {(q-v).mean():8.4f} "
              f"{100*np.median(dq)/GAP:9.2f}% {np.median(g):11.3e}")
print("""
읽는 법:
  |dQ|/격차 가 구간마다 다르면 -> 액션 민감도에 구간 구조가 있다 (상태별 온도의 근거)
  gradQ 가 후반에 커지면       -> critic 이 후반에 액션을 더 바꾸고 싶어한다
  Q-V 가 후반에 작아지면       -> 로그 액션이 후반에 평균 이하다 (개선 여지)
  전부 평평하면                -> 이 신호들로는 구간을 구분할 수 없다""")
