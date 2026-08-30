#!/usr/bin/env python3
"""critic 이 **빨리 끝난 에피소드에 높은 Q** 를 주는가 — 고정 장면 전용 분석.

성공만으로 학습하면 보상이 전부 종단 1 이고 종료에서 부트스트랩이 끊기므로
    Q(s_t, A) = γ^R V(s_{t+R}) = ... = γ^(T-t)
가 참값이다. 즉 **빨리 끝날수록 높은 Q** 여야 한다.

고정 장면이라 t=0 에서 모든 에피소드의 상태가 동일하다 → Q(s0, A_i) 의 차이는 오직
A_i 에서 온다. 따라서 `Q(s0,A_i)` 와 참값 `γ^(T_i)` 의 상관이 곧 **후보 선택이 작동할지**다:
상관이 높으면 argmax Q 가 빠른 성공을 고르고, 0 이면 무작위 선택이다.

홀드아웃(학습에 안 쓴 에피소드)에서의 값이 진짜다 — 학습 에피소드는 외운 것일 수 있다.

  python -m rl.q_vs_speed --exp dexjoco_hammer_nail_fixed --data rl-dataset/dexjoco/hammer_nail_d2r8_s0 \
      --checkpoints checkpoints --critic fixed_successonly/critic_latest.pt --features cogfeat.npy
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
from rl.vla_rldx import load_state_action_processor, normalize_states

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--critic", required=True)
p.add_argument("--features", default="cogfeat.npy")
p.add_argument("--model-path", default="")
p.add_argument("--holdout", default="0.2")
p.add_argument("--frames", default="0,20,40,60",
               help="절대 프레임. 참고용일 뿐이다 — t=0 의 청크는 '뻗기' 동작이라 "
                    "완주 속도를 결정하지 않는다 (그것은 파지·타격에서 갈린다). "
                    "IQL 구조상으로도 Q(s0,A)=γ^R V(s20) 이라 첫 청크는 자기가 도달하는 "
                    "상태를 통해서만 값에 영향을 준다")
p.add_argument("--strike-offsets", default="60,40,20,10",
               help="**타격 사건 기준** 오프셋. eef_z 하강 속도가 최대인 프레임을 타격으로 "
                    "보고 그 앞 N 프레임에서 Q 를 잰다. 타격 시점이 에피소드마다 60~358 로 "
                    "흩어지므로 절대 프레임 정렬은 국면을 섞어 신호를 상쇄시킨다 — "
                    "여기가 critic 이 실제로 후보를 구분해야 하는 지점이다")
p.add_argument("--device", default="cuda")
a = p.parse_args()
dev = a.device

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
R, LAT = exp["replan_steps"], exp["inference_latency"]
G = float(cfg.discount)
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / (a.model_path or exp["base_policy"])

mod, _ = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
flat = build_flat(find_sessions(a.data), mod)
imgs, _ = open_images(work / "images.mm")
norm = np.load(work / "actnorm.npy", mmap_mode="r")
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
FULL = (LAT + R) * mod.action_dim

ck = Path(a.critic)
if not ck.is_file():
    ck = work / a.critic
sd0 = torch.load(ck, map_location="cpu")
if sd0.get("kind") == "qvgm":
    SC = load_stepwise_critic(ck, work, snorm, dev=dev)
    q_fn = lambda k, st: SC.q_all(SC.latent_of(k, st), act_of(k)).min(0).values
else:
    C = load_critic(ck, work, cfg, mod.n_cams, FULL, snorm.shape[1],
                    features=a.features, imgs=imgs, dev=dev)
    q_fn = lambda k, st: C.q(C.latent_of(k, st), st, act_of(k)).min(0).values


def act_of(k):
    return torch.from_numpy(np.ascontiguousarray(
        np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev)


eps = [(e, np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode)]
eps = [(e, fr, bool(flat.is_success[fr[-1]])) for e, fr in eps]
frac = float(a.holdout) if a.holdout.replace(".", "", 1).isdigit() else 0.0
every = max(2, int(round(1 / frac))) if 0 < frac < 1 else 0
hold_ids = set(np.unique(flat.episode)[::every].tolist()) if every else set()
print(f"[데이터] 에피소드 {len(eps)}  성공 {sum(o for _, _, o in eps)}  "
      f"홀드아웃 {len(hold_ids)} (에피소드 {every}개마다 1개 — 학습과 같은 규칙)")

L = np.array([len(fr) for _, fr, _ in eps])
OKM = np.array([o for _, _, o in eps])
HOLD = np.array([e in hold_ids for e, _, _ in eps])
print(f"[길이] 성공 {L[OKM].min()}~{L[OKM].max()} (중앙 {int(np.median(L[OKM]))})")


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# 타격 프레임 = eef_z (state[2]) 하강 속도가 최대인 지점.
# 환경이 못 삽입량을 delta = 0.008*min(3,|vz|/0.02) 로 주므로 수직 속도가 결과를 좌우한다.
STRIKE = {}
for i, (_, fr, _) in enumerate(eps):
    z = flat.state[fr][:, 2]
    STRIKE[i] = int(np.argmin(np.diff(z))) if len(z) > 2 else 0
_st = np.array(list(STRIKE.values()))
print(f"[타격] eef_z 최대 하강 프레임: 중앙 {int(np.median(_st))} 범위 {_st.min()}~{_st.max()}")

rows = [("t=" + str(t), lambda i, t=t: t) for t in [int(x) for x in a.frames.split(",")]]
rows += [("타격-" + x, lambda i, o=int(x): STRIKE[i] - o) for x in a.strike_offsets.split(",")]

print(f"\n{'지점':>8} {'집합':>8} {'표본':>5} {'Q 평균':>9} {'Q std':>8} "
      f"{'r(Q,참값)':>11} {'ρ(Q,참값)':>11} {'ρ(Q,-길이)':>12}")
for label, fn in rows:
    sel = [(i, fr) for i, (_, fr, _) in enumerate(eps)
           if 0 <= fn(i) and len(fr) > fn(i) + LAT + R]
    if len(sel) < 10:
        print(f"{label:>8}  표본 부족 ({len(sel)})"); continue
    idx = np.array([fr[fn(i)] for i, fr in sel])
    t_of = np.array([fn(i) for i, _ in sel])
    with torch.no_grad():
        st = torch.from_numpy(snorm[idx]).to(dev)
        q = q_fn(idx, st).float().cpu().numpy()
    ii = np.array([i for i, _ in sel])
    truth = np.where(OKM[ii], G ** (L[ii] - t_of), 0.0)  # 참값 = γ^(T-t) (성공) / 0
    for name, m in (("전체", np.ones(len(ii), bool)),
                    ("학습", ~HOLD[ii]), ("홀드아웃", HOLD[ii])):
        if m.sum() < 5 or np.std(truth[m]) < 1e-9:
            continue
        r = float(np.corrcoef(q[m], truth[m])[0, 1])
        rho = spearman(q[m], truth[m])
        # 성공만 보고 "빠를수록 높은 Q" 인지 (길이 역순과의 순위상관)
        s = m & OKM[ii]
        rem = (L[ii] - t_of).astype(float)      # 남은 프레임 수 — 이것이 값을 정한다
        rho_len = spearman(q[s], -rem[s]) if s.sum() >= 5 else float("nan")
        print(f"{label:>8} {name:>8} {int(m.sum()):>5} {q[m].mean():9.4f} {q[m].std():8.4f} "
              f"{r:11.3f} {rho:11.3f} {rho_len:12.3f}")

print("\n읽는 법:\n"
      "  r/ρ(Q,참값)  : Q 가 실제 리턴 γ^(T-t) 를 얼마나 맞히는가. 홀드아웃 값이 진짜다.\n"
      "  ρ(Q,-길이)   : **성공 에피소드 안에서** 빠를수록 높은 Q 인가. 이것이 0 이면\n"
      "                 성공끼리 구분을 못 하는 것이고, 후보 32개 중 고르는 의미가 없다.\n"
      "  t=0 은 상태가 동일하지만 청크가 '뻗기' 라 완주 속도를 결정하지 않는다 —\n"
      "     여기서 상관이 낮은 것은 critic 결함이 아니라 사실이다.\n"
      "  **타격-N 행이 핵심이다.** 내려찍기 직전이 액션이 결과를 실제로 가르는 지점이고,\n"
      "     서빙에서도 20프레임마다 replan 하므로 선택이 그 순간에 일어난다.")
