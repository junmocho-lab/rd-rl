#!/usr/bin/env python3
"""체크포인트별로 **일반화 지표**를 재서 과적합 시작점을 찾는다.

성공 에피소드만으로 학습한 critic 을 평가할 때 AUC 는 쓸모가 없다 — 실패를 본 적이 없어
외삽으로 높은 값을 주기 때문이다. 대신 봐야 할 것:

  ρ(Q, -남은프레임) 홀드아웃 : 안 본 성공 에피소드에서 **빨리 끝나는 것에 높은 Q** 를 주는가.
                               성공만 학습하면 Q 의 참값이 γ^(T-t) 이므로 이것이 곧 정확도다.
  ρ 학습 - ρ 홀드            : 벌어지면 외우는 중이다.
  지터                       : 프레임 t 와 t+1 은 0.02초 차이라 거의 같은 상태다. 값이 튀면
                               cog feature 의 노이즈에 적합하고 있다는 뜻이고, guidance 가
                               쓰는 ∇_A Q 가 잡음의 기울기가 된다.
  σ=0.018 민감도             : 실전 후보 폭에서 Q 가 갈리는 정도. **지터와 같이 읽어야 한다** —
                               함수가 뾰족해져도 같이 오르므로 그것만 보면 가짜 진전을 놓친다.
"""
from __future__ import annotations
import argparse, glob, re
from pathlib import Path
import numpy as np, torch, yaml

REPO = Path(__file__).resolve().parent.parent.parent
import sys; sys.path.insert(0, str(REPO))
p = argparse.ArgumentParser()
p.add_argument("--exp", default="dexjoco_hammer_nail_d2r8_s0")
p.add_argument("--data", type=Path, default=REPO / "rl-dataset/dexjoco/hammer_nail_d2r8_s0")
p.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
p.add_argument("--run", default="arm_succ_20k")
p.add_argument("--holdout", type=float, default=0.1)
p.add_argument("--device", default="cuda")
a = p.parse_args()

from rl.critic_io import load_stepwise_critic
from rl.data import build_flat, find_sessions, resolve_modality
from rl.expo import ExpoConfig
from rl.vla_rldx import load_state_action_processor, normalize_states

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
LAT, R = exp["inference_latency"], exp["replan_steps"]
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / exp["base_policy"]
mod, _ = resolve_modality(a.data, None, REPO / "third_party/RLDX-1", exp["rldx_data_config"], base)
flat = build_flat(find_sessions(a.data), mod)
norm = np.load(work / "actnorm.npy", mmap_mode="r")
proc = load_state_action_processor(base, REPO / "third_party/RLDX-1", exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)

eps = [(e, np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode)]
eps = [(e, fr, bool(flat.is_success[fr[-1]])) for e, fr in eps]
every = max(2, int(round(1 / a.holdout)))
hold = set(np.unique(flat.episode)[::every].tolist())
HS = [(e, fr) for e, fr, o in eps if o and e in hold]
TS = [(e, fr) for e, fr, o in eps if o and e not in hold][:40]
HA = [(e, fr, o) for e, fr, o in eps if e in hold]
print(f"홀드아웃 성공 {len(HS)}  학습 성공(표본) {len(TS)}  홀드아웃 전체 {len(HA)}")


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


rng = np.random.default_rng(0)
print(f"\n{'step':>6} {'ρ홀드':>8} {'ρ학습':>8} {'격차':>7} {'지터%':>7} "
      f"{'σ.018':>8} {'shuffled':>9}")
for ck in sorted(glob.glob(str(work / a.run / "critic_[0-9]*.pt"))):
    step = int(re.findall(r"(\d+)\.pt", ck)[0])
    SC = load_stepwise_critic(Path(ck), work, snorm, dev=a.device)

    def qs(fr):
        with torch.no_grad():
            st = torch.from_numpy(snorm[fr]).to(a.device)
            act = torch.from_numpy(np.ascontiguousarray(
                np.asarray(norm[fr])[:, :LAT + R].reshape(len(fr), -1))).to(a.device)
            return SC.q_all(SC.latent_of(fr, st), act).min(0).values.sum(-1).float().cpu().numpy()

    rho = {}
    for nm, S in (("h", HS), ("t", TS)):
        q, rem = [], []
        for e, fr in S:
            v = qs(fr); k = np.arange(0, len(fr), 5)
            q.append(v[k]); rem.append((len(fr) - k).astype(float))
        rho[nm] = spearman(np.concatenate(q), -np.concatenate(rem))

    fin = np.array([qs(fr)[-1] for _, fr, _ in HA])
    okm = np.array([o for _, _, o in HA])
    gap = abs(fin[okm].mean() - fin[~okm].mean()) if okm.any() and (~okm).any() else 1.0

    k = np.concatenate([fr[::max(1, len(fr) // 16)] for _, fr in HS])[:1024]
    with torch.no_grad():
        st = torch.from_numpy(snorm[k]).to(a.device)
        lat = SC.latent_of(k, st)
        a0 = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(a.device)
        f = lambda x: SC.q_all(lat, x).min(0).values.sum(-1).float().cpu().numpy()
        q0 = f(a0)
        g = torch.Generator(device=a.device).manual_seed(0)
        qn = f((a0 + torch.randn(a0.shape, device=a.device, generator=g) * 0.018).clamp(-1, 1))
        qsh = f(a0[torch.from_numpy(rng.permutation(len(k))).to(a.device)])
    jit = np.median([np.median(np.abs(np.diff(qs(fr)))) for _, fr in HS[:8]])
    rgm = np.median([qs(fr).max() - qs(fr).min() for _, fr in HS[:8]])
    print(f"{step:>6} {rho['h']:>8.3f} {rho['t']:>8.3f} {rho['t']-rho['h']:>7.3f} "
          f"{100*jit/max(rgm,1e-9):>6.2f}% {100*np.median(np.abs(qn-q0))/gap:>7.2f}% "
          f"{100*np.median(np.abs(qsh-q0))/gap:>8.1f}%", flush=True)
print("\n고를 체크포인트: ρ홀드가 정점이고 격차·지터가 아직 낮은 지점.")
