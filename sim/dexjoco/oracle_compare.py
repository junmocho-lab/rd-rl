#!/usr/bin/env python3
"""2단계: 후보 32개 vs 오라클 다음 액션, 그리고 critic 이 오라클에 가까운 걸 고르는가.

1단계(oracle_walk.py)가 오라클 궤적 위에서 매 replan 의 후보 32개를 덤프했다. 여기서는
그 후보들을 오라클이 **실제로 할 20스텝** 과 비교한다.

재는 것:
  1) 후보 구름의 산포 (차원당 std) — 고를 거리가 있는가
  2) 각 후보와 오라클의 거리 분포 — 오라클이 구름 안에 있는가 (평균에서 몇 σ)
  3) 후보별 Q → argmax 가 오라클에 **더 가까운가**, 그리고 Q 와 거리의 순위상관
     상관이 뚜렷하게 음수여야 critic 이 옳은 방향을 보는 것이다

기준선: 오픈루프 리플레이는 액션이 차원당 0.005 만 틀어져도 성공률이 40% 로 떨어진다.
따라서 "오라클까지 거리 < 0.005" 인 후보가 성공 후보다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--walk", type=Path, default=Path("/workspace/junmo_cho/dexjoco/oracle_walk/walk.npz"))
p.add_argument("--cand", type=Path, default=Path("/workspace/junmo_cho/dexjoco/oracle_walk/cand.npz"))
p.add_argument("--exp", default="dexjoco_hammer_nail_d2r8_s0")
p.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
p.add_argument("--critic", default="fixed_successonly/critic_latest.pt")
p.add_argument("--device", default="cuda")
a = p.parse_args()

from rl.critic_io import load_stepwise_critic  # noqa: E402
from rl.expo import ExpoConfig  # noqa: E402

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
LAT, R, A_DIM = exp["inference_latency"], exp["replan_steps"], 25
work = a.checkpoints / f"{a.exp}-critic"

w = np.load(a.walk)
c = np.load(a.cand)
frames, oracle = w["frames"], w["oracle"]        # (C,), (C, 20, 25) 정책 출력(raw) 공간
acts, cog, state = c["acts"], c["cog"], c["state"]   # (C, N, 625) 정규화, (C, 4096), (C, 25)
C, N = acts.shape[0], acts.shape[1]
print(f"[데이터] 호출 {C}회 x 후보 {N}개   오라클 ep{int(w['episode'])} ({int(w['length'])}프레임)")
assert acts.shape[0] == len(frames), f"덤프 {acts.shape[0]} != walk {len(frames)}"

# 후보의 실행 구간 [LAT, LAT+R) 만 본다. 오라클도 같은 20스텝이다.
A = acts.reshape(C, N, LAT + R, A_DIM)[:, :, LAT:LAT + R, :]     # (C,N,20,25)

ck = Path(a.critic)
if not ck.is_file():
    ck = work / a.critic
SC = load_stepwise_critic(ck, work, np.zeros((1, state.shape[1]), np.float32), dev=a.device)

print(f"\n{'t':>5} {'후보산포':>9} {'오라클거리':>10} {'최근접':>8} {'최원거리':>8} "
      f"{'<0.005':>7} {'Qstd':>9} {'argmax거리':>10} {'ρ(Q,거리)':>10}")
rows = []
for i in range(C):
    cand = A[i].reshape(N, -1)                      # (N, 500)
    orc = oracle[i].reshape(-1)                     # (500,)
    spread = cand.std(0).mean()
    dist = np.abs(cand - orc[None]).mean(1)         # (N,) 차원당 거리
    with torch.no_grad():
        cg = torch.from_numpy(cog[i]).to(a.device)[None].repeat(N, 1)
        st = torch.from_numpy(state[i]).to(a.device)[None].repeat(N, 1)
        lat = SC.enc(((cg - SC.meta["feat_mu"].to(a.device)) / SC.meta["feat_sd"].to(a.device)), st) \
            if int(SC.meta.get("state_latent") or 0) > 0 else \
            torch.cat([SC.enc((cg - SC.meta["feat_mu"].to(a.device)) / SC.meta["feat_sd"].to(a.device)), st], -1)
        act_t = torch.from_numpy(acts[i]).to(a.device)
        q = SC.q_all(lat, act_t).min(0).values.sum(-1).float().cpu().numpy()
    k = int(np.argmax(q))
    rx = np.argsort(np.argsort(q)).astype(float)
    ry = np.argsort(np.argsort(dist)).astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    print(f"{frames[i]:>5} {spread:9.5f} {dist.mean():10.5f} {dist.min():8.5f} {dist.max():8.5f} "
          f"{int((dist < 0.005).sum()):>3}/{N:<3} {q.std():9.6f} {dist[k]:10.5f} {rho:10.3f}")
    rows.append((dist, q, k))

D = np.concatenate([r[0] for r in rows])
print(f"\n전체: 오라클까지 거리 중앙 {np.median(D):.5f}   0.005 미만 후보 "
      f"{int((D < 0.005).sum())}/{len(D)} ({100*(D<0.005).mean():.1f}%)")
best_r = np.mean([np.argsort(np.argsort(r[0]))[r[2]] for r in rows])
print(f"critic argmax 의 '오라클 근접도' 순위 평균 {best_r:.1f}/{N-1}  "
      f"(무작위면 {(N-1)/2:.1f}, 완벽하면 0)")
print(f"ρ(Q, 오라클거리) 평균 {np.mean([np.corrcoef(np.argsort(np.argsort(r[1])).astype(float), np.argsort(np.argsort(r[0])).astype(float))[0,1] for r in rows]):+.3f}"
      f"   (음수일수록 Q 가 오라클에 가까운 후보를 선호한다는 뜻)")
