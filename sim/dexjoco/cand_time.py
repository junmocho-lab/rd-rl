#!/usr/bin/env python3
"""후보 산포와 후보간 Q 분산을 **에피소드 내 호출 순서별로** 잰다.

왜 순서별인가: 첫 호출(cold start)은 RTC prefix 가 없어 후보가 자유롭게 갈리고, 이후
호출은 앞 d 스텝이 고정된다. 그리고 국면(접근 / 파지 / 타격)마다 정책의 확신도가 달라
후보 다양성도 달라질 수 있다. 전체 중앙값 하나로는 그것이 뭉개진다.

에피소드 경계는 **prefix 산포가 0 이 아닌 호출** = cold start 로 찾는다 (RTC 가 prefix 를
고정하면 후보들의 앞 d 스텝이 완전히 같아져 std 가 정확히 0 이 된다).

  python sim/dexjoco/cand_time.py --dump <cand.npz> --critic <경로> --exp <exp>
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

p = argparse.ArgumentParser()
p.add_argument("--dump", type=Path, required=True)
p.add_argument("--exp", default="dexjoco_hammer_nail_d2r8_s0")
p.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
p.add_argument("--critic", default="arm_200k/critic_040000.pt")
p.add_argument("--bins", default="0,1,2,4,8,16,32,999", help="호출 순서 구간 경계")
p.add_argument("--device", default="cuda")
a = p.parse_args()

from rl.critic_io import load_stepwise_critic  # noqa: E402
exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
LAT, R, A = exp["inference_latency"], exp["replan_steps"], 25
work = a.checkpoints / f"{a.exp}-critic"

d = np.load(a.dump)
acts, cog, state = d["acts"], d["cog"], d["state"]
C, N, FULL = acts.shape
W = FULL // A
a4 = acts.reshape(C, N, W, A)
pre_std = a4[:, :, :LAT, :].std(1).mean(axis=(1, 2))     # 0 이면 RTC 가 고정한 것
exe_std = a4[:, :, LAT:, :].std(1).mean(axis=(1, 2))
cold = pre_std > 1e-6
# 호출 순서: 마지막 cold start 이후 몇 번째인가
idx_in_ep = np.zeros(C, int); k = 0
for i in range(C):
    k = 0 if cold[i] else k + 1
    idx_in_ep[i] = k
print(f"호출 {C}회 x 후보 {N}개  창 {W}스텝({LAT}+{R}) x {A}관절")
print(f"에피소드 시작(cold start) {int(cold.sum())}회 → 에피소드당 평균 {C/max(cold.sum(),1):.1f} 호출\n")

ck = Path(a.critic)
if not ck.is_file():
    ck = work / a.critic
SC = load_stepwise_critic(ck, work, np.zeros((1, state.shape[1]), np.float32), dev=a.device)
mu, sd_ = SC.meta["feat_mu"].to(a.device), SC.meta["feat_sd"].to(a.device)
qstd = np.zeros(C); qrng = np.zeros(C); qmed = np.zeros(C)
for i in range(C):
    with torch.no_grad():
        cg = torch.from_numpy(cog[i]).to(a.device)[None].repeat(N, 1)
        st = torch.from_numpy(state[i]).to(a.device)[None].repeat(N, 1)
        z = (cg - mu) / sd_
        lat = (SC.enc(z, st) if int(SC.meta.get("state_latent") or 0) > 0
               else torch.cat([SC.enc(z), st], -1))
        q = SC.q_all(lat, torch.from_numpy(acts[i]).to(a.device)).min(0).values.sum(-1)
        q = q.float().cpu().numpy()
    qstd[i], qrng[i], qmed[i] = q.std(), q.max() - q.min(), np.median(q)

E = [int(x) for x in a.bins.split(",")]
print(f"{'호출 순서':>12} {'n':>4} {'액션산포':>10} {'후보간Qstd':>12} {'Q범위':>10} {'Q중앙':>9}")
for lo, hi in zip(E, E[1:]):
    m = (idx_in_ep >= lo) & (idx_in_ep < hi)
    if not m.any(): continue
    lab = f"{lo}" if hi == lo + 1 else f"{lo}~{hi-1}"
    print(f"{lab:>12} {int(m.sum()):>4} {np.median(exe_std[m]):10.5f} "
          f"{np.median(qstd[m]):12.6f} {np.median(qrng[m]):10.6f} {np.median(qmed[m]):9.4f}")
print(f"\n{'전체':>12} {C:>4} {np.median(exe_std):10.5f} {np.median(qstd):12.6f} "
      f"{np.median(qrng):10.6f} {np.median(qmed):9.4f}")
print(f"\n참고: 오픈루프 리플레이 실측 — 액션을 차원당 0.005 흔들면 성공률 100%->40%, 0.01 이면 10%.")
print(f"      후보 산포가 그 임계 근처여야 후보들 사이에 성공/실패가 실제로 갈린다.")
