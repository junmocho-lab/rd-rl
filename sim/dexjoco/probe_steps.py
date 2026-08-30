#!/usr/bin/env python3
"""guidance 의 num_steps 가 실제로 의미가 있는지 잰다 (롤아웃 없이 오프라인).

우리 구현에서 총 이동량은 guide_move 로 고정이고 스텝 수와 무관하다:
    step = guide_move * sqrt(d) / num_steps ;  cur += step * grad/||grad||
따라서 num_steps 는 **경로 해상도**다. 1 이면 초기 그래디언트 방향으로 직선 점프,
많으면 Q 지형을 따라 휘어간다 (총 길이는 같다). keep-best 가 매 스텝 최댓값을 갱신하므로
스텝이 많으면 중간에 더 좋은 지점을 잡을 기회도 는다.

이전 근거(steps 4 와 10 의 ΔQ 가 동일)는 액션 625차원 critic 에서 잰 것이라,
90차원 + 성공만 학습한 지금 critic 에는 다시 확인해야 한다.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch, yaml

REPO = Path(__file__).resolve().parent.parent.parent
import sys; sys.path.insert(0, str(REPO))
p = argparse.ArgumentParser()
p.add_argument("--exp", default="dexjoco_hammer_nail_d2r8_s0")
p.add_argument("--data", type=Path, default=REPO / "rl-dataset/dexjoco/hammer_nail_d2r8_s0")
p.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
p.add_argument("--critic", default="arm_succ_20k/critic_020000.pt")
p.add_argument("--steps", default="1,2,4,10,20")
p.add_argument("--moves", default="0.02,0.05")
p.add_argument("--frames", type=int, default=512)
p.add_argument("--device", default="cuda")
a = p.parse_args()

from rl.critic_io import load_stepwise_critic
from rl.data import build_flat, find_sessions, resolve_modality
from rl.expo import ExpoConfig
from rl.nets import explore_spec
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
A_DIM, FULL = mod.action_dim, (LAT + R) * mod.action_dim

ck = Path(a.critic)
if not ck.is_file():
    ck = work / a.critic
SC = load_stepwise_critic(ck, work, snorm, dev=a.device)

# 편집 마스크: explore_groups x 실행 구간 (서빙과 같은 규약)
spec = explore_spec(mod.offsets("action"), list(exp["explore_groups"]), A_DIM, R, LAT)
MASK = torch.zeros(FULL, device=a.device)
MASK[torch.as_tensor(spec.index, device=a.device)] = 1.0
d = int(MASK.sum())

hold = np.isin(flat.episode, np.unique(flat.episode)[::10])
ok = flat.is_success[np.flatnonzero(hold)]
idx = np.flatnonzero(hold)
rng = np.random.default_rng(0)
k = np.sort(rng.permutation(idx)[:a.frames])
st = torch.from_numpy(snorm[k]).to(a.device)
lat = SC.latent_of(k, st)
A0 = torch.from_numpy(np.ascontiguousarray(
    np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(a.device)
qf = lambda x: SC.q_all(lat, x).min(0).values.sum(-1)
q0 = qf(A0)
print(f"[프로브] 홀드아웃 프레임 {len(k)}개  편집 {d}/{FULL}차원  Q0 중앙 {float(q0.median()):.4f}")


def ascend(move, nstep):
    """서빙의 _cog_guide 와 같은 규칙 (정규화 보폭 + keep-best)."""
    step = move * (d ** 0.5) / max(nstep, 1)
    best, bq, cur = A0.clone(), qf(A0).clone(), A0.clone()
    for _ in range(nstep):
        cur = cur.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            g, = torch.autograd.grad(SC.q_all(lat, cur).mean(0).sum(-1).sum(), cur)
        g = g * MASK
        cur = (cur.detach() + step * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(-1, 1)
        with torch.no_grad():
            qq = qf(cur)
        take = qq > bq
        best = torch.where(take[:, None], cur, best)
        bq = torch.maximum(bq, qq)
    with torch.no_grad():
        mv = float((best - A0)[:, MASK.bool()].abs().mean())
    return float((bq - q0).median()), mv, float((bq > q0).float().mean())


print(f"\n{'move':>7} {'steps':>6} {'ΔQ 중앙':>10} {'실제이동/차원':>13} {'개선된 프레임':>12}")
for mv in [float(x) for x in a.moves.split(",")]:
    for ns in [int(x) for x in a.steps.split(",")]:
        dq, mov, frac = ascend(mv, ns)
        print(f"{mv:7.3f} {ns:6d} {dq:10.5f} {mov:13.5f} {100*frac:11.1f}%")
print("\nΔQ 가 스텝 수에 따라 안 늘면 Q 가 국소적으로 선형이라는 뜻이고, 적은 스텝으로 충분하다.")
