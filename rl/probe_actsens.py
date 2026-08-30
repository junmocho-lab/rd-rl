#!/usr/bin/env python3
"""critic 이 **액션에 반응하기는 하는가** — Q 의 액션 민감도를 직접 잰다.

probe_actopt 은 "critic 이 액션을 어디로 밀고 싶은가"(상승 방향)를 보지만, 그 앞에
답해야 할 질문이 있다: `Q(s,a)` 가 `a` 에 대해 **변하기는 하는가**.

dexjoco 롤아웃 실측에서 서빙 로그의 `후보간Qstd` 가 0.0001 (Q 범위 [0.01,0.9]) 이었다.
후보 8개의 Q 가 소수점 4자리까지 같으면 argmax 는 랜덤 선택이고 guidance 도 방향
정보를 못 얻는다. 그 관찰이 학습 데이터에서도 재현되는지, 그리고 그것이 "액션이 정말
무관해서" 인지 "critic 이 액션 항을 학습하지 못해서" 인지 가른다.

재는 것 (전부 홀드아웃 프레임, explore_groups 의 실행 구간만 건드린다):

  logged    로그된 액션 그대로                     — 기준
  noise σ   로그 액션 + N(0, σ)                    — σ 를 훑어 민감도 곡선을 만든다
  shuffled  **다른 프레임**의 로그 액션을 붙여넣기  — 실제 로봇 액션인데 맥락이 틀림
  random    U(-1,1) 난수                           — 완전 OOD (천장)

읽는 법:
  · shuffled 의 ΔQ 가 0 에 가까우면 critic 은 액션을 **전혀** 보지 않는다.
    이때 Q(s,a) = V(s) 이고 test-time 액션 선택은 원리적으로 불가능하다.
  · ΔQ 를 "성공/실패 Q 격차"(홀드아웃 마지막 프레임 기준)와 비교해야 의미가 있다.
    격차의 1% 밖에 안 움직이면 후보 선택으로 결과를 바꿀 수 없다.
  · σ = 후보 청크의 실제 산포(dexjoco 실측 0.018/차원) 에서의 ΔQ 가 서빙의
    `후보간Qstd` 와 같은 크기여야 한다. 다르면 서빙/학습 경로가 어긋난 것이다.

usage:
  PYTHONPATH="$PWD/third_party/RLDX-1:$PWD" third_party/RLDX-1/.venv/bin/python -u \\
      -m rl.probe_actsens --exp dexjoco_hammer_nail --data rl-dataset/dexjoco/hammer_nail_d5r20 \\
      --checkpoints checkpoints --critic n1000_300k/critic_180000.pt \\
      --features cogfeat.npy
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
if torch.load(ck, map_location="cpu").get("kind") == "qvgm":
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

# 프레임 표본
k = np.concatenate([fr[::max(1, len(fr) // 32)] for _, fr, _ in eps])
rng = np.random.default_rng(0)
k = np.sort(rng.permutation(k)[:a.frames])
base_act = torch.from_numpy(np.ascontiguousarray(
    np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev)
q0, s0 = q_at(k)
print(f"[표본] 프레임 {len(k)}개  logged Q 중앙 {np.median(q0):+.4f}  "
      f"앙상블std 중앙 {np.median(s0):.4f}")

g = torch.Generator(device=dev).manual_seed(0)


def report(name, act, extra=""):
    q, s = q_at(k, act)
    d = np.abs(q - q0)
    mv = float((act - base_act)[:, IDX].abs().mean())
    print(f"  {name:<22} 이동 {mv:.4f}/차원  |ΔQ| 중앙 {np.median(d):.5f} "
          f"평균 {d.mean():.5f}  = 격차의 {100 * np.median(d) / GAP:5.2f}%  "
          f"| 앙상블std {np.median(s):.4f} ({np.median(s)/max(np.median(s0),1e-9):.1f}배){extra}")
    return float(np.median(d))


print(f"\n=== Q 의 액션 민감도 (|ΔQ| 를 성공/실패 격차 {GAP:+.4f} 의 절댓값으로 환산) ==="
      + ("\n    ** 격차가 음수다 — critic 이 실패에 더 높은 값을 준다. 순위가 뒤집혔다."
         if GAP < 0 else ""))
for sg in [float(x) for x in a.sigmas.split(",")]:
    act = base_act.clone()
    act[:, IDX] += torch.randn(len(k), NIDX, device=dev, generator=g) * sg
    report(f"noise σ={sg}", act.clamp(-1, 1))

sh = rng.permutation(k)
shv = torch.from_numpy(np.ascontiguousarray(
    np.asarray(norm[sh])[:, :LAT + R].reshape(len(sh), -1))).to(dev)[:, IDX]
act = base_act.clone(); act[:, IDX] = shv
report("shuffled (다른 프레임)", act, "  <- 실제 액션인데 맥락이 틀림")

act = base_act.clone()
act[:, IDX] = torch.rand(len(k), NIDX, device=dev, generator=g) * 2 - 1
report("random U(-1,1)", act, "  <- 완전 OOD (천장)")

print(f"\n판정: shuffled 의 |ΔQ| 가 격차의 몇 %인가가 핵심이다.\n"
      f"      < 5%  → critic 이 액션을 사실상 무시한다 (Q≈V). 액션 선택은 불가능하고,\n"
      f"              고칠 곳은 guidance 하이퍼파라미터가 아니라 critic 학습이다.\n"
      f"      > 20% → 액션 항은 학습되어 있다. 그러면 문제는 **후보 다양성**이다 —\n"
      f"              base 정책 샘플들이 너무 비슷해서 고를 것이 없는 것이다.")
