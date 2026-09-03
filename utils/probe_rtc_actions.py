#!/usr/bin/env python3
"""에피소드 몇 개를 **RTC 결정 시점마다** 열어서 액션이 얼마나 다른지 재고 그린다.

  리플레이(로그)   그 프레임에서 실제로 실행된 액션            <- 기준
  BC 샘플 32개     base policy 에서 그냥 뽑은 것, mean±std     <- 정책의 자연 산포
  PA-RL 최적화     그 32개 -> Q 상위 9개 + 로그 1개 -> ∇_A Q 상승 -> argmax

핵심은 **두 번째가 기준선 역할을 한다**는 것이다. BC 샘플 32개를 하나하나 리플레이와
비교하면 "정책이 스스로와 얼마나 불일치하는가" 의 분포가 나온다 (평균±표준편차 밴드).
|최적화 - 리플레이| 가 그 밴드 안에 들어가면 최적화는 **그냥 다시 뽑은 것과 구별되지
않는다** — 그때는 step_size 를 올릴 문제가 아니라 critic 을 고쳐야 한다.
그 판단을 하려고 만든 스크립트다. 한 장만 뽑으면 그 한 장이 우연히 멀었는지 가까웠는지
알 수 없어서 32개 전부를 쓴다 (후보 샘플링에서 이미 뽑는 것들이라 추가 비용이 없다).

**에피소드 하나로는 판단할 수 없다.** base policy 는 teleop A 세션만으로 학습됐고
critic 은 롤아웃/HiL 을 포함한 전체의 성공분으로 학습됐다. 그래서 "리플레이 액션" 의
의미가 세션마다 다르다 — teleop 프레임에서는 정책의 학습 타깃 그 자체이고, 롤아웃
프레임에서는 정책의 과거 출력이다. 기본값은 세션을 돌아가며 성공 에피소드를 뽑아
그 두 경우가 섞이게 한다.

측정 지표: **L1 평균** = mean|Δ| over (right_arm 7관절 x 실행 16스텝).
    즉 "관절 하나가 한 스텝에 평균 몇만큼 다른가" 다. L2 노름과 달리 차원 수나 한두
    관절의 큰 차이에 휘둘리지 않고, raw 로 환산하면 곧바로 **평균 절대 관절 차이[rad]**
    가 된다. 그래서 요약표에 raw 라디안을 같이 찍는다 — 정규화 공간의 0.01 이 몇 rad
    인지 감이 없으면 어떤 숫자도 해석할 수 없다.
    측정 열은 explore_spec(index) 가 정하는 그 열이다 (prefix LAT 스텝은 이미 커밋된
    구간이라 제외).

RTC 타이밍 (d=4, r=16 -> 청크 20):
    t=0    에서 청크 20스텝을 받아 실행. 프레임 16 (4스텝 남음)에서 다음 추론 시작
    t=16   앞 4스텝은 이미 커밋된 값(prefix)이고 뒤 16스텝이 새 것 -> 프레임 20..35
    t=32   ... 이후 replan 16 간격으로 반복
  즉 결정 프레임은 t = ep0, ep0+16, ep0+32, ... 이고 결정 t 가 프레임
  t+4 … t+19 를 지배한다. 앞 4프레임만 원본이 남는다.
  rl/relabel_parl.py:163-181 과 **같은 규칙**이다 (relabel 이 실제로 쓰는 정렬).

step_size 를 여러 개 주면 후보 샘플링(비싼 쪽, VLA forward)을 한 번만 하고 상승만
다시 돌린다. 그래서 0 부터 쓸어보는 비용이 거의 공짜다 — "0 부터 조절해가며 리플레이
액션과 얼마나 다른지" 를 한 번의 잡으로 본다.

산출물 (<checkpoints>/<exp>-critic/<tag>/plots/probe/):
    <run>/ep<N>.mp4       에피소드별. 왼쪽 카메라 + 오른쪽 2단 (거리 / Q), 빨간 커서
    <run>/ep<N>.png       같은 그림의 정지판
    <run>/summary.png     ★ x=step_size 로 접은 종합판. 여기서 step_size 를 고른다
    <run>/summary.json    표에 찍은 수치 전부 (나중에 critic 을 바꿔 비교할 때 쓴다)
  <run> = <tag>@<Nk>_ns<num_steps>_<에피소드수>ep

사용:
  PY=third_party/RLDX-1/.venv/bin/python
  PYTHONPATH=third_party/RLDX-1:. $PY utils/probe_rtc_actions.py \
      --exp fuji_d4r16 --tag success --critic-step 10000 \
      --episodes auto:6 --step-sizes 0,1e-4,3e-4,1e-3,3e-3,1e-2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for _f in ("Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="fuji_d4r16")
p.add_argument("--tag", default="success", help="critic 학습 태그 (success | all)")
p.add_argument("--critic-step", type=int, default=10000, help="0 이면 critic_latest.pt")
p.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
p.add_argument("--episodes", default="auto:6",
               help="'auto:N'  세션을 돌아가며 성공 에피소드 N개 (기본, 세션 편향을 피한다)\n"
                    "'succ:N'  성공 에피소드 중 길이 분위 N개\n"
                    "'fail:N'  실패 에피소드 N개\n"
                    "'12,45,88' 전역 에피소드 번호 직접 지정\n"
                    "'all'      전부 (오래 걸린다)")
p.add_argument("--list", action="store_true", help="에피소드 목록만 찍고 끝낸다")
p.add_argument("--step-sizes", default="0,3e-4,1e-2,1,10,100",
               help="쓸어볼 step_size 목록. 0 은 '상승 없음'(= 선택만) 이라 선택 효과와\n"
                    "상승 효과를 분리해서 볼 수 있다.\n"
                    "★ PA-RL 원본은 3e-4 지만 그쪽 Q 는 [-100,0] 이고 우리는 [0,1] 이다.\n"
                    "  fuji 실측 |∂Q/∂a|=1.2e-4 에서는 액션을 1프레임치(0.0074) 움직이려면\n"
                    "  step_size ~6 이 필요하다. 그래서 1/10/100 까지 넣는다 — 약한\n"
                    "  gradient 가 **증폭하면 신호인가** 를 재려면 실제로 움직여 봐야 한다.")
p.add_argument("--num-steps", type=int, default=10, help="PA-RL local optimization steps")
p.add_argument("--num-samples", type=int, default=32)
p.add_argument("--num-keep", type=int, default=10)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--no-video", action="store_true", help="에피소드별 mp4 를 건너뛴다 (빠르다)")
p.add_argument("--cam-name", default="", help="비우면 modality 의 첫 카메라")
p.add_argument("--fps", type=int, default=30)
p.add_argument("--row-h", type=int, default=260)
p.add_argument("--plot-w", type=int, default=1000)
p.add_argument("--legend-w", type=int, default=290,
               help="범례를 축 **밖** 오른쪽에 둘 폭. 축 안에 두면 곡선을 가린다")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda")
a = p.parse_args()
dev = a.device
torch.manual_seed(a.seed)
gen = torch.Generator(device=dev).manual_seed(a.seed)

from rl.data import (build_flat, build_images, find_sessions, open_images,  # noqa: E402
                     resolve_modality)
from rl.nets import explore_spec  # noqa: E402
from rl.offline_critic import normalize_all  # noqa: E402
from rl.vla_rldx import load_state_action_processor, normalize_states  # noqa: E402

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
data = REPO / exp["dataset"]
base = a.checkpoints / exp["base_policy"]
work = a.checkpoints / f"{a.exp}-critic"
groups = list(exp["explore_groups"])
G = float(exp["expo"]["discount"])

sessions = find_sessions(data)
mod, _ = resolve_modality(data, None, RLDX, exp["rldx_data_config"], base)
flat = build_flat(sessions, mod)
build_images(sessions, flat, work / "images.mm", mod)
imgs, _meta = open_images(work / "images.mm")
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
A_DIM = mod.action_dim
FULL, PRE = (LAT + R) * A_DIM, LAT * A_DIM
task = json.loads((sessions[0] / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]

if not (work / "actnorm.npy").is_file():
    raise SystemExit(f"actnorm.npy 가 없다: {work} — critic 을 한 번 학습해 캐시를 만들 것")
norm = normalize_all(None, flat, H, cache=work / "actnorm.npy")

spec = explore_spec(mod.offsets("action"), groups, A_DIM, R, LAT)
MASK = torch.zeros(FULL, device=dev)
MASK[spec.index] = 1.0
NIDX = len(spec.index)
SIDX = spec.index.to(dev)
# raw 라디안 환산용: canonical action 배열 안의 right_arm 열.
RAWCOL = np.concatenate([np.arange(s, e) for n, s, e in mod.offsets("action") if n in groups])

# ── 에피소드 선택 ────────────────────────────────────────────────────────────
EPS = [(int(e), np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode)]
EPS = [(e, fr, bool(flat.is_success[fr[-1]]), int(flat.session[fr[0]])) for e, fr in EPS]
if a.list:
    for e, fr, ok, si in EPS:
        print(f"  ep{e:4d}  {len(fr):5d}f  {'성공' if ok else '실패'}  {sessions[si].name}")
    raise SystemExit(0)


def spread(items, n):
    """길이 분위로 n개를 고르게 뽑는다 (짧은 것/긴 것/중간이 다 들어가게)."""
    items = sorted(items, key=lambda x: len(x[1]))
    if n >= len(items):
        return items
    return [items[int(round(i * (len(items) - 1) / (n - 1)))] for i in range(n)] if n > 1 \
        else [items[len(items) // 2]]


sp = a.episodes.strip()
if sp == "all":
    PICK = EPS
elif ":" in sp:
    kind, n = sp.split(":")
    n = int(n)
    if kind == "auto":
        # 세션을 **돌아가며** 뽑는다. base policy 는 teleop A 만으로 학습됐고 critic 은
        # 롤아웃/HiL 까지 봤으므로, 한 세션에서만 뽑으면 그 편향이 결과를 정한다.
        by = {}
        for x in EPS:
            if x[2]:
                by.setdefault(x[3], []).append(x)
        if not by:
            raise SystemExit("성공 에피소드가 없다 — --episodes succ:N 대신 fail:N 이나 번호 지정")
        order = sorted(by, key=lambda s: -len(by[s]))
        PICK, i = [], 0
        while len(PICK) < n and any(by.values()):
            s_ = order[i % len(order)]
            i += 1
            if by[s_]:
                cand = spread(by[s_], 1)[0]           # 그 세션의 중앙 길이
                by[s_].remove(cand)
                PICK.append(cand)
        PICK = sorted(PICK, key=lambda x: x[0])
    elif kind in ("succ", "fail"):
        PICK = spread([x for x in EPS if x[2] == (kind == "succ")], n)
    else:
        raise SystemExit(f"모르는 선택 방식: {kind} (auto | succ | fail)")
else:
    want = [int(t) for t in sp.split(",") if t.strip()]
    hit = {x[0]: x for x in EPS}
    miss = [e for e in want if e not in hit]
    if miss:
        raise SystemExit(f"없는 에피소드: {miss} — --list 로 확인할 것")
    PICK = [hit[e] for e in want]
if not PICK:
    raise SystemExit(f"--episodes {sp} 로 아무것도 안 뽑혔다")

print(f"[에피소드] {len(PICK)}개  (--episodes {sp})")
for e, fr, ok, si in PICK:
    print(f"  ep{e:4d}  {len(fr):5d}f  {'성공' if ok else '실패'}  {sessions[si].name}")

# ── critic ───────────────────────────────────────────────────────────────────
cf = "critic_latest.pt" if a.critic_step == 0 else f"critic_{a.critic_step:06d}.pt"
ck = work / a.tag / cf
if not ck.is_file():
    have = sorted(x.name for x in (work / a.tag).glob("critic_*.pt"))
    raise SystemExit(f"critic 이 없다: {ck}\n있는 것: {have}")
_sd = torch.load(ck, map_location="cpu")
if _sd.get("kind") != "qvgm":
    raise SystemExit(f"qvgm critic 이 아니다: {ck}")
CINFO = {"num_qs": _sd.get("num_qs"), "step": _sd.get("step"), "discount": _sd.get("discount")}
del _sd
from rl.critic_io import load_stepwise_critic  # noqa: E402

C = load_stepwise_critic(ck, work, snorm, dev=dev)


# PA-RL 원본과 같은 집계: 상승 방향도 선택도 앙상블 mean
# (action_optimization.py — optimize_critic_ensemble_min=False, :365 의 .mean(axis=0)).
# rl/relabel_parl.py:135-154 와 **같아야 한다** — 여기서 잰 수치로 저기 파라미터를 정한다.
def q_mean(lat, act):
    return C.q_all(lat, act).mean(0).sum(-1)


q_sel = q_mean

STEPS = [float(s) for s in a.step_sizes.split(",") if s.strip() != ""]
print(f"\n[측정 범위] {groups} → 액션 {FULL}차원 중 {NIDX}개 "
      f"({spec.active_dim}관절 x {R}스텝, prefix {LAT}스텝 제외)")
print(f"[지표] L1 평균 mean|Δ| over ({spec.active_dim}관절 x {R}스텝)")
print(f"[step_size] {STEPS}  (num_steps={a.num_steps}, raw gradient, keep-best)")

# ── base policy ──────────────────────────────────────────────────────────────
from rl.vla_rldx import RLDXVLA  # noqa: E402

vla = RLDXVLA(base, mod, RLDX, exp["rldx_data_config"], device=dev)
print(f"[base policy] {base.name}  M={a.num_samples} K={a.num_keep}")

cams = [c for c, _ in mod.video]
cam_name = a.cam_name or cams[0]
if cam_name not in cams:
    raise SystemExit(f"카메라 '{cam_name}' 가 없다. 가능: {cams}")
cam_key = dict(mod.video)[cam_name]

RUN = f"{a.tag}@{a.critic_step // 1000}k_ns{a.num_steps}_{len(PICK)}ep"
OUT = work / a.tag / "plots/probe" / RUN
OUT.mkdir(parents=True, exist_ok=True)


def vla_obs(idx):
    x = np.asarray(imgs[idx])
    return {"video": {name: x[:, c][:, None] for c, (name, _) in enumerate(mod.video)},
            "state": {name: flat.state[idx][:, None, s:e] for name, s, e in mod.offsets("state")},
            "language": {mod.task_key: [[task]] * len(idx)}}


def l1(x):
    """(..., FULL) 두 액션 차이 -> L1 평균 mean|Δ| (right_arm x 실행스텝).

    "관절 하나가 한 스텝에 평균 몇만큼 다른가". L2 를 쓰면 차원 수(112)와 한두 관절의
    큰 차이에 값이 끌려가는데, 여기서 알고 싶은 것은 전형적인 관절 차이의 크기다.
    """
    return x[..., SIDX].abs().mean(-1)


def raw_l1(x, y):
    """(B, LAT+R, A) 두 raw 청크 -> 관절당 평균 절대차이[rad] (right_arm x 실행스텝)."""
    return np.abs((x - y)[:, LAT:, :][:, :, RAWCOL]).reshape(len(x), -1).mean(1)


def ascend(cand, lat, ss):
    """(B,K,FULL) 후보를 ∇_A Q 로 올린다. rl/relabel_parl.py:231-264 와 같은 절차.

    keep-best: 매 스텝 Q 가 실제로 올랐을 때만 채택한다. 없으면 마지막 스텝의
    과도한 이동을 그대로 받는다 (dexjoco 에서 89% -> 72% 로 무너진 실패 모드).
    """
    b, K = cand.shape[:2]
    rl_ = lat.repeat_interleave(K, 0)
    best = cand.reshape(b * K, FULL)
    with torch.no_grad():
        bq = q_sel(rl_, best)
    if ss == 0:
        return best.view(b, K, FULL), bq.view(b, K)      # 상승 없음 = 선택만
    cur = best
    for _ in range(a.num_steps):
        cur = cur.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            g, = torch.autograd.grad(q_mean(rl_, cur).sum(), cur)
        # raw gradient (정규화 없음) — PA-RL action_optimization.py:129 와 동일
        cur = (cur.detach() + ss * (g * MASK)).clamp(-1.0, 1.0)
        with torch.no_grad():
            qq = q_sel(rl_, cur)
        best = torch.where((qq > bq)[:, None], cur.detach(), best)
        bq = torch.maximum(bq, qq)
    return best.view(b, K, FULL), bq.view(b, K)


def to_frames(dq, q):
    """ΔQ 를 '몇 프레임 빨리 끝난다고 critic 이 믿는가' 로 환산한다.

    보상이 종료 시점 1회뿐이라 성공 궤적의 참값은 V = γ^(T-t) 다. 그러므로
        T-t = ln V / ln γ,      dT = dV / (V ln γ)
    ΔQ 자체는 크기 감각이 없는 숫자지만(우리 Q 는 [0,1]) 프레임으로 바꾸면
    "이 최적화가 몇 프레임을 벌어 주는가" 라는 물리적 질문에 바로 답한다.
    """
    return float(dq / max(q * abs(np.log(G)), 1e-12))


def sanity(dec_all):
    """**critic 이 액션을 보기는 하는가, 그리고 그 방향이 신호인가.**

    ∇_A Q 가 정확히 0 인 경우는 없다 (실측 1.2e-4). 그러니 진짜 질문은 두 개다:
      (1) 크기 — 액션을 끝에서 끝까지 흔들면 Q 가 얼마나 변하나. 그 변화를 프레임으로
          바꾸면 "이 액션이 몇 프레임을 벌어 주는가" 가 된다.
      (2) 방향 — 그 작은 gradient 가 **일관된 방향**인가. 두 가지로 가른다:
          (a) 앙상블 헤드들이 같은 방향을 가리키는가 (cos > 0 이면 공통 신호)
          (b) BC 샘플에서 출발해 ∇_A Q 를 따라가면 **로그된(성공한) 액션 쪽**으로
              가는가. "빨리 끝나는 쪽으로 밀어 준다" 는 가설이 맞다면 성공 시연
              액션이 바로 그 방향이므로 cos > 0 이어야 한다.
      cos ≈ 0 이면 가중치가 학습 압력을 못 받아 남은 초기값/드리프트라는 뜻이고,
      그것을 step_size 로 증폭하면 무작위 방향으로 미는 것이 된다 — 해롭다.
    """
    k = dec_all[:: max(1, len(dec_all) // 64)][:64]
    st = torch.from_numpy(snorm[k]).to(dev)
    lat = C.latent_of(k, st)
    a0 = torch.from_numpy(np.ascontiguousarray(
        np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev)
    with torch.no_grad():
        q0 = q_sel(lat, a0)
    a0g = a0.clone().requires_grad_(True)
    g, = torch.autograd.grad(q_mean(lat, a0g).sum(), a0g)
    g = g * MASK
    gn = g[:, SIDX].abs().mean(-1)                   # 차원당 |∂Q/∂a| (L1 평균)
    qm0, qsd = float(q0.mean()), float(q0.std())
    tgo = float(np.log(max(qm0, 1e-12)) / np.log(G))
    print(f"\n[민감도] 결정 {len(k)}개 표본")
    print(f"  Q(로그된 액션)  평균 {qm0:.5f}  시간축 std {qsd:.5f}"
          f"   <- 이 std 가 '신호' 의 크기다")
    print(f"  = critic 이 믿는 남은 프레임 {tgo:.0f}f  (V=γ^(T-t), γ={G})")
    print(f"  |∂Q/∂a| 차원당  평균 {gn.mean():.3e}")
    ss_nat = float(NAT_HINT / max(a.num_steps * float(gn.mean()), 1e-12))
    print(f"  -> 액션을 '1프레임 자연변화'({NAT_HINT:.5f})만큼 움직이려면 "
          f"step_size ≈ {ss_nat:.3g} 가 필요하다 (PA-RL 기본값 3e-4 의 {ss_nat/3e-4:.0f}배)")

    print(f"\n  (1) 크기 — 액션을 δ 만큼 흔들었을 때")
    print(f"  {'δ':<10}{'|ΔQ|':>12}{'Q std 대비':>12}{'= 몇 프레임':>13}   판정")
    rows = []
    for d in (0.001, 0.01, 0.05, 0.2, 0.5, 1.0):
        sg = torch.randint(0, 2, a0.shape, device=dev, generator=gen).float() * 2 - 1
        ap = (a0 + d * sg * MASK).clamp(-1.0, 1.0)
        with torch.no_grad():
            dq = (q_sel(lat, ap) - q0).abs()
        rel = float(dq.mean() / max(qsd, 1e-12))
        fr = to_frames(float(dq.mean()), qm0)
        rows.append({"delta": d, "abs_dq": float(dq.mean()), "rel": rel, "frames": fr})
        print(f"  {d:<10.3f}{dq.mean():>12.3e}{rel:>11.2%}{fr:>12.2f}f   "
              f"{'무시할 수준' if rel < 0.01 else ('약함' if rel < 0.1 else '반응함')}")
    print(f"     읽는 법: δ=1.0 은 액션 공간(±1) 끝에서 끝까지다. 그 극단에서도 '몇 프레임'")
    print(f"     이 남은 {tgo:.0f}f 에 비해 무시할 만하면, critic 은 **액션이 완주 속도를")
    print(f"     바꾸지 않는다**고 배운 것이다 — 전부 성공인 데이터에서는 옳은 추론이다.")

    # (2) 방향 — 앙상블 간 일치
    ghead = []
    for h in range(C.meta["num_qs"]):
        ag = a0.clone().requires_grad_(True)
        qh = C.q_all(lat, ag)[h].sum()
        gh, = torch.autograd.grad(qh, ag)
        ghead.append((gh * MASK)[:, SIDX])
    cos_ens = []
    for i_ in range(len(ghead)):
        for j_ in range(i_ + 1, len(ghead)):
            cos_ens.append(float(torch.nn.functional.cosine_similarity(
                ghead[i_], ghead[j_], dim=-1).mean()))
    print(f"\n  (2) 방향 — ∇_A Q 가 일관된 방향인가")
    print(f"  앙상블 헤드 간 cos       {np.mean(cos_ens):+.4f}"
          f"   ({len(ghead)}개 헤드, {len(cos_ens)}쌍)")

    # (2b) BC 샘플에서 로그된(성공) 액션 쪽을 가리키는가
    cos_log = []
    for c0 in range(0, len(k), 8):
        kk = k[c0:c0 + 8]
        b = len(kk)
        lg = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[kk])[:, :LAT + R].reshape(b, -1))).to(dev)
        with torch.no_grad():
            ch = vla.sample(vla_obs(kk), 4)
        ab = ch[:, :, :LAT + R].reshape(b, 4, FULL).float()
        if PRE:
            ab[:, :, :PRE] = lg[:, None, :PRE]
        ab = ab.reshape(b * 4, FULL)
        sk = torch.from_numpy(snorm[kk]).to(dev)
        lk = C.latent_of(kk, sk).repeat_interleave(4, 0)
        abg = ab.clone().requires_grad_(True)
        gb, = torch.autograd.grad(q_mean(lk, abg).sum(), abg)
        gb = (gb * MASK)[:, SIDX]
        toward = (lg.repeat_interleave(4, 0) - ab)[:, SIDX]   # BC -> 로그된 액션 방향
        cos_log.append(float(torch.nn.functional.cosine_similarity(
            gb, toward, dim=-1).mean()))
    cl = float(np.mean(cos_log))
    print(f"  BC샘플 -> 로그액션 과 cos {cl:+.4f}"
          f"   (>0 이면 '성공 시연 쪽으로 민다' = 신호, ~0 이면 무작위 방향)")

    # (3) 그 방향의 **정체**. 앙상블이 일치하면(cos_ens 높음) 방향은 실재한다 —
    #     그러면 "무엇을 향한 방향인가" 를 물어야 한다. 후보 가설 두 개를 직접 잰다.
    cs = torch.nn.functional.cosine_similarity
    gsel = g[:, SIDX]
    #  (a) 멈추는 방향?  액션은 use_relative_action=true 라 크기가 곧 '얼마나 움직이나'다.
    #      데이터에서 에피소드 끝에 가까울수록 감속하므로 '작은 액션 = 진행됨 = 높은 Q'
    #      라는 지름길이 존재한다. 그것을 최적화하면 로봇을 **세우는** 쪽으로 민다
    #      (is_stop 프레임을 지운 이유와 같은 함정이 정상 데이터의 감속에도 남아 있다).
    c_shrink = float(cs(gsel, -a0[:, SIDX], dim=-1).mean())
    #  (b) 궤적을 앞당기는 방향?  리플레이가 **다음 결정**에서 내는 액션 쪽이 곧
    #      "같은 궤적을 더 진행한" 방향이다. '빨리 끝낸다' 가 맞으면 여기 정렬돼야 한다.
    k2 = np.minimum(k + R, flat.ep_end[k])
    a_next = torch.from_numpy(np.ascontiguousarray(
        np.asarray(norm[k2])[:, :LAT + R].reshape(len(k), -1))).to(dev)
    c_ahead = float(cs(gsel, (a_next - a0)[:, SIDX], dim=-1).mean())
    #  상승 후 액션 크기가 커지나 작아지나 (ss 는 1 nat 이동 기준값을 쓴다)
    a1 = (a0 + ss_nat * g).clamp(-1.0, 1.0)
    n0 = float(a0[:, SIDX].abs().mean())
    n1 = float(a1[:, SIDX].abs().mean())
    print(f"\n  (3) 그 방향의 정체 — 앙상블이 일치하면 방향은 실재한다. 무엇을 향하나?")
    print(f"  cos(g, -a)  '멈추는 쪽'   {c_shrink:+.4f}"
          f"   (>0.3 이면 로봇을 세우는 방향이다 = 지름길 학습)")
    print(f"  cos(g, a_next-a) '앞당기는 쪽' {c_ahead:+.4f}"
          f"   (>0.3 이면 궤적을 진행시키는 방향 = 진짜 '빨리 끝내기')")
    print(f"  |a| 평균  {n0:.5f} -> {n1:.5f}  ({100 * (n1 - n0) / max(n0, 1e-12):+.1f}%)"
          f"   (step_size={ss_nat:.3g} 로 1스텝)")
    if c_shrink > 0.3:
        print(f"  ★ 상승 방향이 '액션을 0 으로 줄이는 쪽'과 정렬돼 있다. 이 critic 을 따라가면")
        print(f"    로봇이 **덜 움직이게** 된다 — 데이터의 감속 구간이 만든 지름길이다.")
    elif c_ahead > 0.3:
        print(f"  → 궤적을 앞당기는 방향이다. '빨리 끝내기' 가설과 맞는다.")
    elif abs(np.mean(cos_ens)) > 0.5:
        print(f"  → 방향은 일관되지만(cos_ens {np.mean(cos_ens):+.2f}) 두 가설 어느 쪽도 아니다.")
        print(f"    성공만 있는 데이터라 '좋은 액션' 의 정의가 없으므로, 네트워크가 액션 축에서")
        print(f"    찾아낸 임의의 일관된 외삽 방향일 가능성이 높다. 실기에서 쓰기 전에")
        print(f"    반드시 롤아웃으로 검증할 것.")
    if abs(cl) < 0.05 and abs(np.mean(cos_ens)) < 0.3:
        print(f"  ★ 방향도 무작위다. 이 gradient 를 step_size 로 증폭하면 액션을")
        print(f"    **아무 방향으로나** 미는 것이 된다 (keep-best 가 막아 주긴 하지만,")
        print(f"    그것은 '개선이 없다' 를 뜻할 뿐이다).")
    elif cl > 0.05:
        print(f"  → 방향은 약하지만 성공 시연 쪽을 가리킨다. step_size 를 크게 주면")
        print(f"    (≈{ss_nat:.3g}) 의미 있는 개선이 나올 수 있다 — 아래 스윕에서 확인할 것.")
    return {"q_mean": qm0, "q_std": qsd, "time_to_go": tgo,
            "grad_abs_mean": float(gn.mean()), "ss_for_nat_move": ss_nat,
            "cos_ensemble": float(np.mean(cos_ens)), "cos_toward_logged": cl,
            "cos_shrink": c_shrink, "cos_ahead": c_ahead,
            "act_abs_before": n0, "act_abs_after": n1, "perturb": rows}


def probe(E, FR, OK, SI):
    """에피소드 하나를 재고 (수치, 프레임별 배열) 을 돌려준다."""
    # 결정 프레임 (rl/relabel_parl.py:163-181 과 같은 규칙)
    dec, span = [], []
    for t in range(FR[0], FR[-1] + 1, R):
        w = np.arange(t + LAT, min(t + LAT + R, FR[-1] + 1))
        if len(w):
            dec.append(t)
            span.append(w)
    dec = np.asarray(dec, np.int64)

    # 자연 이동량 기준선: 리플레이 궤적 자체가 1프레임에 얼마나 움직이는가.
    # 정규화 공간의 0.01 이라는 숫자만 봐서는 크고 작음을 판단할 수 없다.
    _a1 = np.asarray(norm[FR[:-1]])[:, 0][:, RAWCOL]
    _a2 = np.asarray(norm[FR[1:]])[:, 0][:, RAWCOL]
    nat = float(np.median(np.abs(_a2 - _a1).mean(1)))
    nat_raw = float(np.median(np.abs(flat.action[FR[1:]][:, RAWCOL]
                                     - flat.action[FR[:-1]][:, RAWCOL]).mean(1)))

    r = {"d_bc_m": [], "d_bc_s": [], "q_log": [], "q_bc_m": [], "q_bc_s": [],
         "raw_bc_m": [], "raw_bc_s": [],
         "d_opt": {s: [] for s in STEPS}, "q_opt": {s: [] for s in STEPS},
         "d_asc": {s: [] for s in STEPS}, "raw_opt": {s: [] for s in STEPS},
         "won_log": {s: 0 for s in STEPS}}
    t0 = time.time()
    for c in range(0, len(dec), a.batch):
        idx = dec[c:c + a.batch]
        b = len(idx)
        logged = torch.from_numpy(np.ascontiguousarray(
            np.asarray(norm[idx])[:, :LAT + R].reshape(b, -1))).to(dev)
        with torch.no_grad():
            ch = vla.sample(vla_obs(idx), a.num_samples)          # (B,M,H,A)
        acts = ch[:, :, :LAT + R].reshape(b, a.num_samples, FULL).float()
        if PRE:
            acts[:, :, :PRE] = logged[:, None, :PRE]   # prefix 는 이미 커밋됨 (expo.py:231)
        st = torch.from_numpy(snorm[idx]).to(dev)
        lat = C.latent_of(idx, st)
        rw_l = vla.denormalize_actions(np.asarray(norm[idx])[:, :LAT + R], flat.state[idx])

        # (1) BC 샘플 32개 전부 = Q 로 고르기 **전** 의 draw 들. 편향이 없다.
        #     하나하나 리플레이와 비교해 mean±std 를 낸다 — 이 밴드가 "정책이 스스로와
        #     불일치하는 폭" 이고, 최적화 결과를 여기에 대야 의미를 판단할 수 있다.
        dbc = l1(acts - logged[:, None, :])                       # (B, M)
        r["d_bc_m"].append(dbc.mean(1).cpu().numpy())
        r["d_bc_s"].append(dbc.std(1).cpu().numpy())
        with torch.no_grad():
            r["q_log"].append(q_sel(lat, logged).cpu().numpy())
            qbc = q_sel(lat.repeat_interleave(a.num_samples, 0),
                        acts.reshape(b * a.num_samples, FULL)).view(b, a.num_samples)
            r["q_bc_m"].append(qbc.mean(1).cpu().numpy())
            r["q_bc_s"].append(qbc.std(1).cpu().numpy())

        # 후보 집합 = Q 상위 K-1 + 로그된 액션 1개 (relabel_parl.py:210-228).
        # qbc 를 위에서 이미 냈으므로 다시 채점하지 않는다 — 같은 32개다.
        top = qbc.topk(min(a.num_keep - 1, a.num_samples), dim=1).indices
        keep = torch.gather(acts, 1, top[..., None].expand(-1, -1, FULL))
        cand = torch.cat([keep, logged[:, None, :]], 1)
        K = cand.shape[1]

        # (2) step_size 마다 상승 -> argmax. 후보는 재사용하므로 여기는 싸다.
        for ss in STEPS:
            opt, qq = ascend(cand, lat, ss)
            pick = qq.argmax(1)
            chosen = torch.gather(opt, 1, pick[:, None, None].expand(-1, 1, FULL))[:, 0]
            pre = torch.gather(cand, 1, pick[:, None, None].expand(-1, 1, FULL))[:, 0]
            r["d_opt"][ss].append(l1(chosen - logged).cpu().numpy())
            r["d_asc"][ss].append(l1(chosen - pre).cpu().numpy())  # 상승이 민 거리만
            r["won_log"][ss] += int((pick == K - 1).sum())
            with torch.no_grad():
                r["q_opt"][ss].append(q_sel(lat, chosen).cpu().numpy())
            rw_o = vla.denormalize_actions(chosen.view(b, LAT + R, A_DIM).cpu().numpy(),
                                           flat.state[idx])
            r["raw_opt"][ss].append(raw_l1(rw_o, rw_l))
        # BC 32개도 raw 로. use_relative_action=true 라 기준 state 가 있어야 하고,
        # 정규화 때와 **같은 그 프레임의 state** 여야 한다 (M 개로 반복해 준다).
        rw_b = vla.denormalize_actions(
            acts.reshape(b * a.num_samples, LAT + R, A_DIM).cpu().numpy(),
            np.repeat(flat.state[idx], a.num_samples, axis=0))
        dbr = raw_l1(rw_b, np.repeat(rw_l, a.num_samples, axis=0)).reshape(b, a.num_samples)
        r["raw_bc_m"].append(dbr.mean(1))
        r["raw_bc_s"].append(dbr.std(1))
    el = time.time() - t0

    o = {"ep": E, "n_frames": len(FR), "ok": OK, "session": sessions[SI].name,
         "n_dec": len(dec), "nat": nat, "nat_raw": nat_raw, "sec": el}
    for k in ("d_bc_m", "d_bc_s", "q_log", "q_bc_m", "q_bc_s", "raw_bc_m", "raw_bc_s"):
        o[k] = np.concatenate(r[k])
    for k in ("d_opt", "d_asc", "q_opt", "raw_opt"):
        o[k] = {s: np.concatenate(v) for s, v in r[k].items()}
    o["won_log"] = {s: r["won_log"][s] / max(len(dec), 1) for s in STEPS}
    o["span"], o["fr0"] = span, int(FR[0])
    return o


# ── 에피소드별 그림/비디오 ───────────────────────────────────────────────────
import imageio.v2 as imageio  # noqa: E402
from PIL import Image  # noqa: E402

cmap = plt.get_cmap("turbo")
col = {s: cmap(0.08 + 0.84 * i / max(len(STEPS) - 1, 1)) for i, s in enumerate(STEPS)}


def even(img):
    """libx264 + yuv420p 는 가로·세로가 **짝수**여야 한다.

    macro_block_size=1 로 imageio 의 자동 패딩을 껐기 때문에(리사이즈를 원치 않아서)
    홀수 해상도가 그대로 ffmpeg 에 간다. 잡 1074 가 1867x520 에서
    "width not divisible by 2" -> BrokenPipe 로 죽었다.
    잘라내면 곡선 오른쪽 끝이 사라지므로 흰색으로 1픽셀 패딩한다.
    """
    h, w = img.shape[:2]
    if h % 2 or w % 2:
        img = np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), constant_values=255)
    return img


def render_episode(o):
    """에피소드 하나의 2단 그림(+비디오). 결정 t 가 프레임 t+LAT … t+LAT+R-1 을 지배한다."""
    T_EP, E, OK = o["n_frames"], o["ep"], o["ok"]

    def pf(vals):
        out = np.full(T_EP, np.nan)
        for j, w in enumerate(o["span"]):
            out[w - o["fr0"]] = vals[j]
        return out

    x = np.arange(T_EP)
    ROW_H, PW, LEG = a.row_h, a.plot_w, a.legend_w
    CAM_H = ROW_H * 2
    TOT = PW + LEG                       # 범례를 **축 밖** 오른쪽 열에 둔다
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TOT / 100, CAM_H / 100), dpi=100,
                                   sharex=True)

    # step_size 별 곡선이 전부 같으면 하나로 접는다. critic 이 액션에 평평하면 6개
    # 선이 같은 자리에 겹쳐 그려지는데, 그것은 그림이 아니라 소음이다.
    same = all(np.allclose(o["d_opt"][s], o["d_opt"][STEPS[0]], atol=1e-9) for s in STEPS)
    shown = [STEPS[0]] if same else STEPS
    sslab = (f"ss={STEPS[0]:g}…{STEPS[-1]:g} 전부 동일\n(상승이 액션을 못 움직였다)"
             if same else None)

    m, sd = pf(o["d_bc_m"]), pf(o["d_bc_s"])
    # BC 32개의 mean±std 밴드 — 정책이 스스로와 불일치하는 폭. 판단의 기준선이다.
    ax1.fill_between(x, m - sd, m + sd, color="0.15", alpha=.18, lw=0,
                     label="BC 32샘플 ±1std")
    ax1.plot(x, m, color="0.15", lw=1.8, label="① |BC 32샘플 − 리플레이| 평균")
    for s in shown:
        ax1.plot(x, pf(o["d_opt"][s]), lw=1.4, color=col[s],
                 label=f"② |최적화 − 리플레이|\n    {sslab or f'ss={s:g}'}")
    ax1.axhline(o["nat"], color="crimson", lw=1.2, ls=":",
                label=f"③ 리플레이 1프레임 자연변화\n    {o['nat']:.4f} "
                      f"({o['nat_raw']:.4f} rad)")
    ax1.set_ylabel(f"L1 평균 mean|Δ|\n({spec.active_dim}관절 x {R}스텝, 정규화)", fontsize=8)
    # symlog: ss=0 이고 로그된 액션이 argmax 를 이기면 거리가 **정확히 0** 이다.
    # 순수 log 축이면 그 점이 조용히 사라져 "데이터가 없는 것" 처럼 보인다.
    ax1.set_yscale("symlog", linthresh=1e-5)
    ax1.grid(alpha=.25, which="both")
    ax1.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
               framealpha=0, borderaxespad=0)
    ax1.set_title(f"ep{E} ({T_EP}f, {'성공' if OK else '실패'}, {o['session']})  "
                  f"critic {a.tag}@{a.critic_step}  {groups} "
                  f"{spec.active_dim}관절x{R}스텝={NIDX}차원  "
                  f"num_steps={a.num_steps} M={a.num_samples}", fontsize=8)

    ax2.plot(x, pf(o["q_log"]), color="0.15", lw=1.8, label="Q(리플레이)")
    qm, qs = pf(o["q_bc_m"]), pf(o["q_bc_s"])
    ax2.fill_between(x, qm - qs, qm + qs, color="0.55", alpha=.20, lw=0,
                     label=f"Q(BC 32샘플) ±1std\n    (std {o['q_bc_s'].mean():.2e})")
    ax2.plot(x, qm, color="0.55", lw=1.2, ls="--", label="Q(BC 32샘플) 평균")
    for s in shown:
        ax2.plot(x, pf(o["q_opt"][s]), lw=1.2, color=col[s],
                 label=f"Q(선택) {sslab.splitlines()[0] if sslab else f'ss={s:g}'}")
    ax2.plot(x, G ** (T_EP - x) if OK else np.zeros(T_EP), color="0.5", lw=1.2, ls="-.",
             label="참값 γ^(T-t)")
    # 이 그림의 결론을 숫자로 박아 둔다: 액션 민감도 = (액션 간 Q 산포)/(시간축 Q 산포).
    _rel = o["q_bc_s"].mean() / max(o["q_log"].std(), 1e-12)
    ax2.text(0.012, 0.06,
             f"액션 민감도 = 액션 간 Q std {o['q_bc_s'].mean():.2e}"
             f" / 시간축 Q std {o['q_log'].std():.3f} = {_rel:.3%}",
             transform=ax2.transAxes, fontsize=7.5, va="bottom",
             bbox=dict(fc="white", ec="0.7", alpha=.9))
    ax2.set_ylabel("Q", fontsize=8)
    ax2.set_xlabel("에피소드 내 프레임", fontsize=8)
    ax2.grid(alpha=.25)
    ax2.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
               framealpha=0, borderaxespad=0)
    for ax in (ax1, ax2):
        ax.set_xlim(0, T_EP - 1)
        ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.11 * PW / TOT, right=PW / TOT,
                        top=0.94, bottom=0.08, hspace=0.07)
    fig.canvas.draw()
    BASE = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    bb = ax2.get_position()
    X0, X1 = int(bb.x0 * TOT), int(bb.x1 * TOT)
    fig.savefig(OUT / f"ep{E}.png", dpi=100)
    plt.close(fig)

    if a.no_video:
        return
    si = [x_[3] for x_ in PICK if x_[0] == E][0]
    local = E - flat.ep_offset[si]
    vp = (sessions[si] / f"videos/chunk-{local // 1000:03d}" / cam_key
          / f"episode_{local:06d}.mp4")
    if not vp.is_file():
        print(f"  ⚠ 비디오가 없어 mp4 를 건너뛴다: {vp}")
        return
    frames = imageio.mimread(str(vp), memtest=False)
    h, w = frames[0].shape[:2]
    CAM_W = int(round(CAM_H * w / h))          # 크롭하지 않고 종횡비를 지킨다
    frames = [np.asarray(Image.fromarray(f[..., :3]).resize((CAM_W, CAM_H), Image.BILINEAR))
              for f in frames]
    if len(frames) != T_EP:
        print(f"  ⚠ ep{E} 비디오 {len(frames)}프레임 != parquet {T_EP}프레임 — 커서가 어긋난다")
    wr = imageio.get_writer(str(OUT / f"ep{E}.mp4"), fps=a.fps, codec="libx264",
                            pixelformat="yuv420p", macro_block_size=1,
                            output_params=["-crf", "20"])
    for t in range(T_EP):
        panel = BASE.copy()
        cx = int(X0 + (X1 - X0) * t / max(T_EP - 1, 1))
        panel[:, max(cx - 1, 0):cx + 2] = np.array([220, 40, 40], np.uint8)
        wr.append_data(even(np.hstack([frames[min(t, len(frames) - 1)], panel])
                            ).astype(np.uint8))
    wr.close()


# 먼저 민감도를 잰다 — 여기서 '무시할 수준' 이 나오면 아래 스윕은 전부 같은 값이 나온다.
# NAT_HINT: sanity 는 본 루프보다 먼저 돌므로 자연변화 기준선을 미리 재 둔다
# (선택된 에피소드들의 중앙값). "액션을 1프레임치 움직이려면 step_size 얼마?" 에 쓴다.
NAT_HINT = float(np.median([
    np.median(np.abs(np.asarray(norm[fr[1:]])[:, 0][:, RAWCOL]
                     - np.asarray(norm[fr[:-1]])[:, 0][:, RAWCOL]).mean(1))
    for _, fr, _, _ in PICK]))
_alldec = np.concatenate([np.arange(fr[0], fr[-1] + 1, R) for _, fr, _, _ in PICK])
SANITY = sanity(_alldec)

ALL = []
for n_, (E, FR, OK, SI) in enumerate(PICK, 1):
    print(f"\n[{n_}/{len(PICK)}] ep{E}  {len(FR)}f  {'성공' if OK else '실패'}  "
          f"{sessions[SI].name}", flush=True)
    o = probe(E, FR, OK, SI)
    print(f"    결정 {o['n_dec']}개  {o['sec']:.0f}s   "
          f"BC {o['d_bc_m'].mean():.5f}±{o['d_bc_s'].mean():.5f}  "
          f"nat {o['nat']:.5f}", flush=True)
    render_episode(o)
    ALL.append(o)
    print(f"    -> {OUT / f'ep{E}.png'}" + ("" if a.no_video else f" / ep{E}.mp4"), flush=True)


# ── 종합 (에피소드 전체를 풀링) ──────────────────────────────────────────────
def pool(k, s=None):
    return np.concatenate([o[k][s] if s is not None else o[k] for o in ALL])


D_BC, D_BCS = pool("d_bc_m"), pool("d_bc_s")
Q_LOG, Q_BC, Q_BCS = pool("q_log"), pool("q_bc_m"), pool("q_bc_s")
RAW_BC, RAW_BCS = pool("raw_bc_m"), pool("raw_bc_s")
D_OPT = {s: pool("d_opt", s) for s in STEPS}
D_ASC = {s: pool("d_asc", s) for s in STEPS}
Q_OPT = {s: pool("q_opt", s) for s in STEPS}
RAW_OPT = {s: pool("raw_opt", s) for s in STEPS}
NAT = float(np.median([o["nat"] for o in ALL]))
NAT_RAW = float(np.median([o["nat_raw"] for o in ALL]))
NDEC = sum(o["n_dec"] for o in ALL)
WON = {s: float(np.mean([o["won_log"][s] for o in ALL])) for s in STEPS}


def pm(m, s_):
    return f"{m:.5f}±{s_:.5f}"


print(f"\n{'=' * 108}")
print(f"[종합] 에피소드 {len(ALL)}개 / 결정 {NDEC}개   critic {a.tag}@{a.critic_step} "
      f"(num_qs={CINFO['num_qs']}, γ={CINFO['discount']})   num_steps={a.num_steps}  "
      f"M={a.num_samples} K={a.num_keep}")
print(f"       지표 = L1 평균 mean|Δ| over ({spec.active_dim}관절 x {R}스텝) vs 리플레이")
print(f"       기준선 = 리플레이 1프레임 자연변화 {NAT:.5f} (raw {NAT_RAW:.5f} rad)")
# ±산포를 두 축으로 나눠 찍는다. 하나로 합치면 비교가 성립하지 않는다:
#   ±샘플간  같은 결정에서 액션 32개가 서로 다른 폭 (BC 만 가진다)
#   ±결정간  결정 프레임마다 값이 흔들리는 폭 (두 행 모두 가진다 — 이게 비교 대상)
print(f"\n  {'':<14}{'|Δ-리플레이|':>12}{'±샘플간':>10}{'±결정간':>10}{'nat배':>7}"
      f"{'raw[rad]':>10}{'상승이동':>10}{'Q(선택)':>10}{'±액션간':>10}{'ΔQ':>9}{'로그승':>7}")
print(f"  {'BC 32샘플':<14}{D_BC.mean():>12.5f}{D_BCS.mean():>10.5f}{D_BC.std():>10.5f}"
      f"{D_BC.mean() / NAT:>7.1f}{RAW_BC.mean():>10.5f}{'-':>10}"
      f"{Q_BC.mean():>10.5f}{Q_BCS.mean():>10.2e}{(Q_BC - Q_LOG).mean():>+9.4f}{'-':>7}")
for s in STEPS:
    lab = f"ss={s:g}" + (" 선택만" if s == 0 else "")
    print(f"  {lab:<14}{D_OPT[s].mean():>12.5f}{'-':>10}{D_OPT[s].std():>10.5f}"
          f"{D_OPT[s].mean() / NAT:>7.1f}{RAW_OPT[s].mean():>10.5f}"
          f"{D_ASC[s].mean():>10.5f}{Q_OPT[s].mean():>10.5f}{'-':>10}"
          f"{(Q_OPT[s] - Q_LOG).mean():>+9.4f}{WON[s]:>7.0%}")
print(f"  {'(참고)':<14}Q(리플레이) 평균 {Q_LOG.mean():.5f}  시간축 std {Q_LOG.std():.5f}"
      f"   -> 액션 민감도 = {Q_BCS.mean() / max(Q_LOG.std(), 1e-12):.3%}")
# ΔQ 를 프레임으로 바꿔야 "이 최적화가 쓸 만한가" 를 판단할 수 있다.
_tgo = float(np.log(max(Q_LOG.mean(), 1e-12)) / np.log(G))
print(f"  {'':<14}critic 이 믿는 남은 프레임 {_tgo:.0f}f. ΔQ 를 프레임으로 환산하면:")
for s in STEPS:
    _f = to_frames(float((Q_OPT[s] - Q_LOG).mean()), float(Q_LOG.mean()))
    print(f"    ss={s:<10g} ΔQ {float((Q_OPT[s] - Q_LOG).mean()):+.5f}"
          f"  = {_f:+.2f} 프레임 빨리 끝난다고 믿음 ({_f / max(_tgo, 1e-9):+.2%})")

print("\n  [에피소드별] |최적화-리플레이| (정규화 L1 평균)")
hdr = "".join(f"{('ss=' + f'{s:g}'):>12}" for s in STEPS)
print(f"  {'ep':>5}{'프레임':>7}{'':>4}{'BC':>11}{'nat':>9}{hdr}   세션")
for o in ALL:
    row = "".join(f"{o['d_opt'][s].mean():>12.5f}" for s in STEPS)
    print(f"  {o['ep']:>5}{o['n_frames']:>7}{'성공' if o['ok'] else '실패':>4}"
          f"{o['d_bc_m'].mean():>11.5f}{o['nat']:>9.5f}{row}   {o['session']}")

print("\n  읽는 법:")
print("   · L1 평균이라 raw 열은 곧 **관절 하나가 한 스텝에 평균 몇 rad 다른가** 다.")
print("   · 최적화 행이 'BC 32샘플' 의 mean±std 밴드 안에 있으면 → 최적화는 그냥 다시")
print("     뽑은 것과 구별되지 않는다. step_size 를 올릴 문제가 아니라 critic 문제다.")
print("   · '상승이동' 이 0 에 가까우면 ∇_A Q 가 사실상 0 이다 (critic 이 액션에 평평하다).")
print("   · ss=0 행은 상승을 끈 것이라 **선택만의 효과**다. 나머지 행과의 차이가 상승 효과다.")
print("   · '로그승' = 후보 중 **로그된 액션에서 나온 것**이 argmax 로 뽑힌 비율")
print("     (PA-RL 의 안전장치로 후보 집합에 로그 액션을 하나 넣는다).")
print("     ss=0 일 때만 그것이 |Δ|=0 을 뜻한다 — ss>0 이면 그 후보도 상승하므로 움직인다.")
print(f"     1/K = {1 / a.num_keep:.0%} 에 가까우면 선택이 균등추첨과 구별되지 않는다는 뜻이다.")
print(f"   · nat배 = 1프레임 자연변화({NAT:.5f}) 대비 몇 배. 1 미만이면 로봇이 느끼지 못한다.")
print("   · 에피소드별 표에서 teleop 세션과 롤아웃 세션의 BC 열이 크게 다르면, base policy")
print("     가 teleop A 만으로 학습된 것이 그대로 드러난 것이다 (리플레이의 의미가 다르다).")

# ── 종합 그림: x = step_size ─────────────────────────────────────────────────
# 이것이 step_size 를 고르는 그림이다. 세로축 두 개를 같은 x 로 본다:
#   위  얼마나 움직였나 (BC 밴드/자연변화와 비교)
#   아래 그 대가로 Q 가 얼마나 올랐나
# 위가 밴드를 넘지 않는데 아래가 오르면 그것은 critic 의 자기기만이다 (같은 이웃에서
# 숫자만 오른 것). 반대로 위가 자연변화를 넘어서면 실기에서 위험하다.
XS = np.array([max(s, 1e-6) for s in STEPS])          # 0 은 log 축에 못 올린다
fig, (b1, b2) = plt.subplots(2, 1, figsize=(12.5, 7), dpi=110, sharex=True)
b1.axhspan(D_BC.mean() - D_BCS.mean(), D_BC.mean() + D_BCS.mean(),
           color="0.15", alpha=.15, lw=0, label="BC 32샘플 ±1std")
b1.axhline(D_BC.mean(), color="0.15", lw=1.8, label=f"BC 평균 {D_BC.mean():.5f}")
b1.axhline(NAT, color="crimson", lw=1.2, ls=":",
           label=f"1프레임 자연변화 {NAT:.5f} ({NAT_RAW:.5f} rad)")
for o in ALL:
    b1.plot(XS, [o["d_opt"][s].mean() for s in STEPS], lw=0.9, alpha=.5,
            color="steelblue")
b1.plot(XS, [D_OPT[s].mean() for s in STEPS], lw=2.4, color="darkorange",
        marker="o", ms=5, label="|최적화 - 리플레이| (전체 평균)")
b1.plot(XS, [D_ASC[s].mean() for s in STEPS], lw=1.6, color="seagreen",
        marker="s", ms=4, ls="--", label="그중 상승이 민 거리")
b1.set_ylabel(f"L1 평균 mean|Δ|\n({spec.active_dim}관절 x {R}스텝, 정규화)", fontsize=9)
b1.set_yscale("symlog", linthresh=1e-6)
b1.grid(alpha=.25, which="both")
b1.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5),
          framealpha=0, borderaxespad=0)
b1.set_title(f"step_size 스윕 — 에피소드 {len(ALL)}개 / 결정 {NDEC}개  "
             f"critic {a.tag}@{a.critic_step} (num_qs={CINFO['num_qs']})  "
             f"num_steps={a.num_steps}  얇은 선 = 에피소드별", fontsize=10)

b2.axhline(0, color="0.6", lw=1.0)
b2.axhline((Q_BC - Q_LOG).mean(), color="0.15", lw=1.6, ls="--",
           label=f"BC 32샘플 평균 ΔQ {(Q_BC - Q_LOG).mean():+.4f}")
for o in ALL:
    b2.plot(XS, [(o["q_opt"][s] - o["q_log"]).mean() for s in STEPS], lw=0.9, alpha=.5,
            color="steelblue")
b2.plot(XS, [(Q_OPT[s] - Q_LOG).mean() for s in STEPS], lw=2.4, color="darkorange",
        marker="o", ms=5, label="ΔQ = Q(선택) - Q(리플레이)")
b2r = b2.twinx()
b2r.plot(XS, [WON[s] * 100 for s in STEPS], lw=1.4, color="purple", marker="^", ms=4,
         label="로그된 액션이 이긴 비율")
b2r.set_ylabel("로그승 [%]", fontsize=9, color="purple")
b2r.tick_params(labelsize=8, colors="purple")
b2r.set_ylim(0, 105)
b2.set_ylabel("ΔQ", fontsize=9)
b2.set_xlabel("step_size  (맨 왼쪽 점은 0 = 상승 없음, 선택만)", fontsize=9)
b2.set_xscale("log")
b2.grid(alpha=.25, which="both")
h1, l1_ = b2.get_legend_handles_labels()
h2, l2_ = b2r.get_legend_handles_labels()
b2.legend(h1 + h2, l1_ + l2_, fontsize=8, loc="center left",
          bbox_to_anchor=(1.09, 0.5), framealpha=0, borderaxespad=0)
b2.set_xticks(XS)
b2.set_xticklabels([f"{s:g}" for s in STEPS], fontsize=8)
for ax in (b1, b2):
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "summary.png", dpi=110)
plt.close(fig)

# ── 수치 덤프 (critic 을 바꿔 다시 재고 비교할 때 쓴다) ──────────────────────
js = {"exp": a.exp, "dataset": exp["dataset"], "base_policy": exp["base_policy"],
      "critic": str(ck.relative_to(REPO)), "critic_info": CINFO,
      "groups": groups, "n_edit_dims": NIDX, "active_joints": spec.active_dim,
      "replan": R, "latency": LAT, "num_steps": a.num_steps,
      "num_samples": a.num_samples, "num_keep": a.num_keep,
      "metric": f"L1 mean |delta| over {spec.active_dim} joints x {R} steps",
      "n_episodes": len(ALL), "n_decisions": NDEC, "sanity": SANITY,
      "nat": NAT, "nat_raw": NAT_RAW,
      "bc": {"d_mean": float(D_BC.mean()), "d_std_within": float(D_BCS.mean()),
             "raw_mean": float(RAW_BC.mean()), "raw_std_within": float(RAW_BCS.mean()),
             "q_mean": float(Q_BC.mean()), "dq_mean": float((Q_BC - Q_LOG).mean())},
      "q_logged_mean": float(Q_LOG.mean()),
      "steps": {f"{s:g}": {"d_mean": float(D_OPT[s].mean()),
                           "d_std_across_dec": float(D_OPT[s].std()),
                           "raw_mean": float(RAW_OPT[s].mean()),
                           "ascent_move": float(D_ASC[s].mean()),
                           "dq_mean": float((Q_OPT[s] - Q_LOG).mean()),
                           "won_logged": WON[s]} for s in STEPS},
      "episodes": [{"ep": o["ep"], "n_frames": o["n_frames"], "ok": o["ok"],
                    "session": o["session"], "n_dec": o["n_dec"],
                    "nat": o["nat"], "nat_raw": o["nat_raw"],
                    "d_bc": float(o["d_bc_m"].mean()),
                    "d_opt": {f"{s:g}": float(o["d_opt"][s].mean()) for s in STEPS}}
                   for o in ALL]}
(OUT / "summary.json").write_text(json.dumps(js, indent=2, ensure_ascii=False) + "\n")

print(f"\n[출력] {OUT}")
print(f"  summary.png    ★ x=step_size 종합판 — 여기서 step_size 를 고른다")
print(f"  summary.json   위 표의 수치 전부")
print(f"  ep*.png{'' if a.no_video else ' / ep*.mp4'}   에피소드별 "
      f"({len(ALL)}개)")
