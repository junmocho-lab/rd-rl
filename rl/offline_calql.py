#!/usr/bin/env python3
"""critic 오프라인 학습 — Cal-QL 판 (PA-RL 의 parl_calql 에 대응).

IQL 판과 다른 곳:
    IQL    Q ← r + γ^R·mask·V(s'),   V ← expectile_τ(Q_target(s,a_data) − V)
    Cal-QL Q ← r + γ^R·mask· min_M Q_target(s', a'),   V 네트워크가 없다
           + **보수화 항**:  cql_alpha · [ logsumexp_a Q(s,a) − Q(s,a_data) ]
           + **Cal-QL 하한**: logsumexp 안의 OOD Q 를 mc_return 아래로 누르지 않는다

CQL 항이 데이터 밖 액션의 Q 를 명시적으로 낮추므로, 같은 상태의 액션 다양성이 4% 뿐인
우리 데이터에서도 **액션 방향의 기울기를 손실함수가 만들어낸다** (SARSA/IQL 은 못 하는 것).
Cal-QL 의 하한은 그 억압이 참조(행동) 정책의 실제 리턴보다 낮게 내려가지 않게 해서
offline→online 전환 시의 성능 급락(unlearning)을 막는다.

우리 보상은 성공 에피소드의 마지막 1프레임만 1.0 이므로 mc_return 이 정확히 닫힌 형태다:
    mc[t] = γ^(에피소드끝 − t)  (성공 에피소드),   0  (실패 에피소드)

PA-RL 과 같은 것 / 다른 것은 파일 끝 주석 참고.

캐시(images.mm / actnorm.npy)는 offline_critic_0.py / offline_iql.py 와 공유하고, 산출물만
태그로 분리한다 (예: calql-dist128-a001-g0999-q10-s0).

usage:
  PYTHONPATH="$L_PYTHONPATH" $L_PY -u -m rl.offline_calql \\
      --exp fuji --data <데이터셋> --checkpoints <ckpt> \\
      --discount 0.999 --bins 128 --cql-alpha 0.01 --steps 40000 --holdout 0.2
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

from rl.data import build_flat, build_images, find_sessions, nstep, open_images, resolve_modality
from rl.expo import ExpoConfig
import torch.nn as nn

from rl.nets import BatchEncoder, CriticEnsemble, xavier_
from rl.offline_critic import normalize_all
from rl.vla_rldx import RLDXVLA, load_state_action_processor, normalize_states

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="fuji")
p.add_argument("--data", type=Path, required=True)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval-every", type=int, default=1000)
p.add_argument("--holdout", default="0.2", help="세션 이름 문자열 또는 에피소드 비율(0<x<1)")
p.add_argument("--cql-alpha", type=float, default=0.01,
               help="보수화 항의 가중치. PA-RL 실기 설정 0.01 / antmaze 0.005")
p.add_argument("--cql-n-actions", type=int, default=4,
               help="OOD 액션 수 (종류별). PA-RL 실기 4 / 기본 10")
p.add_argument("--cql-temp", type=float, default=1.0, help="logsumexp 온도")
p.add_argument("--no-calql", action="store_true",
               help="mc_return 하한을 끈다 = 순수 CQL (Cal-QL 의 효과를 분리해 보려면)")
p.add_argument("--ood", default="random,perturb",
               help="OOD 액션 종류. random=균등[-1,1], perturb=데이터 액션+노이즈. "
                    "PA-RL 은 여기에 정책 샘플도 넣지만 VLA 호출이 필요해 뺐다 (아래 주석)")
p.add_argument("--perturb-scale", type=float, default=0.2, help="perturb 노이즈 크기")
p.add_argument("--bins", type=int, default=0,
               help="0 이면 스칼라 MSE. >0 이면 distributional critic (HL-Gauss, PA-RL 기본 128)")
p.add_argument("--q-range", default="0,1",
               help="distributional 일 때 support. 우리 보상은 성공 종료 1프레임뿐이라 리턴 ∈ [0,1]")
p.add_argument("--discount", type=float, default=0.0,
               help="프레임당 할인. 0 이면 exp yaml 값. 결정당 실효는 discount^replan 이다 — "
                    "heuristic 1-1/L 은 L 을 **프레임** 길이로 넣으면 된다 (fuji 1200 → 0.99917)")
p.add_argument("--num-qs", type=int, default=0,
               help="Q 앙상블 크기. 0 이면 exp yaml 값(10). PA-RL 은 distributional 이면 10, "
                    "스칼라면 2 를 쓴다. yaml 과 다르게 주면 ExpoServer 가 못 읽는다")
p.add_argument("--tau", type=float, default=0.0,
               help="target polyak. 0 이면 exp yaml 값(0.005). 시간상수 1/tau 스텝")
p.add_argument("--images", default="gpu", choices=("gpu", "mmap"),
               help="gpu: images.mm 을 통째로 VRAM 에 올린다 (fuji 30.7GB). NFS 랜덤읽기가 "
                    "스텝 시간의 96%% 라 이게 가장 큰 개선. 여유가 없으면 mmap 으로 떨어진다")
p.add_argument("--tag", default="",
               help="산출물 이름. 비우면 설정에서 자동 생성 (판/τ/seed 가 달라도 안 겹치게)")
a = p.parse_args()

torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
gen = torch.Generator().manual_seed(a.seed)
assert torch.cuda.is_available(), "GPU 전용이다"
dev = "cuda"

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
if a.discount:
    cfg.discount = a.discount                        # 지평 스윕용 (yaml 을 고치지 않는다)
if a.num_qs:
    cfg.num_qs = a.num_qs
if a.tau:
    cfg.tau = a.tau
R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
GEFF = cfg.discount ** R
base = a.checkpoints / exp["base_policy"]
work = a.checkpoints / f"{a.exp}-critic"             # images.mm / actnorm.npy 는 공유
# 산출물만 태그로 분리한다 — 스칼라 판과 distributional 판을 같이 비교하려면 필수
TAG = a.tag or (f"{'cql' if a.no_calql else 'calql'}"
                f"-{'dist' + str(a.bins) if a.bins else 'scalar'}"
                f"-a{f'{a.cql_alpha:g}'.replace('.', '')}"
                f"-g{f'{cfg.discount:g}'.replace('.', '')}"
                f"-q{cfg.num_qs}-s{a.seed}")
ckpt_path = work / f"critic_{TAG}.pt"
ev = work / f"eval_{TAG}"
ev.mkdir(parents=True, exist_ok=True)

# --- 1. 데이터 (offline_critic_0.py 와 동일) ---------------------------------
mod, src = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
build_images(sessions, flat, work / "images.mm", mod)
imgs, meta = open_images(work / "images.mm")
# 이미지를 VRAM 에 상주시킨다. 실측: NFS 랜덤 읽기가 배치당 0.7~1.9s (96 MB/s) 인데 스텝의
# GPU 계산은 81ms 뿐이라 가동률이 4% 였다. 순차 업로드는 364~469 MB/s 로 30.7GB 가 1.4분.
GI = None
if a.images == "gpu":
    need = int(np.prod(meta["shape"]))
    freeb = torch.cuda.mem_get_info()[0]
    if freeb < need * 1.15:
        print(f"[이미지] VRAM 부족 (여유 {freeb/1e9:.1f}GB < 필요 {need*1.15/1e9:.1f}GB) → mmap")
    else:
        t0 = time.time()
        GI = torch.empty(tuple(meta["shape"]), dtype=torch.uint8, device=dev)
        for c in range(0, meta["shape"][0], 2048):    # 순차로 읽어 올린다
            GI[c:c + 2048] = torch.from_numpy(np.array(imgs[c:c + 2048])).to(dev)
        print(f"[이미지] VRAM 상주 {need/1e9:.1f}GB  {time.time()-t0:.0f}s  "
              f"(남은 여유 {torch.cuda.mem_get_info()[0]/1e9:.1f}GB)")
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
if not (work / "actnorm.npy").is_file():
    vla = RLDXVLA(base, mod, RLDX, exp["rldx_data_config"], device=dev)
    normalize_all(vla, flat, H, cache=work / "actnorm.npy")
    del vla
    torch.cuda.empty_cache()
norm = normalize_all(None, flat, H, cache=work / "actnorm.npy")

frac = float(a.holdout) if a.holdout.replace(".", "", 1).isdigit() else 0.0
if 0 < frac < 1:
    every = max(2, int(round(1 / frac)))
    hold = np.isin(flat.episode, np.unique(flat.episode)[::every])
    how = f"에피소드 {every}개마다 1개"
else:
    sel = [i for i, n in enumerate(flat.sessions) if a.holdout and a.holdout in n]
    hold = np.isin(flat.session, sel)
    how = f"세션 이름에 '{a.holdout}' 포함 = {[flat.sessions[i] for i in sel]}"
train = np.flatnonzero(~hold[:len(flat) - R])
eps = [(e, np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode[hold])]
eps = [(e, fr, bool(flat.is_success[fr[-1]])) for e, fr in eps]
n_ok = sum(o for _, _, o in eps)
print(f"[exp] {a.exp} replan={R} latency={LAT} horizon={H}")
print(f"[할인] 프레임당 {cfg.discount} → 결정당 {GEFF:.5f}  지평 {1/(1-GEFF):.0f} 결정 "
      f"= {1/(1-GEFF)*R:.0f} 프레임  (min 편향 증폭 {1/(1-GEFF):.0f}배)")
print(f"[critic] 앙상블 {cfg.num_qs}, tau {cfg.tau}, "
      f"cql_alpha {a.cql_alpha}, OOD {a.ood} x{a.cql_n_actions}, "
      f"{'CQL (하한 없음)' if a.no_calql else 'Cal-QL (mc_return 하한)'}")
print(f"[산출물] {ckpt_path}\n          {ev}/")
print(f"[데이터] 세션 {len(sessions)} / 프레임 {len(flat)} / 학습 {len(train)} / "
      f"state {flat.state.shape[1]}→{snorm.shape[1]}차원")
print(f"[평가셋] {how} → 에피소드 {len(eps)} (성공 {n_ok} / 실패 {len(eps) - n_ok})")
if not n_ok or n_ok == len(eps):
    raise SystemExit("평가셋에 성공/실패가 한쪽뿐이라 AUC 를 못 낸다 — --holdout 을 바꿀 것")

# 액션/상태도 GPU 상주 (각각 83MB / 7.5MB). 남는 CPU 작업은 nstep 뿐이다.
NORM = torch.from_numpy(np.ascontiguousarray(np.asarray(norm[:, :LAT + R]))).to(dev)
SNORM = torch.from_numpy(snorm).to(dev)

def obs(i):
    """(B, H, W, 3*n_cams) uint8. **카메라 concat 을 GPU 에서** 한다 — CPU concat 은 340ms 였다."""
    if GI is not None:
        x = GI[torch.as_tensor(i, device=dev)]
    else:
        x = torch.from_numpy(np.ascontiguousarray(np.asarray(imgs[i]))).to(dev)
    return torch.cat([x[:, c] for c in range(x.shape[1])], -1)

act = lambda i: NORM[torch.as_tensor(i, device=dev)].reshape(len(i), -1)
st = lambda i: SNORM[torch.as_tensor(i, device=dev)]

# --- 2. 학습 (Cal-QL) --------------------------------------------------------
enc = BatchEncoder(3 * mod.n_cams, cfg.latent_dim_image, cfg.encoder_stage_sizes,
                   cfg.encoder_num_filters).to(dev)
critic = CriticEnsemble(cfg.latent_dim_image, snorm.shape[1], (LAT + R) * mod.action_dim,
                        cfg.num_qs, cfg.latent_dim_state, cfg.include_state, cfg.hidden_dims,
                        cfg.critic_layer_norm).to(dev)

# --- distributional critic (HL-Gauss) 옵션 ---------------------------------
if a.bins:
    lo_q, hi_q = (float(x) for x in a.q_range.split(","))
    edges = torch.linspace(lo_q, hi_q, a.bins + 1, device=dev)
    centers = ((edges[:-1] + edges[1:]) / 2)
    sigma = 0.75 * (hi_q - lo_q) / a.bins
    for m in critic.qs:
        m.head = xavier_(nn.Linear(m.body.out_dim, a.bins)).to(dev)

    def q_of(x):
        return (x.softmax(-1) * centers).sum(-1)

    def q_loss(x, y, w):
        z = (edges - y.clamp(lo_q, hi_q)[:, None]) / (sigma * 2 ** 0.5)
        cdf = 0.5 * (1 + torch.erf(z))
        pr = (cdf[:, 1:] - cdf[:, :-1])
        pr = pr / pr.sum(-1, keepdim=True).clamp_min(1e-8)
        return (-(pr * x.log_softmax(-1)).sum(-1) * w).mean()
    print(f"[critic] distributional: support [{lo_q},{hi_q}] bins {a.bins} sigma {sigma:.4f}")
else:
    q_of = lambda x: x
    q_loss = lambda x, y, w: (((x - y) ** 2) * w).mean()
    print("[critic] 스칼라 MSE")

with torch.no_grad():                                # PA-RL kernel_scale_final=1e-2 대응
    for m in critic.qs:
        m.head.weight.mul_(1e-2)
        m.head.bias.zero_()

target = copy.deepcopy(critic).requires_grad_(False)
tenc = copy.deepcopy(enc).requires_grad_(False)
opt = torch.optim.Adam(list(critic.parameters()) + list(enc.parameters()), lr=cfg.critic_lr)

# --- mc_return: Cal-QL 의 하한 -----------------------------------------------
# 보상이 성공 에피소드의 마지막 1프레임만 1.0 이므로 닫힌 형태다 (PA-RL calc_return_to_go 대응).
ep_end = np.zeros(len(flat), np.int64)
for e in np.unique(flat.episode):
    fr = np.flatnonzero(flat.episode == e)
    ep_end[fr] = fr[-1]
MC = np.where(flat.is_success, cfg.discount ** (ep_end - np.arange(len(flat))), 0.0)
MC = torch.from_numpy(MC.astype(np.float32)).to(dev)
print(f"[mc_return] 성공 프레임 {int(flat.is_success.sum())} / 평균 {float(MC.mean()):.4f} / "
      f"최대 {float(MC.max()):.4f}  (γ^남은프레임)")

K, A_DIM, FULL = a.cql_n_actions, mod.action_dim, (LAT + R) * mod.action_dim
kinds = [k.strip() for k in a.ood.split(",") if k.strip()]
print(f"[학습] {a.steps} 스텝, batch {cfg.batch_size}, OOD 액션 {len(kinds)*K}개/표본")

for step in range(1, a.steps + 1):
    i = train[rng.integers(0, len(train), cfg.batch_size)]
    n = nstep(flat, i, R, cfg.discount)
    j = n["next_idx"]
    B = len(i)
    a_data = act(i)
    lat = enc(obs(i), stop_gradient=cfg.freeze_critic_encoder)
    with torch.no_grad():
        # TD 타깃: 다음 상태의 **로그된** 액션 (PA-RL 은 정책 후보 max — 아래 주석 참고)
        nq = q_of(target(tenc(obs(j), stop_gradient=True), st(j), act(j),
                         members=critic.subsample(cfg.num_min_qs, gen))).min(dim=0).values
        tq = (torch.from_numpy(n["reward"]).to(dev)
              + (cfg.discount ** R) * torch.from_numpy(n["mask"]).to(dev) * nq)
    valid = torch.from_numpy(n["valid"]).to(dev)
    ql = critic(lat, st(i), a_data)
    q = q_of(ql)                                      # (num_qs, B)
    loss_td = q_loss(ql, tq, valid)

    # --- CQL 보수화 항 -------------------------------------------------------
    ood = []
    if "random" in kinds:                             # PA-RL cql_action_sample_method="uniform"
        ood.append(torch.rand(B, K, FULL, device=dev) * 2 - 1)
    if "perturb" in kinds:                            # 정책 샘플 대신 (VLA 호출 회피)
        ood.append((a_data[:, None] + a.perturb_scale
                    * torch.randn(B, K, FULL, device=dev)).clamp(-1.2, 1.2))
    cand = torch.cat(ood, dim=1)                      # (B, n_ood, FULL)
    n_ood = cand.shape[1]
    lat_r = lat.detach().repeat_interleave(n_ood, 0)  # 인코더는 한 번만 (액션만 바뀐다)
    st_r = st(i).repeat_interleave(n_ood, 0)
    q_ood = q_of(critic(lat_r, st_r, cand.reshape(B * n_ood, FULL)))
    q_ood = q_ood.view(cfg.num_qs, B, n_ood)
    if not a.no_calql:                                # Cal-QL: mc_return 아래로 누르지 않는다
        bound_rate = float((q_ood < MC[torch.as_tensor(i, device=dev)][None, :, None]).float().mean())
        q_ood = torch.maximum(q_ood, MC[torch.as_tensor(i, device=dev)][None, :, None])
    else:
        bound_rate = float("nan")
    z = torch.cat([q_ood, q.detach().unsqueeze(-1)], dim=-1)          # + Q(s,a_data)
    z = z - np.log(z.shape[-1]) * a.cql_temp
    ood_val = torch.logsumexp(z / a.cql_temp, dim=-1) * a.cql_temp    # (num_qs, B)
    loss_cql = (ood_val - q).mean()

    loss = loss_td + a.cql_alpha * loss_cql
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        for tp, pp in zip(list(target.parameters()) + list(tenc.parameters()),
                          list(critic.parameters()) + list(enc.parameters())):
            tp.mul_(1 - cfg.tau).add_(pp, alpha=cfg.tau)
    if step % 100 == 0 or step == a.steps:
        print(f"  step {step:5d}  td {float(loss_td):.5f}  cql_diff {float(loss_cql):+.4f}  "
              f"q {float(q.mean()):+.4f}  q_ood {float(q_ood.mean()):+.4f}  "
              f"하한적용 {bound_rate:.2f}")

    # --- 평가 + 저장 --------------------------------------------------------
    if step % a.eval_every == 0 or step == a.steps:
        with torch.no_grad():
            def q_at(idx, bs=256):
                return np.concatenate([q_of(critic(enc(obs(k), stop_gradient=True), st(k), act(k)))
                                       .min(0).values.float().cpu().numpy()
                                       for k in np.array_split(idx, max(1, len(idx) // bs))])
            qc = {e: q_at(fr) for e, fr, _ in eps}
        fin = np.array([qc[e][-1] for e, _, _ in eps])
        okm = np.array([o for _, _, o in eps])
        sq, fq = fin[okm], fin[~okm]
        auc = float((sq[:, None] > fq[None, :]).mean())
        print(f"  [eval] step {step:5d}  AUC {auc:.3f}  Q(성공끝) {sq.mean():+.3f}  "
              f"Q(실패끝) {fq.mean():+.3f}  Q범위 [{fin.min():+.3f},{fin.max():+.3f}]")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for e, fr, o in eps:
            ax.plot(np.arange(len(fr)), qc[e], lw=1, alpha=0.5,
                    color="tab:green" if o else "tab:red")
        ax.axhline(0, color="gray", lw=0.5); ax.axhline(1, color="gray", lw=0.5)
        ax.set_xlabel("frame in episode"); ax.set_ylabel("Q (min of ensemble)")
        ax.set_title(f"{'CQL' if a.no_calql else 'Cal-QL'} a={a.cql_alpha}"
                     f"{f' dist{a.bins}' if a.bins else ''}  "
                     f"step {step}  AUC {auc:.3f}  green=success")
        fig.tight_layout(); fig.savefig(ev / f"{step:06d}_q.png", dpi=110); plt.close(fig)
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({"enc": enc.state_dict(), "critic": critic.state_dict(),
                    "target": target.state_dict(),
                    "tenc": tenc.state_dict(),
                    "step": step, "exp": a.exp, "seed": a.seed, "cql_alpha": a.cql_alpha,
                    "discount": cfg.discount, "num_qs": cfg.num_qs, "calql": not a.no_calql, "ood": a.ood,
                    "latency": LAT, "replan": R, "action_dim": mod.action_dim,
                    "state_dim": snorm.shape[1], "bins": a.bins,
                    "q_range": a.q_range if a.bins else None}, tmp)
        os.replace(tmp, ckpt_path)
        print(f"  [저장] {ckpt_path} (step {step})")

print(f"[완료] {a.steps} 스텝 / 산출물 {ev}")


# --- PA-RL 의 parl_calql 과 같은 것 / 다른 것 ---------------------------------
#
# 같다:
#   · CQL 항의 형태: logsumexp(집합 ∪ {Q(s,a_data)}) − Q(s,a_data),  cql_alpha 가중
#     (cql.py:274-295 — log(N)·temp 를 빼고 logsumexp/temp·temp 하는 정규화까지 동일)
#   · Cal-QL 하한: OOD Q 를 mc_return 으로 clamp (cql.py:223 jnp.maximum)
#   · OOD 액션에 균등 [-1,1] 랜덤 포함 (cql_action_sample_method="uniform")
#   · 앙상블 10 + 타깃에 무작위 num_min_qs 개 min (REDQ), distributional HL-Gauss 옵션
#   · 마지막 층 1e-2 초기화, target 은 인코더까지 EMA, actor 는 학습하지 않는다
#   · cql_alpha 0.01 / cql_n_actions 4 = PA-RL 실기(real_config.py:79,94) 기본값
#
# 다르다 (이유):
#   · **OOD 집합에 정책 샘플이 없다.** PA-RL 은 {랜덤, π(·|s), π(·|s')} 세 종류를 쓰는데
#     (cql.py:105-145) π 가 곧 action-optimization = base policy 샘플이라 batch 마다 VLA 를
#     호출해야 한다 (update 당 70초). 대신 데이터 액션 + 노이즈(perturb)를 넣었다.
#     제대로 맞추려면 프레임당 후보를 미리 뽑아 캐시해야 한다 (2.4GB, PA-RL 도 OpenVLA 용으로
#     base_policy_offline_cache_path 로 같은 일을 한다).
#   · **TD 타깃의 next action 이 로그된 액션이다.** PA-RL 은 정책 후보의 argmax 를 쓴다
#     (cql_max_target_backup). 같은 VLA 비용 문제이고, 후보 캐시가 있으면 그대로 켤 수 있다.
#   · hidden_dims (256,256,256), tau 0.005 — EXPO yaml 값을 따랐다 (PA-RL 은 (256,256), 0.002)
#   · optimizer 1개 (PA-RL 은 critic/value 별로 두고 update 를 합산)
