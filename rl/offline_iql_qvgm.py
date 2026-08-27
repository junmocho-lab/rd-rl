#!/usr/bin/env python3
"""Q-VGM 식 critic — stepwise IQL + 층마다 액션 재주입.

offline_iql.py 와 무엇이 다른가 (Q-VGM arXiv 2606.08015 4.1 절):

  1. **stepwise 값**: 청크 전체에 값 하나가 아니라 실행 구간 위치별 Q^(i), V^(i)
     (i = 0..replan-1). 청크 안에서 다음 위치로 부트스트랩하고, 경계에서만 다음 청크의
     0번 위치로 넘어간다 (논문 Eq. 4):

         y_i = r_{t+i} + γ (1-d_{t+i}) V^(i+1)(s)      i < R-1
         y_{R-1} = r_{t+R-1} + γ (1-d_{t+R-1}) V^(0)(s')     s' = t+R

     Q^(0) 이 통상적인 프레임 단위 할인 가치이고, Q^(i) 는 i 프레임 뒤부터의 가치다.
     t → t+R 로 가는 데 프레임 단위 부트스트랩이 R번 걸리므로 결정당 실효 할인은 γ^R —
     offline_iql 과 같은 MDP 를 보지만 지도 신호가 R배 촘촘하다.
     ∇_A Q 에 스칼라가 필요하면 논문대로 Q(s,A) = Σ_i Q^(i)(s,A) 로 합친다 (리턴이 아니라
     겹치는 R개 리턴의 합 = 점수).

  2. **층마다 액션 재주입**: latent 4096 대 액션 280 이라 critic 이 액션을 무시하기 쉽다.
     hidden 층 입력마다 액션을 다시 concat 한다 (rl/nets.py StepwiseQ). ablation −4.3%p.

  3. **Q 헤드 2개의 min** (clipped double Q). 앙상블 10개가 아니다. ablation: 1개면 −2.4%p.

  4. **분포형(HL-Gauss) 안 씀**. Q-VGM 은 스칼라다. 분포형은 LWD/DIVL 쪽 선택이다.

γ 는 프레임 단위다 (offline_iql 과 같은 값을 주면 된다 — 여기서는 γ^R 로 안 올린다).

usage:
  PYTHONPATH="$L_PYTHONPATH" $L_PY -u -m rl.offline_iql_qvgm \
      --exp openarm_rim --data <데이터셋> --checkpoints checkpoints \
      --features cogfeat.npy --discount 0.995 --expectile 0.7 --steps 30000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

from rl.data import build_flat, build_images, find_sessions, open_images, resolve_modality
from rl.expo import ExpoConfig
from rl.nets import StepwiseEnsemble, StepwiseV, xavier_
from rl.offline_critic import normalize_all
from rl.vla_rldx import load_state_action_processor, normalize_actions, normalize_states

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="openarm_rim")
p.add_argument("--data", type=Path, required=True)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--features", default="cogfeat.npy",
               help="frozen VLM feature npy. Q-VGM 의 RL token 자리 (ablation: ResNet 이면 -5.1%p)")
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--discount", type=float, default=0.995, help="프레임 단위 γ")
p.add_argument("--expectile", type=float, default=0.7)
p.add_argument("--num-qs", type=int, default=2, help="Q-VGM 은 2 (clipped double Q)")
p.add_argument("--latent", type=int, default=512, help="feature projection 차원")
p.add_argument("--no-inject", action="store_true", help="층마다 액션 재주입을 끈다 (ablation)")
p.add_argument("--holdout", default="0.2",
               help="세션 이름 문자열 또는 에피소드 비율(0<x<1). 기본값을 offline_iql 과\n"
                    "같게 둔다 — 다르면 같은 --holdout 을 줘도 태그 접미사가 한쪽만 붙어\n"
                    "(기본값과 다를 때만 붙는다) 디렉토리 이름이 비대칭이 된다")
p.add_argument("--train-eps", default="all", choices=("all", "success", "fail"))
p.add_argument("--eval-every", type=int, default=3000)
p.add_argument("--tag", default="")
p.add_argument("--keep-last", type=int, default=1)
p.add_argument("--no-wandb", action="store_true", help="wandb 를 끈다")
p.add_argument("--wandb-key", default="", help="비우면 WANDB_API_KEY / ~/.netrc 를 쓴다")
p.add_argument("--wandb-project", default="rd-rl-critic")
p.add_argument("--wandb-entity", default="", help="비우면 계정 기본 entity")
p.add_argument("--wandb-run", default="", help="run 이름. 비우면 태그와 같게 둔다")
p.add_argument("--log-every", type=int, default=100, help="wandb 로깅 주기")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda")
a = p.parse_args()
dev = a.device
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / exp["base_policy"]
TAG = a.tag or (f"qvgm-cog-t{str(a.expectile).replace('.', '')}"
                f"-g{f'{a.discount:g}'.replace('.', '')}-q{a.num_qs}"
                f"{'-noinject' if a.no_inject else ''}"
                f"-s{a.seed}"
                + ("" if a.holdout == p.get_default("holdout")
                   else "-h" + a.holdout.replace(".", ""))
                + ("" if a.train_eps == "all" else f"-{a.train_eps}only"))
run = work / TAG
plots = run / "plots"
plots.mkdir(parents=True, exist_ok=True)

# --- 1. 데이터 --------------------------------------------------------------
mod, _ = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
build_images(sessions, flat, work / "images.mm", mod)
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
if not (work / "actnorm.npy").is_file():
    normalize_all(lambda ch, st: normalize_actions(proc, mod.embodiment_tag, mod, ch, st),
                  flat, H, cache=work / "actnorm.npy")
norm = normalize_all(None, flat, H, cache=work / "actnorm.npy")

fp = np.load(work / a.features, mmap_mode="r")
assert fp.shape[0] == len(flat), f"feature 프레임 {fp.shape[0]} != {len(flat)}"
FEAT = torch.from_numpy(np.ascontiguousarray(np.asarray(fp))).to(dev)
MU, SD = FEAT.mean(0, keepdim=True), FEAT.std(0, keepdim=True).clamp_min(1e-3)
FEAT = (FEAT - MU) / SD
SNORM = torch.from_numpy(snorm).to(dev)
NORM = torch.from_numpy(np.ascontiguousarray(np.asarray(norm[:, :LAT + R]))).to(dev)
A_DIM, FULL = mod.action_dim, (LAT + R) * mod.action_dim

# stepwise 보상/종료: (T, R).  **클램프한 인덱스를 그대로 쓰면 안 된다** — 에피소드 끝을
# 넘는 위치가 마지막 프레임을 반복하므로 성공 종단 보상 1 이 여러 위치에 복제된다.
# 범위 밖은 보상 0 / mask 0 / 손실 가중치 0 으로 죽인다.
_raw = np.arange(len(flat))[:, None] + np.arange(R)[None, :]
_inb = (_raw <= flat.ep_end[:, None]).astype(np.float32)          # (T,R) 에피소드 안인지
off = np.minimum(_raw, flat.ep_end[:, None])
RSTEP = torch.from_numpy((flat.reward[off] * _inb).astype(np.float32)).to(dev)
MSTEP = torch.from_numpy(((1.0 - flat.done[off]) * _inb).astype(np.float32)).to(dev)
NEXT = torch.from_numpy(np.minimum(np.arange(len(flat)) + R, flat.ep_end)).to(dev)
# 범위 밖 위치는 reward 0 / mask 0 이라 타깃이 정확히 0 이다 → **손실 가중치를 0 으로 두지
# 않는다.** 0 으로 두면 그 헤드가 아무 값이나 내고 Σ_i Q^(i) 가 오염된다 (실측: 에피소드
# 마지막 프레임에서 Q_sum 이 3.04, 실제 리턴은 1). 0 으로 회귀시켜 합을 살린다.
# done=1 이 에피소드 마지막 프레임에 반드시 있으므로(실측 300/300) 경계 부트스트랩도
# mask 로 자동 차단된다 — 별도 valid 플래그가 필요 없다.

frac = float(a.holdout) if a.holdout.replace(".", "", 1).isdigit() else 0.0
if 0 < frac < 1:
    every = max(2, int(round(1 / frac)))
    hold = np.isin(flat.episode, np.unique(flat.episode)[::every])
    how = f"에피소드 {every}개마다 1개"
else:
    sel = [i for i, n in enumerate(flat.sessions) if a.holdout and a.holdout in n]
    hold = np.isin(flat.session, sel)
    how = f"세션 '{a.holdout}'"
train = np.flatnonzero(~hold[:len(flat) - R])
n_all = len(train)
if a.train_eps != "all":
    train = train[flat.is_success[train] == (a.train_eps == "success")]
eps = [(e, np.flatnonzero(flat.episode == e)) for e in np.unique(flat.episode[hold])]
eps = [(e, fr, bool(flat.is_success[fr[-1]])) for e, fr in eps]

print(f"[exp] {a.exp} replan={R} latency={LAT} horizon={H} → 액션 {FULL}차원, stepwise {R}개")
print(f"[데이터] 프레임 {len(flat)} / 학습 {len(train)}"
      f"{'' if a.train_eps == 'all' else f' ({len(train)/n_all:.1%} of {n_all})'}"
      f" / 평가 에피소드 {len(eps)} (성공 {sum(o for _,_,o in eps)})  홀드아웃: {how}")
# 부트스트랩이 프레임 단위 1스텝씩 R번 걸려 t → t+R 이 되므로 결정당 실효 할인은 γ^R 이고,
# 지평은 프레임 단위로 1/(1-γ) 이다 (offline_iql 의 γ^R 백업과 같은 MDP).
print(f"[할인] 프레임당 γ={a.discount} → 결정당 γ^R={a.discount**R:.5f}  "
      f"지평 {1/(1-a.discount):.0f} 프레임 = {1/(1-a.discount)/R:.1f} 결정")
print(f"[feature] {a.features} {tuple(fp.shape)} → VRAM {FEAT.numel()*4/1e9:.2f}GB")
print(f"[산출물] {run}/")


# --- wandb (offline_iql.py 와 같은 규약. 토큰은 절대 config 에 적지 않는다) -----
RUN_CFG = {"tag": TAG, **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "wandb_key": "***" if (a.wandb_key or os.environ.get("WANDB_API_KEY")) else ""}
(run / "config.json").write_text(json.dumps(RUN_CFG, indent=2, default=str, ensure_ascii=False))
wb = None
if not a.no_wandb:
    import wandb
    key = a.wandb_key or os.environ.get("WANDB_API_KEY", "")
    try:
        if key:
            wandb.login(key=key)
        wb = wandb.init(project=a.wandb_project, entity=a.wandb_entity or None,
                        name=a.wandb_run or TAG, dir=str(run), config=RUN_CFG)
        print(f"[wandb] {wb.url}")
    except Exception as e:                       # 로깅 때문에 학습을 날리지 않는다
        wb = None
        print(f"[wandb] 붙지 못했다 → 로깅 없이 진행한다 ({type(e).__name__}: {e})")
else:
    print("[wandb] --no-wandb")

# --- 2. critic --------------------------------------------------------------
class Proj(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = xavier_(nn.Linear(din, dout))
        self.ln = nn.LayerNorm(dout)

    def forward(self, x):
        return self.ln(self.lin(x))


enc = Proj(FEAT.shape[1], a.latent).to(dev)
IN = a.latent + snorm.shape[1]                     # state 는 raw 로 붙인다 (offline_iql 과 동일)
inject = not a.no_inject
critic = StepwiseEnsemble(IN, FULL, R, a.num_qs, cfg.hidden_dims,
                          cfg.critic_layer_norm, inject).to(dev)
value = StepwiseV(IN, R, cfg.hidden_dims, cfg.critic_layer_norm).to(dev)
target = StepwiseEnsemble(IN, FULL, R, a.num_qs, cfg.hidden_dims,
                          cfg.critic_layer_norm, inject).to(dev)
tenc = Proj(FEAT.shape[1], a.latent).to(dev)
target.load_state_dict(critic.state_dict())
tenc.load_state_dict(enc.state_dict())
with torch.no_grad():                              # PA-RL kernel_scale_final=1e-2 와 같은 취지
    for m in list(critic.qs) + [value]:
        m.head.weight.mul_(1e-2)
        m.head.bias.zero_()
    target.load_state_dict(critic.state_dict())
opt = torch.optim.Adam(list(enc.parameters()) + list(critic.parameters())
                       + list(value.parameters()), lr=a.lr)
print(f"[critic] stepwise Q x{a.num_qs} (min), 액션 층마다 재주입 {inject}, "
      f"학습 파라미터 {sum(p.numel() for g in opt.param_groups for p in g['params'])/1e6:.1f}M")


def lat_of(i, tgt=False):
    z = (tenc if tgt else enc)(FEAT[i])
    return torch.cat([z, SNORM[i]], -1)


def act_of(i):
    return NORM[i].reshape(len(i), -1)


# --- 3. 학습 ----------------------------------------------------------------
TRAIN = torch.from_numpy(train).to(dev)
t0 = time.time()
for step in range(1, a.steps + 1):
    i = TRAIN[torch.randint(len(TRAIN), (a.batch,), device=dev)]
    j = NEXT[i]
    with torch.no_grad():
        # 청크 안 부트스트랩 (Q-VGM Eq. 4): 위치 i<R-1 은 **같은** 상태의 V^(i+1),
        # 마지막 위치만 다음 청크의 V^(0)(s') 로 넘어간다.
        v_same = value(lat_of(i))                           # (B,R) = V^(0..R-1)(s)
        v_next = torch.cat([v_same[:, 1:], value(lat_of(j))[:, :1]], 1)
        tq = RSTEP[i] + a.discount * MSTEP[i] * v_next
        qt = target(lat_of(i, tgt=True), act_of(i)).min(0).values   # (B,R)

    lat = lat_of(i)
    q = critic(lat, act_of(i))                               # (num_qs, B, R)
    v = value(lat)                                           # (B, R)
    q_loss = ((q - tq[None]) ** 2).mean()
    adv = qt - v
    v_loss = (torch.where(adv > 0, a.expectile, 1 - a.expectile) * adv ** 2).mean()
    loss = q_loss + v_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        for pt, ps in zip(target.parameters(), critic.parameters()):
            pt.mul_(1 - cfg.tau).add_(ps, alpha=cfg.tau)
        for pt, ps in zip(tenc.parameters(), enc.parameters()):
            pt.mul_(1 - cfg.tau).add_(ps, alpha=cfg.tau)

    if wb is not None and (step % a.log_every == 0 or step == a.steps):
        wb.log({"loss/q": float(q_loss), "loss/v": float(v_loss),
                "loss/total": float(loss),
                "q/sum_min": float(q.min(0).values.sum(-1).mean()),
                "v/sum": float(v.sum(-1).mean())}, step=step)
    if step % 200 == 0:
        print(f"  step {step:6d}  q {float(q_loss):.5f}  v {float(v_loss):.5f}  "
              f"Q_sum {float(q.min(0).values.sum(-1).mean()):+.3f}  "
              f"V_sum {float(v.sum(-1).mean()):+.3f}  {(time.time()-t0)/step*1000:.0f}ms/step",
              flush=True)

    if step % a.eval_every == 0 or step == a.steps:
        with torch.no_grad():
            curves = {}
            for e, fr, ok in eps:
                qs, vs = [], []
                for k in np.array_split(fr, max(1, len(fr) // 256)):
                    kk = torch.from_numpy(k).to(dev)
                    l = lat_of(kk)
                    qs.append(critic(l, act_of(kk)).min(0).values.sum(-1).float().cpu().numpy())
                    vs.append(value(l).sum(-1).float().cpu().numpy())
                curves[e] = (np.concatenate(qs), np.concatenate(vs))
        fin = np.array([curves[e][0][-1] for e, _, _ in eps])
        okm = np.array([o for _, _, o in eps])
        auc = float((fin[okm][:, None] > fin[~okm][None, :]).mean())
        print(f"  [eval] step {step:6d}  AUC {auc:.3f}  "
              f"Q(성공끝) {fin[okm].mean():+.3f}  Q(실패끝) {fin[~okm].mean():+.3f}")
        if wb is not None:
            wb.log({"eval/auc": auc, "eval/q_end_success": float(fin[okm].mean()),
                    "eval/q_end_fail": float(fin[~okm].mean())}, step=step)
        fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for e, fr, ok in eps:
            c = "tab:green" if ok else "tab:red"
            qv, vv = curves[e]
            axs[0].plot(np.arange(len(fr)), qv, lw=1, alpha=0.5, color=c)
            axs[1].plot(np.arange(len(fr)), vv, lw=1, alpha=0.5, color=c)
            axs[2].plot(np.arange(len(fr)), qv - vv, lw=1, alpha=0.5, color=c)
        for ax, nm in zip(axs, (f"Q = sum_i Q^(i)  (min of {a.num_qs})",
                                "V = sum_i V^(i)", "A = Q - V")):
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_ylabel(nm, fontsize=9)
            ax.grid(alpha=0.25)
        axs[2].set_xlabel("frame in episode")
        axs[0].set_title(f"Q-VGM critic  step {step}  AUC {auc:.3f}  green=success")
        fig.tight_layout()
        fig.savefig(plots / f"{step:06d}_qv.png", dpi=110)
        plt.close(fig)

        ck = run / f"critic_{step:06d}.pt"
        tmp = ck.with_suffix(".pt.tmp")
        torch.save({"enc": enc.state_dict(), "critic": critic.state_dict(),
                    "value": value.state_dict(), "target": target.state_dict(),
                    "tenc": tenc.state_dict(), "kind": "qvgm",
                    "step": step, "exp": a.exp, "seed": a.seed, "expectile": a.expectile,
                    "discount": a.discount, "num_qs": a.num_qs, "n_steps": R,
                    "latency": LAT, "replan": R, "action_dim": A_DIM, "inject": inject,
                    "state_dim": snorm.shape[1], "latent": a.latent, "tag": TAG,
                    "hidden_dims": list(cfg.hidden_dims),
                    "critic_layer_norm": cfg.critic_layer_norm,
                    "features": a.features, "feat_mu": MU.cpu(), "feat_sd": SD.cpu()}, tmp)
        os.replace(tmp, ck)
        lnk, ltmp = run / "critic_latest.pt", run / "critic_latest.pt.tmp"
        ltmp.unlink(missing_ok=True)
        ltmp.symlink_to(ck.name)
        os.replace(ltmp, lnk)
        print(f"  [저장] {ck.name} ({ck.stat().st_size/1e6:.0f}MB)")
        if a.keep_last > 0:
            for f in sorted(run.glob("critic_[0-9]*.pt"))[:-a.keep_last]:
                f.unlink()

if wb is not None:
    wb.finish()
print(f"\n[완료] {run}")
