#!/usr/bin/env python3
"""Phase D — critic 오프라인 학습과 게이트.

기존 롤아웃만으로 critic 을 TD 학습하고 **Q 가 성공/실패를 가르는지** 본다.
next_action 을 로그된 액션으로 두므로 behavior policy 의 가치 Q^pi 를 배우는 것이고,
VLA 후보 샘플링이 루프에 없어 빠르다 (51ms/obs 가 빠진다).

보상이 성공 에피소드의 마지막 1프레임에만 1.0 이므로 어떤 상태에서든 리턴 상한이 1 이다:
    Q(s,a) ~= gamma_eff^(성공까지 남은 매크로 스텝) x P(성공)   ->   Q in [0, 1]

게이트:
  1 Q 가 [0, 1.2] 안         (넘으면 발산)
  2 held-out 세션에서 성공 에피소드 마지막 Q > 실패 에피소드 마지막 Q  (AUC)
  3 에피소드 구간별 Q 기울기 — 유효 지평이 어디까지 닿는지
  4 후보 간(=액션 간) Q 분산 > 0 — critic 이 상태가 아니라 액션도 구분하는지

usage:
  cd third_party/RLDX-1 && PYTHONPATH="$PWD:<repo>" pixi run -e rldx python -u -m rl.offline_critic \\
      --exp openarm_rim --images <img.mm> [--steps 3000]

**-u 를 붙일 것.** 로그를 파일로 리다이렉트하면 stdout 이 블록 버퍼링되어 프로세스가 정상
동작 중인데도 로그가 얼어붙은 것처럼 보인다 (실제로 그것 때문에 잘 돌던 학습을 죽였다).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from rl.data import build_flat, find_sessions, nstep, open_images, resolve_modality
from rl.expo import ExpoConfig
from rl.nets import BatchEncoder, CriticEnsemble

REPO = Path(__file__).resolve().parent.parent


def normalize_all(vla, flat, action_horizon: int, cache: Path | None = None,
                  chunk: int = 4096) -> np.ndarray:
    """모든 프레임의 액션 청크를 모델 공간으로. (T, H, A)

    state(t) 가 고정이면 결과도 고정이므로 **한 번 계산해 저장한다**. 매 배치 재계산은
    순수 낭비이고 (apply_action 은 python 루프라 CPU 병목이 된다), 실제 파이프라인에서는
    ingest 시점에 새로 온 에피소드만 정규화하면 된다 (5 에피소드면 수 초).
    저장 비용: (T,H,A) float32 = openarm 64,619 프레임이면 116MB.
    """
    from rl.data import action_chunk

    if cache and cache.is_file():
        out = np.load(cache, mmap_mode="r")
        if out.shape == (len(flat), action_horizon, flat.action.shape[1]):
            print(f"  [정규화] 캐시 사용 {cache.name}")
            return np.asarray(out)
        print(f"  [정규화] 캐시 shape 불일치 {out.shape} — 다시 계산")
    T = len(flat)
    out = np.empty((T, action_horizon, flat.action.shape[1]), np.float32)
    t0 = time.time()
    for i in range(0, T, chunk):
        idx = np.arange(i, min(i + chunk, T))
        out[idx] = vla.normalize_actions(action_chunk(flat, idx, action_horizon), flat.state[idx])
    print(f"  [정규화] {T} 프레임 {time.time()-t0:.1f}s")
    if cache:
        np.save(cache, out)
        print(f"  [정규화] 캐시 저장 {cache} ({out.nbytes/1e6:.0f}MB)")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="openarm_rim")
    p.add_argument("--data", type=Path,
                   default=REPO / "rl-dataset/r0/0815_openarm_rh56f1_inference")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--holdout", default="eval",
                   help="세션 이름에 이 문자열이 있으면 평가 전용으로 뺀다")
    p.add_argument("--ckpt-root", type=Path, default=Path("/home/openarm14/ws/junmo_cho/checkpoints"))
    p.add_argument("--save", type=Path, help="학습한 critic/encoder 저장")
    p.add_argument("--load", type=Path, help="저장한 것을 불러와 진단만 한다")
    p.add_argument("--plot", type=Path, help="에피소드별 Q 곡선 PNG")
    p.add_argument("--n-plot", type=int, default=10, help="성공/실패 각 몇 개를 그릴지")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward-frac", type=float, default=0.0,
                   help="배치에서 보상이 걸린 transition 의 비율을 강제한다 (예: 0.08). "
                        "0 이면 균등 샘플링(원본과 동일). 우리 에피소드가 EXPO-FT 의 3.8배로 "
                        "길어 균등 샘플링의 보상 밀도가 2%%(원본 8%%)로 떨어지는 것을 보정하는 용도")
    p.add_argument("--norm-device", default="cuda",
                   help="정규화용 VLA 를 올릴 장치. apply_action 은 순수 numpy 라 GPU 가 "
                        "필요 없다 — 다른 학습이 GPU 를 쓰고 있으면 cpu 로 두면 공존한다")
    a = p.parse_args()

    # 재현성: 이걸 안 걸면 critic 초기값이 실행마다 달라져 같은 스텝에서 Q 가 크게 다르다
    # (실측: 1000 스텝에서 Q 가 -1.13 과 -0.07 로 갈렸다).
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    exp = yaml.safe_load((REPO / "configs" / "exp" / f"{a.exp}.yaml").read_text())
    cfg = ExpoConfig.from_dict(exp.get("expo"))
    R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
    base = a.ckpt_root / exp["base_policy"]

    mod, src = resolve_modality(a.data, None, REPO / "third_party/RLDX-1",
                               exp["rldx_data_config"], base)
    print(f"[exp] {a.exp}  replan={R} latency={LAT} horizon={H}")
    print(f"[modality] {src}")
    sessions = find_sessions(a.data)
    flat = build_flat(sessions, mod)
    imgs, meta = open_images(a.images)
    n_cams = mod.n_cams
    dev = "cuda"

    # 정규화된 액션 (모델 공간) — critic 이 롤아웃 때 보는 것과 같은 공간
    from rl.vla_rldx import RLDXVLA
    cache = a.images.with_suffix(".actnorm.npy")
    if cache.is_file():
        norm = normalize_all(None, flat, H, cache=cache)      # 캐시가 있으면 VLA 를 안 올린다
    else:
        vla = RLDXVLA(base, mod, REPO / "third_party/RLDX-1", exp["rldx_data_config"],
                      device=a.norm_device)
        norm = normalize_all(vla, flat, H, cache=cache)
        del vla
        torch.cuda.empty_cache()

    # held-out 분리
    hold_sid = [i for i, s in enumerate(flat.sessions) if a.holdout in s]
    is_hold = np.isin(flat.session, hold_sid)
    train_idx = np.flatnonzero(~is_hold[: len(flat) - R])
    print(f"[분할] 학습 {len(train_idx)} 프레임 / 평가 세션 {[flat.sessions[i] for i in hold_sid]}"
          f" ({int(is_hold.sum())} 프레임)")

    # 에피소드 단위 정보 (게이트용)
    ep_ids = np.unique(flat.episode)
    ep_last = np.array([np.flatnonzero(flat.episode == e)[-1] for e in ep_ids])
    ep_succ = np.array([bool(flat.is_success[i]) for i in ep_last])
    ep_hold = np.array([bool(is_hold[i]) for i in ep_last])
    print(f"[에피소드] 전체 {len(ep_ids)} (성공 {ep_succ.sum()}) / "
          f"평가 {ep_hold.sum()} (성공 {(ep_succ & ep_hold).sum()})")

    enc = BatchEncoder(3 * n_cams, cfg.latent_dim_image, cfg.encoder_stage_sizes,
                       cfg.encoder_num_filters).to(dev)
    full = R * mod.action_dim
    critic = CriticEnsemble(cfg.latent_dim_image, mod.state_dim, full, cfg.num_qs,
                            cfg.latent_dim_state, cfg.include_state, cfg.hidden_dims,
                            cfg.critic_layer_norm).to(dev)
    import copy
    target = copy.deepcopy(critic).requires_grad_(False)
    opt = torch.optim.Adam(list(critic.parameters()) + list(enc.parameters()), lr=cfg.critic_lr)
    gen = torch.Generator().manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    # 보상이 걸린 학습 인덱스 (창 [t, t+R) 안에 성공 프레임이 들어가는 t)
    rw = nstep(flat, train_idx, R, cfg.discount)["reward"]
    reward_idx = train_idx[rw > 0]
    n_rw = int(round(a.reward_frac * cfg.batch_size))
    print(f"[보상 밀도] 보상 있는 transition {len(reward_idx)}/{len(train_idx)} "
          f"= {len(reward_idx)/len(train_idx):.2%}"
          + (f"  → 배치당 {n_rw}/{cfg.batch_size} 강제 ({n_rw/cfg.batch_size:.0%})"
             if n_rw else "  (강제 없음, 균등 샘플링)"))

    def act_of(idx):        # critic 입력 액션 = 정규화 청크의 [latency, latency+R)
        return torch.from_numpy(norm[idx][:, LAT:LAT + R].reshape(len(idx), -1)).to(dev)

    def obs_of(idx):
        x = np.asarray(imgs[idx])
        return torch.from_numpy(np.concatenate([x[:, c] for c in range(x.shape[1])],
                                               axis=-1)).to(dev)

    def st_of(idx):
        return torch.from_numpy(flat.state[idx]).to(dev)

    @torch.no_grad()
    def q_at(idx, bs=128):
        out = []
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            lat = enc(obs_of(j), stop_gradient=True)
            out.append(critic(lat, st_of(j), act_of(j)).min(dim=0).values.float().cpu().numpy())
        return np.concatenate(out)

    def gate(step):
        # 게이트 2: held-out 에피소드 마지막 프레임의 Q
        hl = ep_last[ep_hold]
        hs = ep_succ[ep_hold]
        q = q_at(hl)
        qs, qf = q[hs], q[~hs]
        # AUC (성공이 실패보다 높을 확률)
        auc = float((qs[:, None] > qf[None, :]).mean()) if len(qs) and len(qf) else float("nan")
        print(f"  step {step:5d}  Q(성공끝) {qs.mean():+.4f}±{qs.std():.3f}  "
              f"Q(실패끝) {qf.mean():+.4f}±{qf.std():.3f}  차이 {qs.mean()-qf.mean():+.4f}  "
              f"AUC {auc:.3f}  Q범위 [{q.min():+.3f},{q.max():+.3f}]")
        return auc, qs.mean() - qf.mean(), float(q.min()), float(q.max())

    auc = gap = qmin = qmax = float("nan")
    if a.load:
        sd = torch.load(a.load, map_location=dev)
        enc.load_state_dict(sd["enc"]); critic.load_state_dict(sd["critic"])
        target.load_state_dict(sd["target"])
        print(f"[불러옴] {a.load} (step {sd.get('step')})")
        auc, gap, qmin, qmax = gate(sd.get("step", 0))
    print(f"[학습] {0 if a.load else a.steps} 스텝, batch {cfg.batch_size}")
    t0 = time.time()
    for step in range(1, (0 if a.load else a.steps) + 1):
        if n_rw:
            idx = np.concatenate([
                reward_idx[rng.integers(0, len(reward_idx), size=n_rw)],
                train_idx[rng.integers(0, len(train_idx), size=cfg.batch_size - n_rw)]])
        else:
            idx = train_idx[rng.integers(0, len(train_idx), size=cfg.batch_size)]
        n = nstep(flat, idx, R, cfg.discount)
        nxt = n["next_idx"]
        with torch.no_grad():
            nl = enc(obs_of(nxt), stop_gradient=True)
            members = critic.subsample(cfg.num_min_qs, gen)
            nq = target(nl, st_of(nxt), act_of(nxt), members=members).min(dim=0).values
            tq = (torch.from_numpy(n["reward"]).to(dev)
                  + (cfg.discount ** R) * torch.from_numpy(n["mask"]).to(dev) * nq)
        lat = enc(obs_of(idx), stop_gradient=cfg.freeze_critic_encoder)
        qs = critic(lat, st_of(idx), act_of(idx))
        loss = (((qs - tq.unsqueeze(0)) ** 2)
                * torch.from_numpy(n["valid"]).to(dev).unsqueeze(0)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, pp in zip(target.parameters(), critic.parameters()):
                tp.mul_(1 - cfg.tau).add_(pp, alpha=cfg.tau)
        if step % a.eval_every == 0 or step == a.steps:
            print(f"  loss {float(loss.detach()):.5f}  q {float(qs.detach().mean()):+.4f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")
            auc, gap, qmin, qmax = gate(step)

    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"enc": enc.state_dict(), "critic": critic.state_dict(),
                    "target": target.state_dict(), "step": a.steps}, a.save)
        print(f"[저장] {a.save}")

    # --- 에피소드별 Q 곡선 진단 (스칼라 3개 + PNG) --------------------------
    print("\n[진단] held-out 에피소드의 프레임별 Q")
    curves = []
    for e, last, ok, hd in zip(ep_ids, ep_last, ep_succ, ep_hold):
        if not hd:
            continue
        fr = np.flatnonzero(flat.episode == e)
        fr = fr[fr < len(flat) - R]
        if len(fr) < 10:
            continue
        step_i = max(1, len(fr) // 60)                  # 에피소드당 최대 60점
        ii = fr[::step_i]
        q = q_at(ii)
        remain = (fr[-1] - ii) / R                       # 남은 매크로 스텝
        curves.append({"succ": bool(ok), "q": q, "remain": remain,
                       "elapsed": (ii - fr[0]) / R, "n_frames": len(fr)})
    sc = [c for c in curves if c["succ"]]
    fc = [c for c in curves if not c["succ"]]
    print(f"  에피소드 {len(sc)} 성공 / {len(fc)} 실패")
    # 길이가 결과와 상관되므로 (실패가 짧다) 진행률 축은 오해를 만든다 — 남은 스텝으로 본다
    ls = np.array([c["n_frames"] for c in sc]); lf = np.array([c["n_frames"] for c in fc])
    print(f"  길이  성공 {ls.mean():.0f} 프레임 ({ls.mean()/R:.1f} 매크로스텝) / "
          f"실패 {lf.mean():.0f} ({lf.mean()/R:.1f})  → 실패가 성공의 {lf.mean()/ls.mean():.0%}")

    # 지표 1: 성공 에피소드에서 Q 가 gamma^남은스텝 과 상관되나
    gam = cfg.discount ** R
    xs = np.concatenate([gam ** c["remain"] for c in sc])
    ys = np.concatenate([c["q"] for c in sc])
    r1 = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")

    # 지표 2: 마지막 10% 구간 평균 Q 의 AUC
    def tail_mean(c, frac=0.1):
        k = max(1, int(len(c["q"]) * frac))
        return float(c["q"][-k:].mean())
    ts = np.array([tail_mean(c) for c in sc]); tf = np.array([tail_mean(c) for c in fc])
    auc_tail = float((ts[:, None] > tf[None, :]).mean()) if len(ts) and len(tf) else float("nan")

    # 지표 3: 마지막 프레임 캘리브레이션 (성공 → 1, 실패 → 0)
    fs = np.array([c["q"][-1] for c in sc]); ff = np.array([c["q"][-1] for c in fc])
    cal = float(np.abs(fs - 1.0).mean() + np.abs(ff - 0.0).mean()) / 2

    print(f"  지표1 Q vs γ^남은스텝 상관 (성공)     {r1:+.3f}   (> 0.7 면 모양이 맞다)")
    print(f"  지표2 마지막 10% 평균 Q 의 AUC        {auc_tail:.3f}   (> 0.8)")
    print(f"  지표3 마지막 프레임 캘리브레이션 오차  {cal:.3f}   (< 0.3, 성공→1 실패→0)")
    print(f"        성공 마지막 Q {fs.mean():+.3f}±{fs.std():.3f} / "
          f"실패 마지막 Q {ff.mean():+.3f}±{ff.std():.3f}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))

        # (1) 종료 정렬 — Q ~ gamma^남은스텝 이므로 이 축이 자연스럽다.
        #     성공/실패를 "종료까지 남은 시간" 이 같은 지점에서 비교한다.
        for c in sc[:a.n_plot]:
            ax[0].plot(-c["remain"], c["q"], color="tab:green", alpha=0.45, lw=1)
        for c in fc[:a.n_plot]:
            ax[0].plot(-c["remain"], c["q"], color="tab:red", alpha=0.45, lw=1)
        grid = np.linspace(-30, 0, 40)
        for cs, col, lab in ((sc, "tab:green", "success"), (fc, "tab:red", "failure")):
            m = np.nanmean([np.interp(grid, -c["remain"][::-1], c["q"][::-1]) for c in cs], axis=0)
            ax[0].plot(grid, m, color=col, lw=3, label=f"{lab} mean (n={len(cs)})")
        ax[0].plot(grid, gam ** (-grid), "k--", lw=1.5, label="ideal γ^remaining (success)")
        ax[0].axhline(0, color="gray", lw=0.5); ax[0].axhline(1, color="gray", lw=0.5)
        ax[0].set_xlabel("macro steps remaining (0 = episode end)")
        ax[0].set_ylabel("Q (min of ensemble)")
        ax[0].set_title("end-aligned — the axis Q actually depends on")
        ax[0].legend(fontsize=8)

        # (2) 시작 정렬 — 절대 시간에서 언제 갈라지는지. 실패가 더 짧다는 것도 보인다.
        for c in sc[:a.n_plot]:
            ax[1].plot(c["elapsed"], c["q"], color="tab:green", alpha=0.45, lw=1)
        for c in fc[:a.n_plot]:
            ax[1].plot(c["elapsed"], c["q"], color="tab:red", alpha=0.45, lw=1)
        ax[1].axhline(0, color="gray", lw=0.5)
        ax[1].set_xlabel("macro steps from start")
        ax[1].set_title(f"start-aligned  (len: succ {ls.mean()/R:.0f} / fail {lf.mean()/R:.0f} steps)")

        ax[2].hist(fs, bins=20, alpha=0.6, color="tab:green", label=f"success final (n={len(fs)})")
        ax[2].hist(ff, bins=20, alpha=0.6, color="tab:red", label=f"failure final (n={len(ff)})")
        ax[2].axvline(0, color="k", ls=":"); ax[2].axvline(1, color="k", ls=":")
        ax[2].set_xlabel("Q at final frame")
        ax[2].set_title(f"AUC(tail 10%) = {auc_tail:.3f}  |  corr = {r1:+.2f}")
        ax[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        print(f"  [그림] {a.plot}")

    # 길이 통제 진단 — AUC 가 "성공이라서" 높은지 "더 길어서" 높은지 가른다.
    # 성공 306 프레임 / 실패 202 프레임이라 마지막 10% 지점의 절대 시각이 다르다.
    print("\n[길이 통제] 같은 절대 결정 스텝에서 (그 시점에 아직 진행 중인 에피소드만)")
    for s in (5, 10, 15, 20, 25):
        qs_, qf_ = [], []
        for c in curves:
            k = np.flatnonzero(c["elapsed"] >= s)
            if len(k) == 0:
                continue
            (qs_ if c["succ"] else qf_).append(float(c["q"][k[0]]))
        if len(qs_) < 3 or len(qf_) < 3:
            continue
        qs_, qf_ = np.array(qs_), np.array(qf_)
        auc_s = float((qs_[:, None] > qf_[None, :]).mean())
        print(f"  결정 {s:3d} 스텝  성공 {qs_.mean():+.4f} (n={len(qs_)})  "
              f"실패 {qf_.mean():+.4f} (n={len(qf_)})  차이 {qs_.mean()-qf_.mean():+.4f}  "
              f"AUC {auc_s:.3f}")
    print("  → 차이가 0 근처면 앞서의 AUC 는 결과 신호가 아니라 길이 차이에서 온 것")

    # 게이트 3: 에피소드 구간별 Q (크레딧 지평 관측)
    print("\n[게이트 3] held-out 에피소드 구간별 Q (진행률 기준)")
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        sel = []
        for e, last, ok, hd in zip(ep_ids, ep_last, ep_succ, ep_hold):
            if not hd:
                continue
            fr = np.flatnonzero(flat.episode == e)
            k = min(int(len(fr) * frac), len(fr) - 1)
            if fr[k] < len(flat) - R:
                sel.append((fr[k], ok))
        if not sel:
            continue
        ii = np.array([s[0] for s in sel]); oo = np.array([s[1] for s in sel])
        q = q_at(ii)
        print(f"  진행률 {frac:4.0%}  Q(성공) {q[oo].mean():+.4f}  Q(실패) {q[~oo].mean():+.4f}  "
              f"차이 {q[oo].mean()-q[~oo].mean():+.4f}")

    # 게이트 4: 같은 상태에서 액션을 바꾸면 Q 가 변하나
    print("\n[게이트 4] 같은 상태 / 다른 액션의 Q 분산")
    ii = train_idx[rng.integers(0, len(train_idx), size=32)]
    with torch.no_grad():
        lat = enc(obs_of(ii), stop_gradient=True)
        base_a = act_of(ii)
        qs = []
        for scale in (0.0, 0.1, 0.3, 1.0):
            pert = base_a + scale * torch.randn_like(base_a)
            qs.append(critic(lat, st_of(ii), pert.clamp(-1.2, 1.2)).min(0).values)
    qs = torch.stack(qs).float().cpu().numpy()
    for s, q in zip((0.0, 0.1, 0.3, 1.0), qs):
        print(f"  노이즈 {s:4.1f} → Q 평균 {q.mean():+.4f} (원본 대비 {q.mean()-qs[0].mean():+.4f})")
    print(f"  액션 섭동에 따른 Q 변화폭 {np.abs(qs - qs[0]).max():.4f}"
          f"  {'(0 이면 critic 이 액션을 무시)' if np.abs(qs - qs[0]).max() < 1e-3 else ''}")

    print("\n=== 게이트 판정 ===")
    # 보상이 성공 종료 1프레임에만 1.0 이므로 리턴 상한이 1 이다 → Q 는 [0,1] 이어야 한다.
    # 음수로 눌려 있으면 부트스트랩이 아직 안 올라온 것(수렴 부족)이고, 1.2 를 넘으면 발산.
    ok1 = qmax <= 1.2 and qmin >= -0.3
    ok2 = auc >= 0.7 and gap > 0.05
    print(f"  {'OK  ' if ok1 else 'FAIL'} 1 Q 가 [-0.3, 1.2] 안 — 범위 [{qmin:+.3f}, {qmax:+.3f}]")
    print(f"  {'OK  ' if ok2 else 'FAIL'} 2 held-out 성패 분리 — AUC {auc:.3f}, 차이 {gap:+.4f}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
