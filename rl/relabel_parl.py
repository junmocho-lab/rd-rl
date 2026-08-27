#!/usr/bin/env python3
"""PA-RL 액션 최적화로 데이터셋의 action 컬럼을 relabel 한다 (오프라인 distillation 1단계).

PA-RL 의 distillation 은 손실을 바꾸지 않는다. 배치의 ``actions`` 만 최적화된 액션으로
갈아끼우고 원래 BC 손실을 그대로 쓴다 (third_party/PolicyAgnosticRL/train.py:249, 1122):

    batch["actions"] = action_distribution.sample(seed=rng)
    base_policy_agent.update(batch)

우리 데이터가 LeRobot parquet 이라 그 교체를 **디스크 레벨**에서 한다. 그러면 RLDX-1 의
학습 스택(RTC prefix 샘플링 / flow matching / 증강 / 체크포인트)을 한 줄도 건드리지 않고
``--action_model_use_lora`` 만 켜서 그대로 돌릴 수 있다.

프레임 정렬 — parquet 의 action 은 **프레임당 1스텝**이라 청크 타깃을 되돌려야 하는데,
RTC 실행 의미론이 그것을 하나로 결정한다:

    결정 프레임 t (replan 간격)에서 최적화 → 편집 창은 청크 [latency, latency+replan)
    → 로봇이 실제 실행하는 구간 → 전역 프레임 t+latency … t+latency+replan-1
    다음 결정은 t+replan → 프레임 t+latency+replan … 부터

  즉 latency 이후 모든 프레임이 **정확히 한 번** 덮인다. 겹침도 빈틈도 없다.
  에피소드 앞 latency 프레임만 원본 액션을 유지한다.

후보 구성은 EXPO 온라인 경로(rl/expo.py:225-231)와 같게 맞춘다:
  · base policy 에서 M개 청크를 뽑고 앞 (latency+replan) 스텝만 쓴다
  · latency prefix 블록은 **로그된 값으로 덮는다** — 실행이 이미 확정된 구간이다
  · 여기에 로그된 액션 자체를 후보로 하나 넣는다 (PA-RL action_optimization.py:435-455 의
    "skip the worst action" 안전장치). 오프라인에서는 롤아웃으로 회복할 수 없으니 이게 더
    중요하다 — 최악의 경우 원래 BC 로 수렴한다
  · ∇_a Q 상승은 explore_groups × 스텝 [latency, latency+replan) 에만 (spec.index)

usage:
  PYTHONPATH="$L_PYTHONPATH" $L_PY -u -m rl.relabel_parl \\
      --exp openarm_rim --data rl-dataset/0825_openarm_f1_inference \\
      --checkpoints checkpoints --features cogfeat.npy \\
      --critic iql-cog-dist128-t07-g0995-q10all-s0/critic_latest.pt \\
      --auto-step 0.02 --out rl-dataset/0825_openarm_f1_parl
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from rl.critic_io import load_critic
from rl.data import (action_chunk, build_flat, build_images, find_sessions, open_images,
                     resolve_modality)
from rl.expo import ExpoConfig
from rl.nets import explore_spec
from rl.offline_critic import normalize_all
from rl.vla_rldx import load_state_action_processor, normalize_states

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="openarm_rim")
p.add_argument("--data", type=Path, required=True)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--critic", required=True, help="work 디렉토리 기준 상대경로도 된다")
p.add_argument("--features", default="", help="cog feature critic 이면 npy 이름")
p.add_argument("--model-path", default="", help="후보를 뽑을 base policy (기본: exp yaml)")
p.add_argument("--out", type=Path, help="출력 데이터셋 (기본: <data>-parl)")
p.add_argument("--groups", default="", help="편집할 action 그룹 (기본: exp yaml 의 explore_groups)")
p.add_argument("--num-samples", type=int, default=32, help="PA-RL num_base_policy_actions")
p.add_argument("--num-keep", type=int, default=10, help="PA-RL num_actions_to_keep")
p.add_argument("--num-steps", type=int, default=10, help="PA-RL local optimization steps")
p.add_argument("--step-size", type=float, default=3e-4, help="PA-RL 기본값")
p.add_argument("--auto-step", type=float, default=0.0,
               help="차원당 목표 이동거리 D. 주면 ‖g‖ 를 재서 step_size 를 잡는다 "
                    "(PA-RL 의 3e-4 는 우리 Q 스케일에서 사실상 아무 일도 안 한다)")
p.add_argument("--temp", type=float, default=0.0,
               help="0 이면 argmax. >0 이면 Categorical(Q/temp) 로 샘플 (PA-RL 기본은 "
                    "logits=Q 그대로지만 우리 Q 는 [0,1] 이라 거의 균등해진다)")
p.add_argument("--batch", type=int, default=8, help="한 번에 처리할 결정 프레임 수")
p.add_argument("--limit", type=int, default=0, help="결정 프레임 수 제한 (디버그)")
p.add_argument("--dry-run", action="store_true", help="아무것도 쓰지 않고 통계만 낸다")
p.add_argument("--no-parquet", action="store_true",
               help="npy 만 쓰고 데이터셋 사본은 만들지 않는다 (rl.train_policy 로 학습할 때)")
p.add_argument("--npy-out", default="parl_actions.npy",
               help="relabel 된 raw 액션 (T, action_dim). work 디렉토리에 저장")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda")
a = p.parse_args()
dev = a.device
gen = torch.Generator(device=dev).manual_seed(a.seed)

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
cfg = ExpoConfig.from_dict(exp.get("expo"))
R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / (a.model_path or exp["base_policy"])
groups = [g.strip() for g in a.groups.split(",") if g.strip()] or list(exp["explore_groups"])
out_root = a.out or a.data.parent / f"{a.data.name}-parl"

# --- 1. 데이터 / critic ------------------------------------------------------
mod, _ = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
build_images(sessions, flat, work / "images.mm", mod)
imgs, meta = open_images(work / "images.mm")
proc = load_state_action_processor(base, RLDX, exp["rldx_data_config"])
snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
A_DIM = mod.action_dim
FULL, PRE = (LAT + R) * A_DIM, LAT * A_DIM
tasks = json.loads((sessions[0] / "meta/tasks.jsonl").read_text().splitlines()[0])
task = tasks["task"]

ck = Path(a.critic)
if not ck.is_file():
    ck = work / a.critic
if not ck.is_file():
    raise SystemExit(f"체크포인트가 없다: {a.critic}  (work={work})")
C = load_critic(ck, work, cfg, mod.n_cams, FULL, snorm.shape[1],
                features=a.features, imgs=imgs, dev=dev)

spec = explore_spec(mod.offsets("action"), groups, A_DIM, R, LAT)
MASK = torch.zeros(FULL, device=dev)
MASK[spec.index] = 1.0
NIDX = len(spec.index)

# --- 2. 결정 프레임 ----------------------------------------------------------
# 에피소드 안에서 replan 간격으로 t 를 놓고, 각 t 가 프레임 t+LAT … t+LAT+R-1 을 쓴다.
# 에피소드 끝을 넘는 쓰기 대상은 버린다 (원본 액션 유지).
dec, span = [], []
for e in np.unique(flat.episode):
    fr = np.flatnonzero(flat.episode == e)
    for t in range(fr[0], fr[-1] + 1, R):
        w = np.arange(t + LAT, min(t + LAT + R, fr[-1] + 1))
        if len(w) == 0:
            continue
        dec.append(t)
        span.append(w)
dec = np.asarray(dec, np.int64)
if a.limit:
    dec, span = dec[:a.limit], span[:a.limit]
covered = int(sum(len(w) for w in span))
print(f"[데이터] 프레임 {len(flat)} / 에피소드 {len(np.unique(flat.episode))} / 세션 {len(sessions)}")
print(f"[결정 프레임] {len(dec)}개 (replan {R} 간격) → 덮는 프레임 {covered} "
      f"({covered/len(flat):.1%}, 앞 {LAT}프레임 x 에피소드는 원본 유지)")
print(f"[편집 범위] {groups} → 액션 {FULL}차원 중 {NIDX}개 "
      f"({spec.active_dim}관절 x {R}스텝, prefix {LAT}스텝 제외)")

norm = normalize_all(None, flat, H, cache=work / "actnorm.npy") \
    if (work / "actnorm.npy").is_file() else None
if norm is None:
    raise SystemExit(f"actnorm.npy 가 없다: {work} — offline_iql 을 한 번 돌려 캐시를 만들어라")

# --- 3. base policy ----------------------------------------------------------
from rl.vla_rldx import RLDXVLA                      # noqa: E402  (무거운 import 를 뒤로)

vla = RLDXVLA(base, mod, RLDX, exp["rldx_data_config"], device=dev)
print(f"[base policy] {base.name}  horizon {vla.action_horizon}  M={a.num_samples} K={a.num_keep}")


def vla_obs(idx):
    x = np.asarray(imgs[idx])
    return {"video": {name: x[:, c][:, None] for c, (name, _) in enumerate(mod.video)},
            "state": {name: flat.state[idx][:, None, s:e] for name, s, e in mod.offsets("state")},
            "language": {mod.task_key: [[task]] * len(idx)}}


def candidates(idx):
    """(B, K, FULL) 후보. base 샘플 M개 → Q 상위 K-1개 + 로그된 액션 1개."""
    b = len(idx)
    with torch.no_grad():
        ch = vla.sample(vla_obs(idx), a.num_samples)              # (B,M,H,A) 모델 공간
    acts = ch[:, :, :LAT + R].reshape(b, a.num_samples, FULL).float()
    logged = torch.from_numpy(np.ascontiguousarray(
        np.asarray(norm[idx])[:, :LAT + R].reshape(b, -1))).to(dev)
    if PRE:
        acts[:, :, :PRE] = logged[:, None, :PRE]     # prefix 는 이미 커밋된 값 (expo.py:231)
    st = torch.from_numpy(snorm[idx]).to(dev)
    lat = C.latent_of(idx, st)
    with torch.no_grad():
        q = C.q(lat.repeat_interleave(a.num_samples, 0),
                st.repeat_interleave(a.num_samples, 0),
                acts.reshape(b * a.num_samples, FULL)).mean(0)     # PA-RL: 앙상블 mean
    top = q.view(b, a.num_samples).topk(min(a.num_keep - 1, a.num_samples), dim=1).indices
    keep = torch.gather(acts, 1, top[..., None].expand(-1, -1, FULL))
    return torch.cat([keep, logged[:, None, :]], 1), lat, st, logged


def ascend(cand, lat, st):
    """(B,K,FULL) 후보를 ∇_a Q 로 올린 뒤 Q 와 함께 반환. 마지막 ‖g‖ 도 같이."""
    b, K = cand.shape[:2]
    rl_, rs = lat.repeat_interleave(K, 0), st.repeat_interleave(K, 0)
    act = cand.reshape(b * K, FULL)
    g_last = torch.zeros_like(act)
    for _ in range(a.num_steps):
        act = act.detach().requires_grad_(True)
        qm = C.q(rl_, rs, act).mean(0).sum()
        g, = torch.autograd.grad(qm, act)
        g_last = g
        act = (act + a.step_size * g * MASK).clamp(-1.0, 1.0)
    with torch.no_grad():
        q = C.q(rl_, rs, act.detach()).mean(0).view(b, K)
    return act.detach().view(b, K, FULL), q, (g_last * MASK)[:, spec.index].norm(dim=-1)


# --- step_size 캘리브레이션 (probe_actopt 와 같은 규칙) ----------------------
if a.auto_step:
    k = dec[:: max(1, len(dec) // 32)][:32]
    st = torch.from_numpy(snorm[k]).to(dev)
    lat = C.latent_of(k, st)
    act0 = torch.from_numpy(np.ascontiguousarray(
        np.asarray(norm[k])[:, :LAT + R].reshape(len(k), -1))).to(dev).requires_grad_(True)
    qm = C.q(lat, st, act0).mean(0).sum()
    g, = torch.autograd.grad(qm, act0)
    gmed = float(((g * MASK)[:, spec.index].norm(dim=-1) / NIDX ** 0.5).median())
    a.step_size = a.auto_step / max(a.num_steps * gmed, 1e-12)
    print(f"[auto-step] median‖g‖/√d = {gmed:.3e} → step_size {a.step_size:.4g} "
          f"(목표 이동 {a.auto_step}/차원, PA-RL 기본값의 {a.step_size/3e-4:.0f}배)")
print(f"[상승] num_steps {a.num_steps}, step_size {a.step_size:.4g}, "
      f"선택 {'argmax' if a.temp <= 0 else f'Categorical(Q/{a.temp})'}")

# --- 4. relabel --------------------------------------------------------------
NEW = flat.action.copy()                              # (T, A) raw 공간, 원본에서 출발
stat = {"dq": [], "d": [], "won_logged": 0, "n": 0, "g": []}
t0 = time.time()
for c in range(0, len(dec), a.batch):
    idx = dec[c:c + a.batch]
    cand, lat, st, logged = candidates(idx)
    opt, q, gn = ascend(cand, lat, st)
    K = cand.shape[1]
    if a.temp <= 0:
        pick = q.argmax(1)
    else:
        pick = torch.distributions.Categorical(logits=q / a.temp).sample()
    chosen = torch.gather(opt, 1, pick[:, None, None].expand(-1, 1, FULL))[:, 0]
    with torch.no_grad():
        q_log = C.q(lat, st, logged).mean(0)
        q_new = C.q(lat, st, chosen).mean(0)
    stat["dq"].append((q_new - q_log).cpu().numpy())
    stat["d"].append((chosen - logged)[:, spec.index].norm(dim=-1).cpu().numpy() / NIDX ** 0.5)
    stat["g"].append((gn / NIDX ** 0.5).cpu().numpy())
    stat["won_logged"] += int((pick == K - 1).sum())   # 로그된 액션이 이긴 횟수
    stat["n"] += len(idx)

    # 모델 공간 → raw 공간. 기준 state 는 정규화 때와 같은 그 프레임의 state.
    blk = chosen.view(len(idx), LAT + R, A_DIM).cpu().numpy()
    raw = vla.denormalize_actions(blk, flat.state[idx])            # (B, LAT+R, A)
    for j, t in enumerate(idx):
        w = span[c + j]
        NEW[w] = raw[j, LAT:LAT + len(w)]
    if (c // a.batch) % 20 == 0:
        el = time.time() - t0
        done = c + len(idx)
        print(f"  {done}/{len(dec)}  {el:.0f}s  ({done/max(el,1e-9):.1f} 결정/s, "
              f"남은 {(len(dec)-done)/max(done/max(el,1e-9),1e-9)/60:.0f}분)  "
              f"ΔQ {np.concatenate(stat['dq']).mean():+.4f}  "
              f"이동 {np.concatenate(stat['d']).mean():.4f}", flush=True)

dq, dd, gg = (np.concatenate(stat[k]) for k in ("dq", "d", "g"))
print(f"\n[결과] 결정 {stat['n']}개")
print(f"  ΔQ (선택 - 로그)  평균 {dq.mean():+.4f}  중앙 {np.median(dq):+.4f}  "
      f"p95 {np.percentile(dq,95):+.4f}  개선된 비율 {(dq>0).mean():.1%}")
print(f"  이동거리/차원      평균 {dd.mean():.4f}  중앙 {np.median(dd):.4f}  "
      f"p95 {np.percentile(dd,95):.4f}   (액션 공간 ±1)")
print(f"  ‖g‖/√d            중앙 {np.median(gg):.3e}")
print(f"  로그된 액션이 이긴 비율 {stat['won_logged']/max(stat['n'],1):.1%}  "
      f"(높으면 critic 이 base policy 를 개선하지 못한다는 뜻)")
chg = np.abs(NEW - flat.action).max(1)
print(f"  raw 액션이 바뀐 프레임 {(chg>1e-6).sum()}/{len(flat)} "
      f"({(chg>1e-6).mean():.1%})  최대 변화 {chg.max():.4f}")

# --- 5a. relabel 결과를 npy 로 --------------------------------------------
# rl/train_policy.py --method parl 이 이걸 읽는다 (데이터셋 사본이 없어도 학습된다).
np.save(work / a.npy_out, NEW.astype(np.float32))
print(f"\n[출력] {work / a.npy_out}  ({NEW.nbytes/1e6:.0f}MB, raw 액션 (T,A))")

# --- 5b. 데이터셋 사본 (RLDX-1 자체 트레이너로 돌릴 때만 필요) -----------------
if a.dry_run or a.no_parquet:
    print("[skip] parquet 사본을 만들지 않았다 (--dry-run / --no-parquet).")
    raise SystemExit(0)

import pandas as pd                                  # noqa: E402

out_root.mkdir(parents=True, exist_ok=True)
n_files = 0
for si, sess in enumerate(sessions):
    dst = out_root / sess.name
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("meta", "videos"):                    # 비디오는 심링크 — 24GB 를 다시 안 쓴다
        d = dst / sub
        if not d.exists():
            if sub == "meta":
                shutil.copytree(sess / sub, d)
            else:
                d.symlink_to((sess / sub).resolve(), target_is_directory=True)
    for src in sorted((sess / "data").rglob("*.parquet")):
        df = pd.read_parquet(src)
        # 세션 안 episode_index → 전역 에피소드 번호 (rl/data.py:261 의 규약).
        gep = int(df["episode_index"].iloc[0]) + flat.ep_offset[si]
        sel = np.flatnonzero((flat.session == si) & (flat.episode == gep))
        if len(sel) != len(df) or not np.array_equal(flat.frame[sel], np.arange(len(df))):
            raise SystemExit(f"프레임 정렬 실패: {src} (parquet {len(df)}, flat {len(sel)})")
        # canonical concat 순서(neck, left_arm, right_arm, left_hand, right_hand) -> 원본
        # 컬럼 내부 순서로 되돌린다. mod.action 의 (key, s, e) 가 원본 슬라이스다.
        # 이걸 빼먹으면 관절이 조용히 뒤섞인다 (openarm 은 right_arm 과 left_hand 가 바뀐다).
        raw = {}
        for name, key, s0, e0 in mod.action:
            if key not in raw:
                raw[key] = np.stack(df[key].to_numpy()).astype(np.float32)
        cum = 0
        for name, key, s0, e0 in mod.action:
            w = e0 - s0
            raw[key][:, s0:e0] = NEW[sel][:, cum:cum + w]
            cum += w
        assert cum == NEW.shape[1], f"canonical 폭 {cum} != {NEW.shape[1]}"
        for key, arr in raw.items():
            df[key] = list(arr)
            # 되돌린 결과를 다시 gather 하면 원래 값이 나와야 한다 (순열 검증)
        chk = np.concatenate([raw[key][:, s0:e0] for _, key, s0, e0 in mod.action], 1)
        assert np.allclose(chk, NEW[sel]), f"순열 왕복 실패: {src}"
        out = dst / src.relative_to(sess)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        n_files += 1
print(f"\n[출력] {out_root}  (parquet {n_files}개 재작성, meta 복사, videos 심링크)")
print(f"[다음] RLDX-1 finetune 에 --action_model_use_lora 를 켜고 이 경로를 데이터로 준다. "
      f"PARL-DISTILL.md 참고")
