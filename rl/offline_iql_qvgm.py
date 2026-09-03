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
from rl.nets import FuseProj, StepwiseEnsemble, StepwiseV, xavier_
from rl.offline_critic import normalize_all
from rl.vla_rldx import load_state_action_processor, normalize_actions, normalize_states

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="openarm_rim")
p.add_argument("--data", type=Path, required=True)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--features", default="cogfeat.npy",
               help="frozen VLM feature npy. Q-VGM 의 RL token 자리 (ablation: ResNet 이면 -5.1%%p)")
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--discount", type=float, default=0.995, help="프레임 단위 γ")
p.add_argument("--expectile", type=float, default=0.8,
               help="Q-VGM 논문 부록 값이 0.8 이다")
p.add_argument("--num-qs", type=int, default=2, help="Q-VGM 은 2 (clipped double Q)")
p.add_argument("--num-min-qs", type=int, default=0,
               help="REDQ: 타깃의 min 을 앙상블 전체가 아니라 매 스텝 무작위로 고른\n"
                    "이만큼의 부분집합에서 잡는다. 0 이면 전체 (= clipped double Q).\n"
                    "num_qs=10, num_min_qs=2 가 REDQ 기본값이다")
p.add_argument("--bins", type=int, default=128,
               help="분포형 critic (HL-Gauss) 의 bin 수. 0 이면 스칼라 MSE. "
                    "**stepwise 스칼라 헤드는 발산했다** (실측: 실패 Q 가 0 이어야 하는데 "
                    "29 까지 상승, AUC 0.38->0.20). IQL 의 V<-expectile(Q) 는 위로 편향되고 "
                    "Q 가 그 V 를 부트스트랩하므로 상한이 없으면 되먹임이 멈추지 않는다. "
                    "support 를 [0,1] 로 고정하면 Q^(i) 가 그 밖을 표현할 수 없다")
p.add_argument("--q-range", default="0,1",
               help="분포형 support. Q^(i) 는 종단 보상 1 짜리 감가 리턴이라 [0,1] 이 참값 범위다")
p.add_argument("--latent", type=int, default=512, help="feature projection 차원")
p.add_argument("--hidden", default="",
               help="critic/value MLP 폭. 쉼표 구분 (예 '1024,1024'). 비면 exp yaml 의 "
                    "expo.hidden_dims. **액션이 층마다 재주입되므로 폭이 중요하다** — "
                    "hidden 256 에 액션 625 면 2층 이후 입력의 71%%가 액션이라 latent 가 "
                    "묻힌다. Q-VGM 부록은 2층을 쓴다")
p.add_argument("--state-latent", type=int, default=64,
               help="state projection 차원. >0 이면 Q-VGM 방식으로 proj(feat) 과 "
                    "proj(state) 를 concat 한 뒤 LayerNorm 을 한 번 건다. "
                    "0 이면 예전 방식 — state 를 raw 로 붙인다 (ablation 용)")
p.add_argument("--no-inject", action="store_true", help="층마다 액션 재주입을 끈다 (ablation)")
p.add_argument("--action-groups", default="",
               help="critic 이 볼 action 그룹 (쉼표 구분, 예 'eef_position,eef_rotation'). "
                    "비면 전부. hammer_nail 에서 손가락 16관절을 빼면 창 25스텝 기준 "
                    "625 -> 225 차원이고, 창을 10스텝으로 줄이면 90차원이 된다. "
                    "액션 차원이 작을수록 critic 이 Q 의 액션 의존성을 배우기 쉽다 — "
                    "625차원에서는 후보간 Qstd 가 0.0001 로 사실상 0 이었다. "
                    "**서빙은 여전히 전 차원 액션을 넘기고 critic 이 내부에서 잘라 쓴다** "
                    "(체크포인트에 열 인덱스를 저장한다)")
p.add_argument("--no-stepwise", action="store_true",
               help="stepwise 헤드를 끄고 **청크 하나에 값 하나**를 쓴다 (offline_iql 과 같은 "
                    "n-step 리턴). 논문은 stepwise 를 쓰지만 Table 4 ablation 에 없어 "
                    "검증된 선택이 아니다. 그리고 우리 실측에서 stepwise 는 스케일이 깨진다: "
                    "헤드 20개가 각각 [0,1] 로 균등 초기화되어 Q_sum 이 10 에서 시작하는데 "
                    "청크 하나의 참값은 [0,1] 이다. 실패 종단 Q 도 0 으로 안 내려갔다 "
                    "(관측 2.6, 참값 0) — 범위 밖 헤드들이 0 회귀에 실패한다")
p.add_argument("--holdout", default="0.2",
               help="세션 이름 문자열 또는 에피소드 비율(0<x<1). 기본값을 offline_iql 과\n"
                    "같게 둔다 — 다르면 같은 --holdout 을 줘도 태그 접미사가 한쪽만 붙어\n"
                    "(기본값과 다를 때만 붙는다) 디렉토리 이름이 비대칭이 된다")
p.add_argument("--eval-frac", type=float, default=0.1,
               help="--holdout 0 (전 데이터 학습)일 때 진단용으로 골라 볼 에피소드 비율.\n"
                    "학습에서 빼지 않으므로 이 수치는 in-sample 이다 — 과적합 탐지에는\n"
                    "쓸 수 없고, Q 곡선 모양/앙상블 std/발산 여부를 보는 용도다")
p.add_argument("--train-eps", default="all", choices=("all", "success", "fail"))
p.add_argument("--plot-eps", default="",
               help="플롯에 그릴 홀드아웃 에피소드 (all/success/fail). 비면 --train-eps 를 "
                    "따른다 — 성공만 학습했으면 실패 곡선은 critic 이 본 적 없는 외삽이라 "
                    "같이 그리면 성공들 사이의 차이가 눈에 안 들어온다. "
                    "AUC 는 (가능하면) 홀드아웃 전체로 계속 계산한다")
p.add_argument("--eval-every", type=int, default=3000)
p.add_argument("--tag", default="")
p.add_argument("--keep-last", type=int, default=1)
p.add_argument("--keep-steps", default="",
               help="쉼표로 나열한 스텝의 체크포인트는 --keep-last 정리에서 지키다.\n예: --steps 200000 --keep-steps 1000,5000,20000,100000,200000 이면\n한 번의 학습으로 스텝 ablation 격자가 전부 나온다")
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
KEEP_STEPS = {int(x) for x in a.keep_steps.replace(",", " ").split() if x}
# 학습은 항상 스텝 0 에서 시작한다 (resume 없음). 그러므로 run 디렉토리에 남아 있는
# 체크포인트는 정의상 "다른 런" 의 것이다. 안 지우면 preempt 후 재시작(background 파티션은
# PreemptMode=REQUEUE)마다 --keep-steps 로 보호된 ckpt 가 쌓여 서로 다른 런이 한 디렉토리에
# 섞인다 — 실제로 그렇게 됐었다. 지우고 시작한다.
_stale = sorted(run.glob("critic_*.pt"))
if _stale:
    print(f"[정리] 이전 런의 체크포인트 {len(_stale)}개를 지운다 "
          f"({_stale[0].name} ... {_stale[-1].name})")
    for _f in _stale:
        _f.unlink()

# --- 1. 데이터 --------------------------------------------------------------
mod, _ = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
# images.mm 은 **이미지 판 critic 일 때만** 필요하다. --features 로 미리 뽑아둔
# cogfeat 을 쓰면 아래에서 npy 를 읽으므로 이 파일을 만들 이유가 없다 (실제로 이 함수의
# 반환값을 쓰는 곳이 없고 open_images 도 호출되지 않는다).
#   d5r20 기준 106GB / 비디오 디코딩 수십 분. 다른 머신으로 옮길 때도 이만큼이 빠진다.
# 정렬은 아래 `fp.shape[0] == len(flat)` assert 가 지킨다.
if not a.features:
    build_images(sessions, flat, work / "images.mm", mod)
else:
    print(f"[skip] images.mm 생략 — feature 판 critic ({a.features})")
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
A_DIM = mod.action_dim
# critic 에 넣을 액션 열. 평탄화된 (창 x 관절) 중에서 고른 그룹의 관절만 남긴다.
if a.action_groups:
    _g = [x.strip() for x in a.action_groups.split(",") if x.strip()]
    _jc = [j for nm, s0, e0 in mod.offsets("action") if nm in _g for j in range(s0, e0)]
    if not _jc:
        raise SystemExit(f"--action-groups {a.action_groups} 에 맞는 그룹이 없다 "
                         f"(가능: {[nm for nm, _, _ in mod.offsets('action')]})")
else:
    _jc = list(range(A_DIM))
AIDX = np.array([t * A_DIM + j for t in range(LAT + R) for j in _jc], dtype=np.int64)
FULL = len(AIDX)
AIDX_T = torch.from_numpy(AIDX).to(dev)
print(f"[액션] critic 입력 = 창 {LAT+R}스텝 x 관절 {len(_jc)}개 = {FULL}차원"
      + (f"  (그룹 {a.action_groups})" if a.action_groups else "  (전 관절)"))

# stepwise 보상/종료: (T, R).  **클램프한 인덱스를 그대로 쓰면 안 된다** — 에피소드 끝을
# 넘는 위치가 마지막 프레임을 반복하므로 성공 종단 보상 1 이 여러 위치에 복제된다.
# 범위 밖은 보상 0 / mask 0 / 손실 가중치 0 으로 죽인다.
_raw = np.arange(len(flat))[:, None] + np.arange(R)[None, :]
_inb = (_raw <= flat.ep_end[:, None]).astype(np.float32)          # (T,R) 에피소드 안인지
off = np.minimum(_raw, flat.ep_end[:, None])
RSTEP = torch.from_numpy((flat.reward[off] * _inb).astype(np.float32)).to(dev)
MSTEP = torch.from_numpy(((1.0 - flat.done[off]) * _inb).astype(np.float32)).to(dev)
NEXT = torch.from_numpy(np.minimum(np.arange(len(flat)) + R, flat.ep_end)).to(dev)
NST = 1 if a.no_stepwise else R                    # 값 헤드 개수 (청크 위치별 vs 청크 하나)
if a.no_stepwise:
    # 청크 전체를 하나의 n-step 전이로 접는다 (rl/data.py nstep 과 같은 식):
    #     G   = Σ_{k<R} γ^k (Π_{j<k} m_j) r_{t+k}
    #     boot= γ^R (Π_{j<R} m_j) · V(s')
    # 종료 이후 항은 mask 누적곱이 0 으로 만들고, 에피소드 밖은 _inb 가 이미 0 이다.
    # 이렇게 하면 Q 의 참값이 정확히 [0,1] 이라 분포형 support 와 딱 맞는다.
    _gp = (a.discount ** torch.arange(R, device=dev)).view(1, R)
    _prev = torch.cumprod(
        torch.cat([torch.ones(len(MSTEP), 1, device=dev), MSTEP[:, :-1]], 1), 1)
    GRET = (RSTEP * _gp * _prev).sum(-1, keepdim=True)              # (T,1)
    BOOT = (a.discount ** R) * (_prev[:, -1:] * MSTEP[:, -1:])      # (T,1)
    print(f"[타깃] 청크 n-step 리턴 (헤드 1개). γ^R={a.discount ** R:.5f}  "
          f"리턴>0 인 프레임 {int((GRET > 0).sum())}/{len(GRET)}")
else:
    print(f"[타깃] stepwise (헤드 {R}개, Q-VGM Eq.4) — Q 합의 참값 범위는 [0,{R}]")
# 범위 밖 위치는 reward 0 / mask 0 이라 타깃이 정확히 0 이다 → **손실 가중치를 0 으로 두지
# 않는다.** 0 으로 두면 그 헤드가 아무 값이나 내고 Σ_i Q^(i) 가 오염된다 (실측: 에피소드
# 마지막 프레임에서 Q_sum 이 3.04, 실제 리턴은 1). 0 으로 회귀시켜 합을 살린다.
# done=1 이 에피소드 마지막 프레임에 반드시 있으므로(실측 300/300) 경계 부트스트랩도
# mask 로 자동 차단된다 — 별도 valid 플래그가 필요 없다.

# --holdout 0 / none / off / "" → 홀드아웃 없음. **명시적으로 처리해야 한다**:
# 예전에는 이것들이 아래 세션 이름 부분일치 분기로 떨어졌는데, 세션 이름에 날짜가
# 들어 있어 "0" 이 거의 **모든** 세션에 매치돼 전 데이터가 홀드아웃이 되고 학습
# 집합이 비었다 (조용히 죽는 함정이었다).
_ho = a.holdout.strip().lower()
NOHOLD = _ho in ("", "0", "0.0", "none", "off")
frac = 0.0 if NOHOLD else (
    float(a.holdout) if a.holdout.replace(".", "", 1).isdigit() else 0.0)
_ep_ids = np.unique(flat.episode)
if NOHOLD:
    # 학습에서 아무것도 빼지 않는다. 그래도 진단(Q 곡선 / AUC / 앙상블 std)은 필요하므로
    # **학습 데이터에서** 평가 에피소드를 골라 쓴다 = in-sample 평가.
    # ★ in-sample 수치는 홀드아웃 수치보다 반드시 좋게 나온다. 과적합 탐지에는 못 쓴다.
    #   그래서 로그·플롯에 'in-sample' 을 붙여 오독을 막는다.
    hold = np.zeros(len(flat), bool)
    eval_ids = _ep_ids[:: max(2, int(round(1 / a.eval_frac)))]
    how = (f"없음 — 전 데이터 학습. 평가는 학습 에피소드 {len(eval_ids)}개를 "
           f"골라 **in-sample**")
elif 0 < frac < 1:
    every = max(2, int(round(1 / frac)))
    eval_ids = _ep_ids[::every]
    hold = np.isin(flat.episode, eval_ids)
    how = f"에피소드 {every}개마다 1개"
else:
    sel = [i for i, n in enumerate(flat.sessions) if a.holdout in n]
    if not sel:
        raise SystemExit(f"--holdout '{a.holdout}' 에 맞는 세션이 없다. 있는 세션:\n  "
                         + "\n  ".join(flat.sessions))
    hold = np.isin(flat.session, sel)
    eval_ids = np.unique(flat.episode[hold])
    how = f"세션 {[flat.sessions[i] for i in sel]}"
IN_SAMPLE = NOHOLD
train = np.flatnonzero(~hold[:len(flat) - R])
n_all = len(train)
if a.train_eps != "all":
    train = train[flat.is_success[train] == (a.train_eps == "success")]
eps = [(e, np.flatnonzero(flat.episode == e)) for e in eval_ids]
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


if a.state_latent > 0:
    enc = FuseProj(FEAT.shape[1], snorm.shape[1], a.latent, a.state_latent).to(dev)
    IN = enc.out_dim
else:
    enc = Proj(FEAT.shape[1], a.latent).to(dev)    # 예전 방식: state 를 raw 로 concat
    IN = a.latent + snorm.shape[1]
inject = not a.no_inject
HID = (tuple(int(x) for x in a.hidden.split(",") if x.strip())
       if a.hidden else tuple(cfg.hidden_dims))
critic = StepwiseEnsemble(IN, FULL, NST, a.num_qs, HID,
                          cfg.critic_layer_norm, inject, a.bins).to(dev)
value = StepwiseV(IN, NST, HID, cfg.critic_layer_norm).to(dev)  # V 는 스칼라 (offline_iql 과 동일)
target = StepwiseEnsemble(IN, FULL, NST, a.num_qs, HID,
                          cfg.critic_layer_norm, inject, a.bins).to(dev)

# --- 분포형 헤드 (HL-Gauss). offline_iql.py 와 같은 식이되 마지막 축이 위치별로 하나씩 있다.
if a.bins:
    lo_q, hi_q = (float(x) for x in a.q_range.split(","))
    edges = torch.linspace(lo_q, hi_q, a.bins + 1, device=dev)
    centers = (edges[:-1] + edges[1:]) / 2
    sigma = 0.75 * (hi_q - lo_q) / a.bins

    def q_of(x):                       # (..., R, bins) logits -> (..., R) 기대값
        return (x.softmax(-1) * centers).sum(-1)

    def q_loss_fn(x, y):               # x (num_qs,B,R,bins), y (B,R) -> CE
        xf = x.reshape(-1, a.bins)
        yf = y[None].expand(x.shape[0], *y.shape).reshape(-1)
        z = (edges - yf.clamp(lo_q, hi_q)[:, None]) / (sigma * 2 ** 0.5)
        cdf = 0.5 * (1 + torch.erf(z))
        pr = (cdf[:, 1:] - cdf[:, :-1])
        pr = pr / pr.sum(-1, keepdim=True).clamp_min(1e-8)
        return -(pr * xf.log_softmax(-1)).sum(-1).mean()
    print(f"[critic] distributional stepwise: support [{lo_q},{hi_q}] "
          f"bins {a.bins} sigma {sigma:.5f}  (Q^(i) 가 support 밖으로 못 나간다)")
else:
    q_of = lambda x: x
    q_loss_fn = lambda x, y: ((x - y[None]) ** 2).mean()
    print("[critic] 스칼라 MSE — **발산 이력 있음**, --bins 128 을 권장한다")
tenc = (FuseProj(FEAT.shape[1], snorm.shape[1], a.latent, a.state_latent)
        if a.state_latent > 0 else Proj(FEAT.shape[1], a.latent)).to(dev)
target.load_state_dict(critic.state_dict())
tenc.load_state_dict(enc.state_dict())
with torch.no_grad():                              # PA-RL kernel_scale_final=1e-2 와 같은 취지
    for m in list(critic.qs) + [value]:
        m.head.weight.mul_(1e-2)
        m.head.bias.zero_()
    target.load_state_dict(critic.state_dict())
opt = torch.optim.Adam(list(enc.parameters()) + list(critic.parameters())
                       + list(value.parameters()), lr=a.lr)
_in_desc = (f"FuseProj: LayerNorm(concat[proj(cog) {a.latent}, "
            f"proj(state) {a.state_latent}]) = {IN}" if a.state_latent > 0
            else f"Proj(cog) {a.latent} + state raw {snorm.shape[1]} = {IN}")
print(f"[입력] {_in_desc}")
print(f"[MLP] hidden {HID}  |  층마다 재주입 시 2층 이후 입력 = "
      f"{HID[0]} + 액션 {FULL} (액션 비중 {100*FULL/(HID[0]+FULL):.0f}%)")
print(f"[critic] stepwise Q x{a.num_qs} (min), 액션 층마다 재주입 {inject}, "
      f"학습 파라미터 {sum(p.numel() for g in opt.param_groups for p in g['params'])/1e6:.1f}M")


def lat_of(i, tgt=False):
    e = tenc if tgt else enc
    if a.state_latent > 0:
        return e(FEAT[i], SNORM[i])                # 합친 뒤 LayerNorm (Q-VGM)
    return torch.cat([e(FEAT[i]), SNORM[i]], -1)


def act_of(i):
    return NORM[i].reshape(len(i), -1)[:, AIDX_T]


# --- 3. 학습 ----------------------------------------------------------------
TRAIN = torch.from_numpy(train).to(dev)
# wandb 없이도 학습 경과를 남긴다 — 끝날 때 run/plots/training_curve.png 로 굽는다.
HIST = {"step": [], "q": [], "v": [], "qsum": [], "vsum": []}
EVH = {"step": [], "auc": [], "qs": [], "qf": []}

t0 = time.time()
for step in range(1, a.steps + 1):
    i = TRAIN[torch.randint(len(TRAIN), (a.batch,), device=dev)]
    j = NEXT[i]
    with torch.no_grad():
        # 청크 안 부트스트랩 (Q-VGM Eq. 4): 위치 i<R-1 은 **같은** 상태의 V^(i+1),
        # 마지막 위치만 다음 청크의 V^(0)(s') 로 넘어간다.
        if a.no_stepwise:
            tq = GRET[i] + BOOT[i] * value(lat_of(j))       # (B,1) 청크 n-step 리턴
        else:
            v_same = value(lat_of(i))                       # (B,R) = V^(0..R-1)(s)
            v_next = torch.cat([v_same[:, 1:], value(lat_of(j))[:, :1]], 1)
            tq = RSTEP[i] + a.discount * MSTEP[i] * v_next
        _mem = (None if a.num_min_qs <= 0 or a.num_min_qs >= a.num_qs
                else torch.randperm(a.num_qs, device=dev)[:a.num_min_qs].tolist())
        qt = q_of(target(lat_of(i, tgt=True), act_of(i), members=_mem)).min(0).values   # (B,R)

    lat = lat_of(i)
    ql = critic(lat, act_of(i))                  # bins 면 (num_qs,B,R,bins), 아니면 (num_qs,B,R)
    q = q_of(ql)                                             # (num_qs, B, R)
    v = value(lat)                                           # (B, R)
    q_loss = q_loss_fn(ql, tq)
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
        HIST["step"].append(step); HIST["q"].append(float(q_loss))
        HIST["v"].append(float(v_loss))
        HIST["qsum"].append(float(q.min(0).values.sum(-1).mean()))
        HIST["vsum"].append(float(v.sum(-1).mean()))
        print(f"  step {step:6d}  q {float(q_loss):.5f}  v {float(v_loss):.5f}  "
              f"Q_sum {float(q.min(0).values.sum(-1).mean()):+.3f}  "
              f"V_sum {float(v.sum(-1).mean()):+.3f}  {(time.time()-t0)/step*1000:.0f}ms/step",
              flush=True)

    # eps 가 비면(홀드아웃 없음) 아래 np.concatenate 가 ValueError 로 죽는다.
    if eps and (step % a.eval_every == 0 or step == a.steps):
        with torch.no_grad():
            curves = {}
            for e, fr, ok in eps:
                qs, vs = [], []
                for k in np.array_split(fr, max(1, len(fr) // 256)):
                    kk = torch.from_numpy(k).to(dev)
                    l = lat_of(kk)
                    qs.append(q_of(critic(l, act_of(kk))).min(0).values.sum(-1).float().cpu().numpy())
                    vs.append(value(l).sum(-1).float().cpu().numpy())
                curves[e] = (np.concatenate(qs), np.concatenate(vs))
        with torch.no_grad():
            _k = torch.as_tensor(
                np.concatenate([fr[::max(1, len(fr) // 8)] for _, fr, _ in eps])[:512],
                device=dev)
            ens_std_ref = float(q_of(critic(lat_of(_k), act_of(_k))).sum(-1).std(0).mean())
        _WH = "in-sample" if IN_SAMPLE else "홀드아웃"
        print(f"  [OOD 기준] {_WH} 로그 액션의 앙상블 std = {ens_std_ref:.4f}")
        fin = np.array([curves[e][0][-1] for e, _, _ in eps])
        okm = np.array([o for _, _, o in eps])
        # 한쪽이 비면 쌍이 없어 nan 이 된다 (전부 성공인 데이터셋). 정상이다.
        auc = (float((fin[okm][:, None] > fin[~okm][None, :]).mean())
               if okm.any() and (~okm).any() else float("nan"))
        EVH["step"].append(step); EVH["auc"].append(float(auc))
        EVH["qs"].append(float(fin[okm].mean())); EVH["qf"].append(float(fin[~okm].mean()))
        print(f"  [eval/{_WH}] step {step:6d}  AUC {auc:.3f}  "
              f"Q(성공끝) {fin[okm].mean():+.3f}  Q(실패끝) {fin[~okm].mean():+.3f}")
        if wb is not None:
            wb.log({"eval/auc": auc, "eval/q_end_success": float(fin[okm].mean()),
                    "eval/q_end_fail": float(fin[~okm].mean())}, step=step)
        # 그릴 에피소드. 성공만 학습했으면 기본으로 성공만 그린다.
        _pe = a.plot_eps or (a.train_eps if a.train_eps != "all" else "all")
        _peps = ([x for x in eps if x[2] == (_pe == "success")] if _pe in ("success", "fail")
                 else eps)
        fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for e, fr, ok in _peps:
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
        axs[0].set_title(f"Q-VGM critic  step {step}  AUC {auc:.3f}  "
                         + (f"{_pe} episodes only (n={len(_peps)})" if _pe != "all"
                            else "green=success"))
        fig.tight_layout()
        fig.savefig(plots / f"{step:06d}_qv.png", dpi=110)
        plt.close(fig)

        ck = run / f"critic_{step:06d}.pt"
        tmp = ck.with_suffix(".pt.tmp")
        torch.save({"enc": enc.state_dict(), "critic": critic.state_dict(),
                    "value": value.state_dict(), "target": target.state_dict(),
                    "tenc": tenc.state_dict(), "kind": "qvgm",
                    "step": step, "exp": a.exp, "seed": a.seed, "expectile": a.expectile,
                    "discount": a.discount, "num_qs": a.num_qs, "n_steps": NST,
                    "num_min_qs": a.num_min_qs,
                    "stepwise": not a.no_stepwise,
                    "latency": LAT, "replan": R, "action_dim": A_DIM, "inject": inject,
                    "state_dim": snorm.shape[1], "latent": a.latent,
                    "action_index": AIDX.tolist(), "action_groups": a.action_groups,
                    "state_latent": a.state_latent, "tag": TAG,
                    "hidden_dims": list(HID), "bins": a.bins,
                    "q_range": a.q_range if a.bins else None,
                    "critic_layer_norm": cfg.critic_layer_norm,
                    "features": a.features, "feat_mu": MU.cpu(), "feat_sd": SD.cpu(),
                    "ens_std_ref": ens_std_ref}, tmp)
        os.replace(tmp, ck)
        lnk, ltmp = run / "critic_latest.pt", run / "critic_latest.pt.tmp"
        ltmp.unlink(missing_ok=True)
        ltmp.symlink_to(ck.name)
        os.replace(ltmp, lnk)
        print(f"  [저장] {ck.name} ({ck.stat().st_size/1e6:.0f}MB)")
        if a.keep_last > 0:
            for f in sorted(run.glob("critic_[0-9]*.pt"))[:-a.keep_last]:
                if int(f.stem.split("_")[1]) not in KEEP_STEPS:
                    f.unlink()

if wb is not None:
    wb.finish()

# --- 학습 곡선. wandb 를 안 써도 나중에 무슨 일이 있었는지 볼 수 있게 남긴다 --------------
# 특히 (1) Q_sum 이 계속 올라가면 부트스트랩 발산, (2) AUC 는 평평한데 손실만 떨어지면
# 과적합 — 둘 다 숫자 로그를 훑는 것보다 그림에서 훨씬 빨리 보인다.
if HIST["step"]:
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(HIST["step"], HIST["q"], lw=1, label="q_loss")
    axs[0].plot(HIST["step"], HIST["v"], lw=1, label="v_loss")
    axs[0].set_yscale("log"); axs[0].set_ylabel("loss")
    axs[1].plot(HIST["step"], HIST["qsum"], lw=1, label="Q (min of ens.)")
    axs[1].plot(HIST["step"], HIST["vsum"], lw=1, label="V")
    axs[1].set_ylabel("train batch mean")
    if EVH["step"]:
        _wl = "in-sample" if IN_SAMPLE else "holdout"
        axs[2].plot(EVH["step"], EVH["auc"], "o-", lw=1.5, label=f"{_wl} AUC")
        axs[2].plot(EVH["step"], EVH["qs"], "s-", lw=1, label="Q(success end)")
        axs[2].plot(EVH["step"], EVH["qf"], "^-", lw=1, label="Q(fail end)")
        axs[2].axhline(0.5, color="gray", lw=.5)
    axs[2].set_ylabel("in-sample" if IN_SAMPLE else "holdout")
    axs[2].set_xlabel("step")
    for ax in axs:
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    axs[0].set_title(f"{TAG}  train_eps={a.train_eps}  action {FULL}d  "
                     f"({a.action_groups or 'all joints'})")
    fig.tight_layout()
    fig.savefig(plots / "training_curve.png", dpi=110)
    plt.close(fig)
    print(f"[학습곡선] {plots / 'training_curve.png'}")

print(f"\n[완료] {run}")
