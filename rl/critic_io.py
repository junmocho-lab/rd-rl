#!/usr/bin/env python3
"""학습된 critic 체크포인트를 복원한다 — offline_iql / offline_critic_0 산출물 공용.

인코더가 두 종류라 다운스트림(probe_actopt, relabel_parl)이 매번 분기를 재작성하고 있었다.
여기 한 곳에 모은다.

  · ResNet 인코더 (기본)      : latent = BatchEncoder(카메라 concat 이미지)
  · frozen VLM feature (--features): latent = LayerNorm(Linear(cogfeat)) — 이미지를 아예 안 쓴다

feature 판은 학습 때 데이터셋 전체 평균/표준편차로 표준화했으므로, 그 통계를 그대로 써야
Q 가 학습 시점과 같은 값을 낸다. 체크포인트에 있으면 그것을 쓰고, 없으면(구버전) cogfeat.npy
에서 다시 계산한다 — 같은 파일이면 결정적으로 같은 값이 나온다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rl.nets import (BatchEncoder, CriticEnsemble, FuseProj, StepwiseEnsemble,
                     StepwiseV, xavier_)


class Proj(nn.Module):
    """frozen feature -> critic latent. offline_iql 의 내부 클래스와 같은 구조."""

    def __init__(self, din, dout):
        super().__init__()
        self.lin = xavier_(nn.Linear(din, dout))
        self.ln = nn.LayerNorm(dout)

    def forward(self, x, stop_gradient=False):
        z = self.ln(self.lin(x))
        return z.detach() if stop_gradient else z


class Critic:
    """enc/critic/value + q_of + latent_of(idx) 를 묶어 둔 것."""

    def __init__(self, enc, critic, value, q_of, latent_of, state_of, bins, meta):
        self.enc, self.critic, self.value = enc, critic, value
        self.q_of, self.latent_of, self.state_of = q_of, latent_of, state_of
        self.bins, self.meta = bins, meta

    def q(self, lat, st, act):
        """(num_qs, B) 스칼라 Q. 분포형이면 bin 기댓값으로 환산한 값."""
        return self.q_of(self.critic(lat, st, act))

    def v(self, lat, st):
        if self.value is None:
            return torch.zeros(len(lat), device=lat.device)
        return self.value(lat, st, torch.zeros(len(lat), 0, device=lat.device))[0]


def load_critic(ckpt: Path, work: Path, cfg, n_cams: int, action_full: int, state_dim: int,
                features: str = "", imgs=None, feat_path: Path | None = None,
                dev: str = "cuda") -> Critic:
    """ckpt 를 읽어 Critic 을 만든다.

    features 가 비어 있지 않으면 그 이름의 npy (기본 work/<features>) 를 mmap 으로 열어
    latent_of(idx) 가 feature 를 쓴다. 비어 있으면 imgs (open_images 결과) 로 이미지를 쓴다.
    ckpt 에 'features' 가 기록돼 있으면 인자가 비어 있어도 그것을 따른다.
    """
    sd = torch.load(ckpt, map_location=dev)
    feat_name = features or sd.get("features") or ""
    if features and sd.get("features") and features != sd["features"]:
        print(f"[경고] ckpt 는 '{sd['features']}' 로 학습됐는데 '{features}' 를 준다")
    bins = sd.get("bins") or 0

    if feat_name:
        fp = np.load(feat_path or (work / feat_name), mmap_mode="r")
        FEAT = torch.from_numpy(np.ascontiguousarray(np.asarray(fp))).to(dev)
        if sd.get("feat_mu") is not None:
            mu, sdv = sd["feat_mu"].to(dev), sd["feat_sd"].to(dev)
        else:                                      # 구버전 ckpt — 같은 파일에서 다시 계산
            mu = FEAT.mean(0, keepdim=True)
            sdv = FEAT.std(0, keepdim=True).clamp_min(1e-3)
            print("[feature] ckpt 에 정규화 통계가 없다 → npy 에서 재계산")
        FEAT = (FEAT - mu) / sdv
        enc = Proj(FEAT.shape[1], cfg.latent_dim_image).to(dev).eval()
        latent = cfg.latent_dim_image + state_dim
        incl_state = False
        enc_in = lambda i: FEAT[torch.as_tensor(i, device=dev)]
        print(f"[feature] {feat_name} {tuple(fp.shape)} -> VRAM {FEAT.numel()*4/1e9:.2f}GB")
    else:
        assert imgs is not None, "이미지 판 critic 인데 imgs 를 안 줬다"
        enc = BatchEncoder(3 * n_cams, cfg.latent_dim_image, cfg.encoder_stage_sizes,
                           cfg.encoder_num_filters).to(dev).eval()
        latent, incl_state = cfg.latent_dim_image, cfg.include_state

        def enc_in(i):
            x = np.asarray(imgs[i])
            return torch.from_numpy(np.ascontiguousarray(
                np.concatenate([x[:, c] for c in range(x.shape[1])], -1))).to(dev)

    critic = CriticEnsemble(latent, state_dim, action_full, sd.get("num_qs", cfg.num_qs),
                            cfg.latent_dim_state, incl_state, cfg.hidden_dims,
                            cfg.critic_layer_norm).to(dev).eval()
    value = CriticEnsemble(latent, state_dim, 0, 1, cfg.latent_dim_state, incl_state,
                           cfg.hidden_dims, cfg.critic_layer_norm).to(dev).eval()
    if bins:
        for m in critic.qs:
            m.head = xavier_(nn.Linear(m.body.out_dim, bins)).to(dev)
        lo_q, hi_q = (float(x) for x in (sd.get("q_range") or "0,1").split(","))
        edges = torch.linspace(lo_q, hi_q, bins + 1, device=dev)
        centers = (edges[:-1] + edges[1:]) / 2
        q_of = lambda x: (x.softmax(-1) * centers).sum(-1)
    else:
        q_of = lambda x: x
    enc.load_state_dict(sd["enc"])
    critic.load_state_dict(sd["critic"])
    if "value" in sd:
        value.load_state_dict(sd["value"])
    else:
        value = None

    SN = None                                        # state 는 호출측이 넘겨준 배열을 쓴다

    def latent_of(i, state):
        """critic 이 먹는 latent. feature 판은 state 를 raw 로 concat 한다."""
        z = enc(enc_in(i), stop_gradient=True)
        return torch.cat([z, state], -1) if feat_name else z

    print(f"[critic] {ckpt}\n          step {sd.get('step')}  γ={sd.get('discount')}  "
          f"τ={sd.get('expectile')}  bins={bins}  num_qs={sd.get('num_qs')}  "
          f"features={feat_name or '(이미지)'}")
    return Critic(enc, critic, value, q_of, latent_of, SN, bins, sd)


class StepwiseCritic:
    """offline_iql_qvgm.py 산출물. Q^(i) 합을 스칼라 Q 로 준다 (Q-VGM 4.1)."""

    def __init__(self, enc, critic, value, feat, mu, sd, snorm, meta, dev, act_index=None):
        self.enc, self.critic, self.value = enc, critic, value
        self.FEAT, self.SNORM, self.meta, self.dev = feat, snorm, meta, dev
        # critic 이 액션의 일부 열만 보도록 학습된 경우(--action-groups), 호출측은 전 차원
        # 액션을 넘기고 여기서 자른다. 이미 잘린 액션이 들어오면 그대로 쓴다.
        self.act_index = act_index

    def latent_of(self, i, state=None):
        idx = torch.as_tensor(i, device=self.dev)
        st = self.SNORM[idx] if state is None else state
        if int(self.meta.get("state_latent") or 0) > 0:
            return self.enc(self.FEAT[idx], st)    # 합친 뒤 LayerNorm
        return torch.cat([self.enc(self.FEAT[idx]), st], -1)

    def q_all(self, lat, act):
        """(num_qs, B, n_steps). 분포형이면 bin 기대값까지 적용한 뒤 돌려준다.

        앙상블 축을 남긴다 — 호출측이 min(신뢰 하한)과 std(OOD 판정)를 직접 잡는다.
        """
        if self.act_index is not None and act.shape[-1] != len(self.act_index):
            act = act[..., self.act_index]
        x = self.critic(lat, act)
        b = int(self.meta.get("bins") or 0)
        if b:
            lo, hi = (float(v) for v in self.meta["q_range"].split(","))
            e = torch.linspace(lo, hi, b + 1, device=x.device)
            x = (x.softmax(-1) * ((e[:-1] + e[1:]) / 2)).sum(-1)
        return x

    def q_steps(self, lat, act):
        """(num_qs, B, n_steps) → min → (B, n_steps)."""
        return self.q_all(lat, act).min(0).values

    def q(self, lat, act):
        """스칼라 점수 Q(s,A) = Σ_i Q^(i). ∇_A Q 는 이 합에서 받는다."""
        return self.q_steps(lat, act).sum(-1)

    def v(self, lat):
        return self.value(lat).sum(-1)


def load_stepwise_critic(ckpt: Path, work: Path, snorm, dev: str = "cuda") -> StepwiseCritic:
    """offline_iql_qvgm.py 체크포인트를 복원한다."""
    sd = torch.load(ckpt, map_location=dev)
    assert sd.get("kind") == "qvgm", f"qvgm critic 이 아니다 (kind={sd.get('kind')})"
    fp = np.load(work / sd["features"], mmap_mode="r")
    FEAT = torch.from_numpy(np.ascontiguousarray(np.asarray(fp))).to(dev)
    FEAT = (FEAT - sd["feat_mu"].to(dev)) / sd["feat_sd"].to(dev)
    hid = tuple(sd["hidden_dims"])
    ln = bool(sd["critic_layer_norm"])
    slat = int(sd.get("state_latent") or 0)
    if slat > 0:                                   # Q-VGM 입력 융합 (학습과 같아야 한다)
        enc = FuseProj(FEAT.shape[1], sd["state_dim"], sd["latent"], slat).to(dev).eval()
        IN = enc.out_dim
    else:
        enc = Proj(FEAT.shape[1], sd["latent"]).to(dev).eval()
        IN = sd["latent"] + sd["state_dim"]
    # 액션 열 인덱스가 있으면 critic 은 그 열만 본다 (--action-groups 로 학습한 경우).
    aidx = sd.get("action_index")
    full = len(aidx) if aidx else (sd["latency"] + sd["replan"]) * sd["action_dim"]
    critic = StepwiseEnsemble(IN, full, sd["n_steps"], sd["num_qs"], hid, ln,
                              sd.get("inject", True), int(sd.get("bins") or 0)).to(dev).eval()
    value = StepwiseV(IN, sd["n_steps"], hid, ln).to(dev).eval()
    enc.load_state_dict(sd["enc"])
    critic.load_state_dict(sd["critic"])
    value.load_state_dict(sd["value"])
    SN = torch.from_numpy(snorm).to(dev)
    AIDX = torch.as_tensor(aidx, device=dev) if aidx else None
    print(f"[critic] {ckpt}\n          step {sd.get('step')}  γ={sd.get('discount')}  "
          f"τ={sd.get('expectile')}  stepwise {sd['n_steps']}  q x{sd['num_qs']}  "
          f"inject={sd.get('inject')}  features={sd['features']}")
    return StepwiseCritic(enc, critic, value, FEAT, sd["feat_mu"], sd["feat_sd"], SN, sd, dev,
                          act_index=AIDX)


# --------------------------------------------------------------------------- #
# 서빙용 로더 — cogfeat.npy 없이 체크포인트만으로 critic 을 세운다
#
# 학습 경로는 cogfeat.npy 를 인덱스로 읽지만 서빙에는 그 파일이 없다 (실시간 관측이고,
# actor 머신에 1GB 를 둘 이유도 없다). 대신 백본이 액션 생성하면서 이미 계산한
# backbone_features 에서 cognition token 을 그 자리에서 mean-pool 한다.
# --------------------------------------------------------------------------- #
class ServingCritic:
    """추론 시점에 cog feature 로부터 critic latent 와 Q 를 만든다.

    학습 때의 latent 와 **비트 수준으로 같아야** 한다:
        latent = concat[ LayerNorm(Linear(cog_mean)) , state_raw ]
    표준화(feat_mu/feat_sd)와 Proj 가중치를 체크포인트에서 그대로 가져오므로, 같은
    cog feature 를 넣으면 같은 Q 가 나온다. rl.vla_rldx verify-cog 가 그것을 대조한다.
    """

    def __init__(self, enc, critic, target, q_of, mu, sd, n_cog, meta, dev,
                 fuse: bool = False, stepwise: bool = False, act_index=None):
        self.enc, self.critic, self.target, self.q_of = enc, critic, target, q_of
        self.mu, self.sd, self.n_cog, self.meta, self.dev = mu, sd, n_cog, meta, dev
        # offline_iql_qvgm 산출물이면 배선이 다르다:
        #   fuse     : latent = LayerNorm(concat[proj(cog), proj(state)])  (FuseProj)
        #              — state 가 latent 안에 이미 들어간다
        #   stepwise : critic(lat, act) 가 (num_qs, B, n_steps[, bins]) 를 내므로
        #              위치 축을 더해 스칼라 Q 로 만든다 (--no-stepwise 면 n_steps=1)
        self.fuse, self.stepwise = fuse, stepwise
        # --action-groups 로 학습한 critic 은 액션의 일부 열만 본다. 호출측(ExpoServer)은
        # 전 차원 액션을 넘기므로 여기서 자른다.
        self.act_index = act_index

    def cog_of(self, backbone_features: torch.Tensor) -> torch.Tensor:
        """(B, seq, d) 백본 출력 → (B, d) cog mean-pool.

        extract_cogfeat.py 와 **같은 규칙**이어야 한다: 마지막 n_cog 토큰의 평균, fp32.
        """
        return backbone_features[:, -self.n_cog:, :].float().mean(1)

    def latent(self, cog: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        z = (cog - self.mu) / self.sd
        if self.fuse:
            return self.enc(z, state)                  # 합친 뒤 LayerNorm
        return torch.cat([self.enc(z), state], -1)

    def q(self, lat, state, act, target: bool = False):
        """(num_qs, B) 스칼라 Q. 분포형이면 bin 기댓값, stepwise 면 위치 축 합."""
        m = self.target if (target and self.target is not None) else self.critic
        if self.act_index is not None and act.shape[-1] != len(self.act_index):
            act = act[..., self.act_index]
        if self.stepwise:
            return self.q_of(m(lat, act)).sum(-1)      # state 는 lat 안에 있다
        return self.q_of(m(lat, state, act))


def load_serving_critic(ckpt: Path, cfg, state_dim: int, action_dim: int, exec_off: int,
                        replan: int, n_cog: int, dev: str = "cuda") -> ServingCritic:
    """cogfeat.npy 를 읽지 않고 체크포인트만으로 ServingCritic 을 만든다.

    action_full 은 인자로 받지 않는다 — **체크포인트가 정한다**. exec_off/replan 은
    실행 구간이 critic 창 안에 들어가는지 확인하는 데만 쓴다.
    """
    sd = torch.load(ckpt, map_location=dev)
    feat = sd.get("features") or ""
    if not feat:
        raise SystemExit(
            f"{ckpt} 는 cog feature critic 이 아니다 (features 키가 비어 있다).\n"
            f"  이미지(ResNet) critic 은 기존 경로를 그대로 쓴다 — 이 로더가 필요 없다.")
    if sd.get("feat_mu") is None:
        raise SystemExit(
            f"{ckpt} 에 feat_mu/feat_sd 가 없다 (구버전 체크포인트).\n"
            f"  표준화 통계 없이는 학습 때의 latent 를 재현할 수 없다 — critic 을 다시 학습하거나\n"
            f"  cogfeat.npy 에서 통계를 계산해 넣어야 한다.")
    # 체크포인트에 학습 때의 latency/replan/action_dim/state_dim 이 기록돼 있다.
    # 서버가 yaml 에서 계산한 값과 다르면 **여기서 잡아야 한다** — 안 그러면 torch 의
    # size mismatch 로만 드러나서 "학습이 깨졌나" 로 오해하게 된다. 실제로 actor 클론의
    # inference_latency 가 0 (learner 는 2) 이라 820 vs 764 로 어긋난 적이 있다.
    # critic 의 액션 창은 **체크포인트가 고정한다** — 학습 때 본 청크 스텝 수다.
    # 실행 오프셋(rrc 의 inference_latency_steps)과는 별개다: 그건 rrc 설정을 베끼는 값이고
    # 학습 때와 달라도 된다. 둘을 묶어 두면 rrc 를 바꿀 때마다 critic 을 다시 학습해야 한다.
    ck_l, ck_r, ck_a = sd.get("latency"), sd.get("replan"), sd.get("action_dim")
    if None in (ck_l, ck_r, ck_a):
        raise SystemExit(f"{ckpt} 에 latency/replan/action_dim 이 없다 (구버전 체크포인트)")
    if ck_a != action_dim:
        raise SystemExit(f"action_dim 불일치: 체크포인트 {ck_a} vs 지금 {action_dim}. "
                         f"modality 가 학습 때와 다르다.")
    if sd.get("state_dim") is not None and sd["state_dim"] != state_dim:
        raise SystemExit(f"state 차원 불일치: 체크포인트 {sd['state_dim']} vs 지금 {state_dim}. "
                         f"modality 가 학습 때와 다르다.")
    window = ck_l + ck_r                      # critic 이 보는 청크 스텝 수
    if window < exec_off + replan:
        raise SystemExit(
            f"critic 의 액션 창이 실행 구간을 못 덮는다.\n"
            f"  critic 창    : {window} 스텝 (체크포인트: latency {ck_l} + replan {ck_r})\n"
            f"  실행 구간    : [{exec_off}, {exec_off + replan}) — yaml 의 "
            f"inference_latency={exec_off}, replan_steps={replan}\n"
            f"  → critic 을 더 긴 창으로 다시 학습하거나 실행 오프셋을 줄일 것")
    action_full = window * ck_a
    dim_feat = sd["feat_mu"].shape[-1]
    latent_img = sd.get("latent") or cfg.latent_dim_image

    # ── offline_iql_qvgm 산출물 (Q-VGM 배선) ────────────────────────────────────
    if sd.get("kind") == "qvgm":
        # --action-groups 로 학습했으면 critic 은 액션의 일부 열만 본다. 서빙은 전 차원
        # 액션을 그대로 넘기고 ServingCritic.q 가 잘라 쓴다 — 그래야 ExpoServer 의
        # 후보 확장·guidance 마스크(전 차원 기준)를 건드리지 않는다.
        aidx = sd.get("action_index")
        if aidx:
            action_full = len(aidx)
        slat = int(sd.get("state_latent") or 0)
        if slat > 0:
            enc = FuseProj(dim_feat, state_dim, latent_img, slat).to(dev).eval()
            in_latent = enc.out_dim
        else:
            enc = Proj(dim_feat, latent_img).to(dev).eval()
            in_latent = latent_img + state_dim
        enc.load_state_dict(sd["enc"])
        hid = tuple(sd["hidden_dims"])
        nst = int(sd.get("n_steps", 1))

        def build_q():
            return StepwiseEnsemble(in_latent, action_full, nst,
                                    sd.get("num_qs", 2), hid,
                                    bool(sd["critic_layer_norm"]),
                                    sd.get("inject", True),
                                    int(sd.get("bins") or 0)).to(dev).eval()

        critic, target = build_q(), build_q()
        critic.load_state_dict(sd["critic"])
        target.load_state_dict(sd.get("target", sd["critic"]))
        if sd.get("bins"):
            lo, hi = (float(x) for x in (sd.get("q_range") or "0,1").split(","))
            e = torch.linspace(lo, hi, sd["bins"] + 1, device=dev)
            ctr = (e[:-1] + e[1:]) / 2
            q_of = lambda x: (x.softmax(-1) * ctr).sum(-1)
        else:
            q_of = lambda x: x
        print(f"  [critic] {ckpt.name}  kind=qvgm  step {sd.get('step')}  "
              f"bins {sd.get('bins')}  num_qs {sd.get('num_qs')}  n_steps {nst}  "
              f"inject {sd.get('inject')}  latent {in_latent}"
              + (f" = LN(concat[{latent_img},{slat}])" if slat else
                 f" = {latent_img}+{state_dim} raw"))
        print(f"  [critic] 액션 창 {window} 스텝 x {ck_a} 관절 = {action_full}차원 "
              f"(체크포인트 기준)  |  실행 구간 [{exec_off}, {exec_off + replan}) (yaml 기준)")
        c = ServingCritic(enc, critic, target, q_of, sd["feat_mu"].to(dev),
                          sd["feat_sd"].to(dev), n_cog, sd, dev,
                          fuse=slat > 0, stepwise=True,
                          act_index=torch.as_tensor(aidx, device=dev) if aidx else None)
        # full 은 **호출측이 넘길 액션의 차원**이다 (전 차원). 잘라 쓰는 것은 내부 사정이라
        # ExpoServer 의 편집 마스크는 전 차원 기준으로 그대로 둔다.
        c.window, c.full, c.action_dim = window, window * ck_a, ck_a
        if aidx:
            print(f"  [critic] 액션 열 제한: {len(aidx)}/{window*ck_a} 차원 "
                  f"(그룹 {sd.get('action_groups')}) — 서빙은 전 차원을 넘기고 내부에서 자른다")
        return c

    enc = Proj(dim_feat, latent_img).to(dev).eval()
    enc.load_state_dict(sd["enc"])
    in_latent = latent_img + state_dim          # state 를 raw 로 붙인다 (offline_iql 과 동일)

    def build():
        m = CriticEnsemble(in_latent, state_dim, action_full, sd.get("num_qs", cfg.num_qs),
                           cfg.latent_dim_state, False, cfg.hidden_dims,
                           cfg.critic_layer_norm).to(dev).eval()
        if sd.get("bins"):
            for q in m.qs:
                q.head = xavier_(nn.Linear(q.body.out_dim, sd["bins"])).to(dev)
        return m

    critic, target = build(), build()
    critic.load_state_dict(sd["critic"])
    target.load_state_dict(sd.get("target", sd["critic"]))
    if sd.get("bins"):
        lo, hi = (float(x) for x in (sd.get("q_range") or "0,1").split(","))
        edges = torch.linspace(lo, hi, sd["bins"] + 1, device=dev)
        centers = (edges[:-1] + edges[1:]) / 2
        q_of = lambda x: (x.softmax(-1) * centers).sum(-1)
    else:
        q_of = lambda x: x
    print(f"  [critic] {ckpt.name}  step {sd.get('step')}  bins {sd.get('bins')}  "
          f"num_qs {sd.get('num_qs')}  latent {in_latent} = {latent_img}+{state_dim}  "
          f"features {feat}  n_cog {n_cog}")
    print(f"  [critic] 액션 창 {window} 스텝 x {ck_a} 관절 = {action_full}차원 "
          f"(체크포인트 기준)  |  실행 구간 [{exec_off}, {exec_off + replan}) (yaml 기준)")
    c = ServingCritic(enc, critic, target, q_of, sd["feat_mu"].to(dev),
                      sd["feat_sd"].to(dev), n_cog, sd, dev)
    c.window, c.full, c.action_dim = window, action_full, ck_a
    return c
