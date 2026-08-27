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

from rl.nets import (BatchEncoder, CriticEnsemble, StepwiseEnsemble, StepwiseV,
                     xavier_)


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

    def __init__(self, enc, critic, value, feat, mu, sd, snorm, meta, dev):
        self.enc, self.critic, self.value = enc, critic, value
        self.FEAT, self.SNORM, self.meta, self.dev = feat, snorm, meta, dev

    def latent_of(self, i, state=None):
        idx = torch.as_tensor(i, device=self.dev)
        z = self.enc(self.FEAT[idx])
        return torch.cat([z, self.SNORM[idx] if state is None else state], -1)

    def q_steps(self, lat, act):
        """(num_qs, B, n_steps) → min → (B, n_steps)."""
        return self.critic(lat, act).min(0).values

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
    enc = Proj(FEAT.shape[1], sd["latent"]).to(dev).eval()
    IN = sd["latent"] + sd["state_dim"]
    full = (sd["latency"] + sd["replan"]) * sd["action_dim"]
    critic = StepwiseEnsemble(IN, full, sd["n_steps"], sd["num_qs"], hid, ln,
                              sd.get("inject", True)).to(dev).eval()
    value = StepwiseV(IN, sd["n_steps"], hid, ln).to(dev).eval()
    enc.load_state_dict(sd["enc"])
    critic.load_state_dict(sd["critic"])
    value.load_state_dict(sd["value"])
    SN = torch.from_numpy(snorm).to(dev)
    print(f"[critic] {ckpt}\n          step {sd.get('step')}  γ={sd.get('discount')}  "
          f"τ={sd.get('expectile')}  stepwise {sd['n_steps']}  q x{sd['num_qs']}  "
          f"inject={sd.get('inject')}  features={sd['features']}")
    return StepwiseCritic(enc, critic, value, FEAT, sd["feat_mu"], sd["feat_sd"], SN, sd, dev)


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

    def __init__(self, enc, critic, target, q_of, mu, sd, n_cog, meta, dev):
        self.enc, self.critic, self.target, self.q_of = enc, critic, target, q_of
        self.mu, self.sd, self.n_cog, self.meta, self.dev = mu, sd, n_cog, meta, dev

    def cog_of(self, backbone_features: torch.Tensor) -> torch.Tensor:
        """(B, seq, d) 백본 출력 → (B, d) cog mean-pool.

        extract_cogfeat.py 와 **같은 규칙**이어야 한다: 마지막 n_cog 토큰의 평균, fp32.
        """
        return backbone_features[:, -self.n_cog:, :].float().mean(1)

    def latent(self, cog: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        z = (cog - self.mu) / self.sd
        return torch.cat([self.enc(z), state], -1)

    def q(self, lat, state, act, target: bool = False):
        """(num_qs, B) 스칼라 Q. 분포형이면 bin 기댓값."""
        m = self.target if (target and self.target is not None) else self.critic
        return self.q_of(m(lat, state, act))


def load_serving_critic(ckpt: Path, cfg, state_dim: int, action_full: int,
                        n_cog: int, dev: str = "cuda") -> ServingCritic:
    """cogfeat.npy 를 읽지 않고 체크포인트만으로 ServingCritic 을 만든다."""
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
    dim_feat = sd["feat_mu"].shape[-1]
    latent_img = sd.get("latent") or cfg.latent_dim_image
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
    return ServingCritic(enc, critic, target, q_of, sd["feat_mu"].to(dev),
                         sd["feat_sd"].to(dev), n_cog, sd, dev)
