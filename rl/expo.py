#!/usr/bin/env python3
"""EXPO-FT 학습 루프를 torch 로 옮긴 것.

원본: expo-ft/expo_ft/agents/alg/expo_ft.py (EXPOLearner)

update() 한 번에 벌어지는 일 (원본 _update_jit 과 같은 순서):
    1. batch (batch_size * utd_ratio) 를 utd_ratio 개 미니배치로 쪼갠다
    2. 미니배치마다 critic 업데이트   → utd_ratio 회
    3. actor(VLA) 업데이트 1회        → actor_success_only 면 성공 전용 배치로
    4. residual actor 업데이트 1회    → 마지막 미니배치로
    5. temperature 업데이트 1회

하이퍼파라미터는 configs/model/expo_ft_pi_config.py 값을 그대로 쓴다 (바꾸지 않는다).

원본과 다른 점:
  - **target_actor_params(VLA 의 EMA 사본) 을 두지 않는다.** 원본은 만들고 EMA 갱신까지
    하지만 어디서도 읽지 않는다 (expo_ft.py / bc.py 모두). torch 로 옮기면 1.24B 파라미터
    사본이 3.2GB GPU 메모리를 아무 효과 없이 점유한다. 쓰이는 곳이 생기면 그때 넣는다.
  - residual actor 는 탐색 대상 차원만 출력한다 (rl/nets.py 의 2번 참고).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields

import torch
import torch.nn as nn

from rl import ddp
from rl.nets import (BatchEncoder, CriticEnsemble, ExploreSpec, FuseProj,
                     FuseResidualActor, ResidualActor, StepwiseEnsemble, Temperature)


# --------------------------------------------------------------------------- #
# VLA 경계 — 원본 expo_ft/agents/vla/vla_base.py 에 대응
# --------------------------------------------------------------------------- #
class VLA:
    """학습 루프가 VLA 에 요구하는 것 전부. RLDXVLA / DummyVLA 가 이걸 구현한다.

    액션은 **환경 차원**(패딩 제거, 정규화된 모델 공간)으로 주고받는다.
    모델 내부의 패딩(RLDX: 64) 처리는 구현체가 감춘다.
    """

    action_dim: int          # 환경 액션 차원 (openarm 28, rby1m_rh56f1 34)
    action_horizon: int      # 모델이 한 번에 내는 청크 길이 (16 / 40)

    def sample(self, obs, num_samples: int) -> torch.Tensor:
        """(B, num_samples, action_horizon, action_dim). 백본은 1회만 돌아야 한다."""
        raise NotImplementedError

    def train_step(self, obs, target_actions: torch.Tensor) -> dict:
        """BC(flow matching) 한 스텝. target_actions: (B, action_horizon, action_dim)."""
        raise NotImplementedError


class DummyVLA(VLA):
    """CPU 검증용. 관측을 보지 않고 균등분포 청크를 낸다.

    실제 RLDXVLA 를 끼우기 전에 학습 루프의 shape·수치·흐름을 전부 검증하기 위한 것.
    """

    def __init__(self, action_dim: int, action_horizon: int, seed: int = 0):
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.gen = torch.Generator().manual_seed(seed)
        self.bias = nn.Parameter(torch.zeros(action_horizon, action_dim))
        self.opt = torch.optim.Adam([self.bias], lr=1e-3)
        self.calls = {"sample": 0, "train_step": 0}

    def sample(self, obs, num_samples: int) -> torch.Tensor:
        self.calls["sample"] += 1
        b = obs["batch_size"]
        x = torch.rand((b, num_samples, self.action_horizon, self.action_dim),
                       generator=self.gen) * 2 - 1
        return x + self.bias.detach()

    def train_step(self, obs, target_actions: torch.Tensor) -> dict:
        self.calls["train_step"] += 1
        loss = ((self.bias.unsqueeze(0) - target_actions) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"actor_loss": float(loss.detach())}


# --------------------------------------------------------------------------- #
@dataclass
class ExpoConfig:
    """configs/model/expo_ft_pi_config.py 값 그대로."""

    N: int = 8
    n_edit_samples: int = 8
    edit_scale: float = 0.2
    num_qs: int = 10
    num_min_qs: int = 2
    critic_layer_norm: bool = True
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    latent_dim_image: int = 512
    latent_dim_state: int = 64
    include_state: bool = True
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    encoder_stage_sizes: tuple[int, ...] = (3, 4, 6, 3)
    encoder_num_filters: int = 64
    init_temperature: float = 1.0
    entropy_scale: float = 1.0
    batch_size: int = 64
    utd_ratio: int = 20
    freeze_critic_encoder: bool = False
    actor_success_only: bool = True


    @classmethod
    def from_dict(cls, d: dict | None) -> "ExpoConfig":
        """실험 yaml 의 expo 블록 → ExpoConfig. 모르는 키는 실패시킨다."""
        d = dict(d or {})
        fields = {f.name for f in _dc_fields(cls)}
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"expo 블록에 모르는 키: {sorted(unknown)} (가능: {sorted(fields)})")
        for k in ("hidden_dims", "encoder_stage_sizes"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def deviations(self) -> dict:
        """EXPO-FT 원본 기본값과 다른 항목. 라운드 manifest 에 기록해 눈에 보이게 한다."""
        base = ExpoConfig()
        return {f.name: (getattr(base, f.name), getattr(self, f.name))
                for f in _dc_fields(ExpoConfig)
                if getattr(base, f.name) != getattr(self, f.name)}


class QvgmCritic(nn.Module):
    """qvgm(cog-feature) critic 을 EXPOLearner 의 (latent, state, action) 규약으로 감싼다.

    offline_iql_qvgm / critic_io.ServingCritic 과 같은 배선 — "지금 critic 구조 그대로":
        latent 인자 = **표준화된 cog feature** (B, dfeat). 이미지 latent 가 아니다.
        x = FuseProj(feat, state)                      state 는 여기서 융합된다
        q = StepwiseEnsemble(x, action[:, action_index])   창 전체 x 탐색 그룹 열만
        스칼라 Q = 분포형(bins)이면 bin 기대값, 위치 축 합 → (num_qs, B)

    forward 는 CriticEnsemble 과 같은 (lat, state, action, members) 시그니처라
    select_from_chunks / update_residual_actor 가 분기 없이 그대로 쓴다.
    학습 손실만 다르다 — update_critic 이 logits() + hl_gauss() 를 쓴다
    (스칼라 MSE 는 발산 이력이 있어 offline 과 같이 HL-Gauss CE 를 기본으로 한다).
    """

    def __init__(self, dfeat: int, state_dim: int, action_full: int,
                 action_index: list[int] | None, latent: int = 2048,
                 state_latent: int = 256, n_steps: int = 1, num_qs: int = 10,
                 hidden: tuple[int, ...] = (1024, 512), layer_norm: bool = True,
                 inject: bool = True, bins: int = 128,
                 q_range: tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        self.state_latent = int(state_latent)
        if self.state_latent <= 0:
            raise ValueError("QvgmCritic 은 FuseProj(state_latent>0) 배선만 지원한다 — "
                             "현재 critic 체크포인트들이 전부 이 구조다 (state_latent 256)")
        self.enc = FuseProj(dfeat, state_dim, latent, self.state_latent)
        in_dim = self.enc.out_dim
        adim = len(action_index) if action_index else action_full
        self.q = StepwiseEnsemble(in_dim, adim, n_steps, num_qs, tuple(hidden),
                                  layer_norm, inject, bins)
        self.num_qs, self.bins, self.n_steps = num_qs, bins, n_steps
        idx = torch.as_tensor(list(action_index), dtype=torch.long) if action_index else None
        self.register_buffer("action_index", idx)
        self.lo, self.hi = (float(q_range[0]), float(q_range[1]))
        edges = torch.linspace(self.lo, self.hi, bins + 1) if bins else torch.zeros(0)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2 if bins else torch.zeros(0))
        self.sigma = 0.75 * (self.hi - self.lo) / bins if bins else 0.0
        # PA-RL kernel_scale_final=1e-2 — offline_iql_qvgm 과 같은 head 축소 초기화
        with torch.no_grad():
            for m in self.q.qs:
                m.head.weight.mul_(1e-2)
                m.head.bias.zero_()

    def _x(self, feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.enc(feat, state)

    def _sel(self, action: torch.Tensor) -> torch.Tensor:
        return action[..., self.action_index] if self.action_index is not None else action

    def logits(self, feat, state, action, members: list[int] | None = None) -> torch.Tensor:
        """(m, B, n_steps[, bins]) — 분포형 로짓 (HL-Gauss 손실용)."""
        return self.q(self._x(feat, state), self._sel(action), members=members)

    def q_of(self, out: torch.Tensor) -> torch.Tensor:
        if self.bins:
            out = (out.softmax(-1) * self.centers).sum(-1)
        return out.sum(-1)                                        # (m, B)

    def forward(self, feat, state, action, members: list[int] | None = None) -> torch.Tensor:
        return self.q_of(self.logits(feat, state, action, members))

    def subsample(self, num_min_qs: int, generator: torch.Generator | None = None) -> list[int]:
        perm = torch.randperm(self.num_qs, generator=generator)
        return perm[:num_min_qs].tolist()

    def hl_gauss(self, logits: torch.Tensor, y: torch.Tensor,
                 weight: torch.Tensor | None = None) -> torch.Tensor:
        """offline_iql_qvgm 의 q_loss_fn 과 같은 HL-Gauss CE.

        logits (m,B,R,bins), y (B,) 스칼라 TD 타깃, weight (B,) = valid 마스크.
        """
        m, B, R = logits.shape[:3]
        yf = y[None, :, None].expand(m, B, R).reshape(-1)
        z = (self.edges - yf.clamp(self.lo, self.hi)[:, None]) / (self.sigma * 2 ** 0.5)
        cdf = 0.5 * (1 + torch.erf(z))
        pr = cdf[:, 1:] - cdf[:, :-1]
        pr = pr / pr.sum(-1, keepdim=True).clamp_min(1e-8)
        ce = -(pr * logits.reshape(-1, self.bins).log_softmax(-1)).sum(-1)
        if weight is not None:
            w = weight[None, :, None].expand(m, B, R).reshape(-1)
            return (ce * w).sum() / w.sum().clamp_min(1e-8)
        return ce.mean()


class EXPOLearner:
    """critic 앙상블 + residual(edit) actor + temperature + VLA."""

    def __init__(self, vla: VLA, spec: ExploreSpec, state_dim: int, n_cams: int,
                 replan_steps: int, cfg: ExpoConfig | None = None,
                 device: str | torch.device = "cpu", seed: int = 0, latency: int = 0,
                 qvgm: dict | None = None):
        self.vla, self.spec, self.cfg = vla, spec, cfg or ExpoConfig()
        self.replan_steps, self.latency = replan_steps, latency
        if vla.action_horizon < latency + replan_steps:
            raise ValueError(f"action_horizon({vla.action_horizon}) < latency({latency}) + "
                             f"replan_steps({replan_steps})")
        self.device = torch.device(device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        c = self.cfg
        # critic 과 residual 이 **같은 액션 벡터**를 본다: 청크 [0, latency+replan) 평탄화
        # (= spec.full_dim). prefix(이미 커밋된 latency 스텝)는 결정의 일부가 아니지만
        # 결정 이후 실제로 실행되는 액션이라 Q 의 인자에 들어가야 하고 (그래야 보상 창
        # [t, t+replan) 을 일으킨 액션이 전부 입력에 있다), 편집은 spec.index 가 실행
        # 구간만 가리켜 scatter 가 prefix 자리에 0 을 넣는 것으로 막는다.
        # rrc 도 prefix 를 버리므로 편집해봐야 실행되지 않고, critic 은 학습에서 항상
        # **실제 실행된** prefix 를 봤으므로 편집된 값을 넣으면 학습 분포 밖으로 나간다.
        full = spec.full_dim
        self.prefix_dim = latency * (full // (latency + spec.replan_steps))

        # 파라미터 초기화까지 seed 로 고정한다. self.gen 은 REDQ 부분집합 뽑기용이고
        # nn.Linear/Conv 의 초기화는 전역 RNG 를 타므로 이게 없으면 **같은 seed 로도
        # 매번 다른 θ₀** 가 나온다. 그러면 라운드 0 을 무엇으로 모았는지 기록할 수 없다
        # (actor 와 learner 가 서로 다른 θ₀ 를 갖는다 — 실제 θ₀ 일치는 learner 가
        # init 산출물을 내보내고 actor 가 그걸 로드해서 보장한다. 여기 seed 는 learner 가
        # 재시작해도 같은 θ₀ 를 다시 만들 수 있게 하는 것).
        #
        # 모듈은 CPU 에서 만들어진 뒤 .to(device) 되므로 초기화는 CPU RNG 만 쓴다 =
        # 기계·GPU 와 무관하다 (torch 버전은 같아야 한다). torch.manual_seed 는 CUDA
        # 전역 RNG 도 같이 잡는데, 학습 시작 시점에 한 번 부르는 것이라 의도한 동작이다
        # (VLA 디노이저 샘플링도 같은 seed 아래로 들어온다).
        torch.manual_seed(seed)

        self.qvgm = dict(qvgm) if qvgm is not None else None
        if self.qvgm is not None:
            # cog-feature critic (오프라인 qvgm 과 같은 구조). 이미지 인코더가 없다 —
            # 배치의 obs/next_obs 자리에 **표준화된 cog feature** 가 온다 (encode 는 통과).
            # 백본이 RL 에서 완전 동결이라 (setup_training) cog feature 는 라운드 간
            # 불변 = ingest 때 한 번 추출해 캐시하면 학습 루프에 백본이 안 들어온다
            # (next 후보 샘플링 vla.sample 제외 — 그건 EXPO 의 정의상 필요하다).
            self.encoder = None
            dfeat = int(self.qvgm["dfeat"])
            self.critic = QvgmCritic(
                dfeat, state_dim, full, self.qvgm.get("action_index"),
                latent=int(self.qvgm.get("latent", 2048)),
                state_latent=int(self.qvgm.get("state_latent", 256)),
                n_steps=int(self.qvgm.get("n_steps", 1)),
                num_qs=c.num_qs,
                hidden=tuple(self.qvgm.get("hidden", (1024, 512))),
                layer_norm=c.critic_layer_norm,
                inject=bool(self.qvgm.get("inject", True)),
                bins=int(self.qvgm.get("bins", 128)),
                q_range=tuple(self.qvgm.get("q_range", (0.0, 1.0)))).to(self.device)
            self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
            # edit policy 도 critic 과 같은 입력 배선 (FuseProj + action_index 열만)
            self.residual = FuseResidualActor(
                dfeat, state_dim, spec, self.qvgm.get("action_index") or [],
                latent=int(self.qvgm.get("latent", 2048)),
                state_latent=int(self.qvgm.get("state_latent", 256)),
                hidden_dims=c.hidden_dims).to(self.device)
        else:
            self.encoder = BatchEncoder(3 * n_cams, c.latent_dim_image, c.encoder_stage_sizes,
                                        c.encoder_num_filters).to(self.device)
            self.critic = CriticEnsemble(c.latent_dim_image, state_dim, full, c.num_qs,
                                         c.latent_dim_state, c.include_state, c.hidden_dims,
                                         c.critic_layer_norm).to(self.device)
            self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
            self.residual = ResidualActor(c.latent_dim_image, state_dim, spec, c.latent_dim_state,
                                          c.include_state, c.hidden_dims).to(self.device)
        self.temp = Temperature(c.init_temperature).to(self.device)

        # 원본은 critic 과 batch_encoder 에 각각 Adam(critic_lr) 을 둔다. Adam 은 파라미터
        # 단위라 하나로 묶어도 동일하다. (qvgm 은 인코더가 critic 안에 있다 — FuseProj)
        _enc_params = list(self.encoder.parameters()) if self.encoder is not None else []
        self.opt_critic = torch.optim.Adam(
            list(self.critic.parameters()) + _enc_params, lr=c.critic_lr)
        self.opt_residual = torch.optim.Adam(self.residual.parameters(), lr=c.actor_lr)
        self.opt_temp = torch.optim.Adam(self.temp.parameters(), lr=c.temp_lr)
        self.steps = {"critic": 0, "actor": 0, "residual": 0, "temp": 0}

        # 멀티 GPU 에서 backward 뒤 opt.step() 전에 gradient 를 rank 평균으로 맞춘다.
        # 단일 프로세스면 ddp.all_reduce_grads 가 즉시 반환하므로 분기하지 않는다.
        # (자세한 이유는 rl/ddp.py 머리말)
        self._critic_params = list(self.critic.parameters()) + _enc_params
        self._residual_params = list(self.residual.parameters())
        self._temp_params = list(self.temp.parameters())

    # --- 공통 ---------------------------------------------------------------
    def encode(self, obs: torch.Tensor, stop_gradient: bool) -> torch.Tensor:
        if self.encoder is None:
            # qvgm: obs 가 이미 표준화된 cog feature 다. 융합(FuseProj)은 critic 내부에서
            # state 와 함께 일어나고, 그 gradient 는 opt_critic 이 관리한다.
            return obs
        return self.encoder(obs, stop_gradient=stop_gradient)

    def _members(self) -> list[int]:
        return self.critic.subsample(self.cfg.num_min_qs, self.gen)

    # --- 후보 생성 + Q argmax (원본 sample_batch_actions) --------------------
    def select_from_chunks(self, chunks: torch.Tensor, latent: torch.Tensor, state: torch.Tensor,
                           prefix: torch.Tensor | None = None,
                           ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """이미 만들어진 후보 청크 (B,N,H,A) → 실행 구간 argmax.

        반환 (chosen (B,full), best (B,), info). best < N 이면 base 후보 그 자체,
        best >= N 이면 base 후보 (best-N) 에 edit 을 더한 것이다.

        샘플링과 분리해 둔 이유: 정책 서버는 후보를 RLDX 추론 파이프라인(RTC prefix 주입
        포함)에서 이미 받아 놓으므로 다시 sample() 을 부르면 안 된다. 그리고 서버는 best
        인덱스로 원래 청크를 되찾아 실행 구간만 갈아끼운다 (청크 전체를 돌려줘야 한다).
        """
        c = self.cfg
        B = latent.shape[0]
        s = self.latency
        n_base = chunks.shape[1]
        acts = chunks[:, :, :s + self.replan_steps].reshape(B, n_base, -1)   # (B,N,full)
        if self.prefix_dim and prefix is not None:
            # 학습 경로는 **로그된** prefix 를 넘긴다 — 학습 때는 RTC 를 끄고 샘플링하므로
            # 청크의 앞 latency 개가 실제 커밋된 값이 아니다. 서빙에서는 RTC 가 후보 전부에
            # 같은 prefix 를 박아 두므로 청크에 들어 있는 그대로 쓴다 (prefix=None).
            acts = acts.clone()
            acts[:, :, :self.prefix_dim] = prefix[:, None, :]

        if c.n_edit_samples > 0:
            k = min(c.n_edit_samples, n_base)
            base = acts[:, :k].reshape(B * k, -1)
            rep_lat = latent.repeat_interleave(k, 0)
            rep_st = state.repeat_interleave(k, 0)
            with torch.no_grad():
                edit, _ = self.residual.sample(rep_lat, rep_st, base, c.edit_scale)
            acts = torch.cat([acts, (base + edit).reshape(B, k, -1)], dim=1)

        total = acts.shape[1]
        members = self._members()
        with torch.no_grad():
            qs = self.target_critic(latent.repeat_interleave(total, 0),
                                    state.repeat_interleave(total, 0),
                                    acts.reshape(B * total, -1), members=members)
            q = qs.min(dim=0).values.view(B, total)
        best = q.argmax(dim=1)
        chosen = acts[torch.arange(B, device=acts.device), best]
        with_edit = (best >= n_base).float()
        return chosen, best, {
            "select_ratio_with_residual": float(with_edit.mean()),
            "select_ratio_without_residual": float(1 - with_edit.mean()),
            "candidate_q_std": float(q.std(dim=1).mean()),   # 후보 간 Q 분산 (0 이면 critic 이 액션을 구분 못함)
            "chosen_q": float(q.gather(1, best[:, None]).mean()),
        }

    def candidate_actions(self, vla_obs, latent: torch.Tensor, state: torch.Tensor,
                          prefix: torch.Tensor | None = None,
                          ) -> tuple[torch.Tensor, dict]:
        """관측에서 후보 N + n_edit 개를 만들고 target critic 으로 argmax.

        원본과 달리 base N 개를 잘라내지 않고 그대로 유지한 뒤 edit 을 덧붙인다
        (원본 롤아웃 경로는 N == n_edit_samples 를 가정하는 버그가 있다).
        """
        with torch.no_grad():
            chunks = self.vla.sample(vla_obs, num_samples=self.cfg.N)        # (B,N,H,A)
        chosen, _, info = self.select_from_chunks(chunks, latent, state, prefix)
        return chosen, info

    # --- critic (원본 update_critic) ----------------------------------------
    def update_critic(self, b: dict) -> dict:
        c = self.cfg
        next_lat = self.encode(b["next_obs"], stop_gradient=True)
        next_pre = b.get("next_action_prefix")
        next_action, sel = self.candidate_actions(b["vla_next_obs"], next_lat, b["next_state"],
                                                 next_pre)

        with torch.no_grad():
            members = self._members()
            next_qs = self.target_critic(next_lat, b["next_state"], next_action,
                                         members=members)
            next_q = next_qs.min(dim=0).values
            nan = torch.isnan(next_q)
            next_q = torch.where(nan, torch.zeros_like(next_q), next_q)
            target_q = b["reward"] + (c.discount ** self.replan_steps) * b["mask"] * next_q

        lat = self.encode(b["obs"], stop_gradient=c.freeze_critic_encoder)
        if self.qvgm is not None and self.critic.bins:
            # 분포형 헤드는 HL-Gauss CE (offline_iql_qvgm 과 동일 — 스칼라 MSE 는
            # 발산 이력이 있다). qs 는 로깅용 스칼라 환산값.
            logits = self.critic.logits(lat, b["state"], b["action"])
            loss = self.critic.hl_gauss(logits, target_q, weight=b["valid"])
            qs = self.critic.q_of(logits.detach())
        else:
            qs = self.critic(lat, b["state"], b["action"])
            loss = (((qs - target_q.unsqueeze(0)) ** 2) * b["valid"].unsqueeze(0)).mean()

        self.opt_critic.zero_grad(set_to_none=True)
        loss.backward()
        # rank 평균은 clip 전에 한다 — grad norm 은 로그용 지표이므로 실효 배치(평균 뒤)의
        # 값이어야 rank 수를 바꿔도 같은 것을 보게 된다.
        ddp.all_reduce_grads(self._critic_params)
        gnorm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float("inf"))
        self.opt_critic.step()

        with torch.no_grad():                                    # polyak
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.mul_(1 - c.tau).add_(p, alpha=c.tau)
        self.steps["critic"] += 1
        qd = qs.detach()
        return {"critic_loss": float(loss.detach()), "q": float(qd.mean()),
                "q_min": float(qd.min()), "q_max": float(qd.max()),
                "target_q_mean": float(target_q.mean()),
                "next_q_nan_ratio": float(nan.float().mean()),
                "critic_grad_norm": float(gnorm), **sel}

    # --- residual actor (원본 update_residual_actor) -------------------------
    def update_residual_actor(self, b: dict) -> dict:
        c = self.cfg
        lat = self.encode(b["obs"], stop_gradient=True).detach()
        edit, log_prob = self.residual.sample(lat, b["state"], b["action"], c.edit_scale)
        actions = edit + b["action"]
        q = self.critic(lat, b["state"], actions).mean(dim=0)     # 앙상블 평균 (원본과 동일)
        alpha = self.temp().detach()
        loss = (c.entropy_scale * log_prob * alpha - q).mean()

        self.opt_residual.zero_grad(set_to_none=True)
        loss.backward()
        ddp.all_reduce_grads(self._residual_params)
        self.opt_residual.step()
        self.steps["residual"] += 1
        entropy = float(-log_prob.mean().detach())
        return {"residual_actor_loss": float(loss.detach()), "residual_q": float(q.detach().mean()),
                "entropy": entropy, "mean_edit_norm": float(edit.detach().norm(dim=-1).mean())}

    # --- temperature (원본 update_temperature) -------------------------------
    def update_temperature(self, entropy: float) -> dict:
        loss = self.temp() * (entropy - self.spec.target_entropy)
        self.opt_temp.zero_grad(set_to_none=True)
        loss.backward()
        # entropy 는 rank 마다 자기 배치의 값이다. gradient 를 평균하면 결과적으로 실효
        # 배치의 평균 엔트로피로 갱신한 것과 같아진다 (loss 가 entropy 에 선형이라서).
        ddp.all_reduce_grads(self._temp_params)
        self.opt_temp.step()
        self.steps["temp"] += 1
        return {"temperature": float(self.temp().detach()), "temperature_loss": float(loss.detach())}

    # --- actor (VLA) --------------------------------------------------------
    def update_actor(self, b: dict) -> dict:
        info = self.vla.train_step(b["vla_obs"], b["full_action"])
        self.steps["actor"] += 1
        return info

    # --- 한 번의 update (원본 _update_jit) -----------------------------------
    def update(self, batch: dict, actor_batch: dict | None = None) -> dict:
        c = self.cfg
        total = batch["action"].shape[0]
        if total % c.utd_ratio:
            raise ValueError(f"배치 {total} 가 utd_ratio {c.utd_ratio} 의 배수가 아니다")
        n = total // c.utd_ratio

        info: dict = {}
        for i in range(c.utd_ratio):
            sl = slice(i * n, (i + 1) * n)
            info = self.update_critic(_slice(batch, sl))          # 원본은 마지막 것만 로그
        last = _slice(batch, slice(total - n, total))

        info.update(self.update_actor(actor_batch if (c.actor_success_only and actor_batch)
                                     else last))
        if c.n_edit_samples > 0:
            r = self.update_residual_actor(last)
            info.update(r)
            info.update(self.update_temperature(r["entropy"]))
        return info

    # --- 롤아웃 (원본 sample_actions) ---------------------------------------
    @torch.no_grad()
    def act(self, vla_obs, obs: torch.Tensor, state: torch.Tensor,
            prefix: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        lat = self.encode(obs, stop_gradient=True)
        return self.candidate_actions(vla_obs, lat, state, prefix)


def _sub(v, sl: slice):
    """배치 축을 따라 자른다. tensor / ndarray / list / 중첩 dict 를 모두 다룬다."""
    if torch.is_tensor(v):
        return v[sl]
    if isinstance(v, (list, tuple)):
        return v[sl]
    if isinstance(v, dict):
        return {k: _sub(x, sl) for k, x in v.items()}
    if hasattr(v, "shape") and hasattr(v, "__getitem__"):      # numpy 등
        return v[sl]
    return v


def _slice(b: dict, sl: slice) -> dict:
    """미니배치 자르기.

    vla_obs 는 {"video": {카메라: (B,1,H,W,3)}, "state": {...}, "language": {키: [[task]]*B}}
    형태의 **중첩 numpy dict** 이므로 안쪽까지 잘라야 한다. 예전 구현은 batch_size 만
    갈아끼웠는데 그건 DummyVLA(관측을 보지 않는다) 에서만 맞고, 실제 RLDXVLA 에서는
    critic 미니배치 n 개에 대해 후보를 B=total 개 뽑아 latent 와 shape 이 어긋난다.
    """
    n = sl.stop - sl.start
    out = {}
    for k, v in b.items():
        if isinstance(v, dict) and k.startswith("vla_"):
            sliced = _sub(v, sl)
            if "batch_size" in v:                              # DummyVLA 용
                sliced["batch_size"] = n
            out[k] = sliced
        else:
            out[k] = _sub(v, sl)
    return out


# --------------------------------------------------------------------------- #
def _verify() -> int:
    """CPU + DummyVLA 로 update 루프를 검증한다.

    utd_ratio / batch_size 는 CPU 시간 때문에 작게 줄인다 (알고리즘 값이 아니라 테스트 값).
    나머지 하이퍼파라미터는 EXPO-FT 기본값 그대로.
    """
    import json
    from pathlib import Path

    from rl.nets import explore_spec

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    repo = Path(__file__).resolve().parent.parent
    m = json.loads((repo / "modality" / "rby1m_rh56f1" / "modality.json").read_text())
    groups = sorted(((k, v["start"], v["end"]) for k, v in m["action"].items()), key=lambda g: g[1])
    offs, cum = [], 0
    for n, s, e in groups:
        offs.append((n, cum, cum + (e - s))); cum += e - s
    adim, replan, n_cams, latency = cum, 8, 3, 3
    spec = explore_spec(offs, ["right_arm_joints"], adim, replan, latency)

    torch.manual_seed(0)
    B, UTD = 8, 4
    cfg = ExpoConfig(utd_ratio=UTD, batch_size=B)
    vla = DummyVLA(adim, action_horizon=40)
    L = EXPOLearner(vla, spec, adim, n_cams, replan, cfg, seed=0, latency=latency)
    print(f"  fuji: action_dim={adim} replan={replan} latency={latency} "
          f"full={spec.full_dim} active={spec.out_dim} target_entropy={spec.target_entropy}")

    def batch(n):
        return {
            "obs": torch.randint(0, 255, (n, 192, 320, 3 * n_cams), dtype=torch.uint8),
            "next_obs": torch.randint(0, 255, (n, 192, 320, 3 * n_cams), dtype=torch.uint8),
            "state": torch.randn(n, adim), "next_state": torch.randn(n, adim),
            "action": torch.rand(n, spec.full_dim) * 2 - 1,
            "next_action_prefix": torch.rand(n, latency * adim) * 2 - 1,
            "full_action": torch.rand(n, 40, adim) * 2 - 1,
            "reward": (torch.rand(n) < 0.1).float(),
            "mask": (torch.rand(n) > 0.1).float(),
            "valid": torch.ones(n),
            "vla_obs": {"batch_size": n}, "vla_next_obs": {"batch_size": n},
        }

    # 1) 스텝 수
    tgt_before = [p.clone() for p in L.target_critic.parameters()]
    info = L.update(batch(B * UTD), actor_batch=batch(B))
    check("1 critic 스텝 == utd_ratio", L.steps["critic"] == UTD, str(L.steps))
    check("1 actor/residual/temp 각 1회",
          (L.steps["actor"], L.steps["residual"], L.steps["temp"]) == (1, 1, 1))
    check("1 VLA sample 호출 == critic 스텝 (백본 1회/스텝)",
          vla.calls["sample"] == UTD, f"sample={vla.calls['sample']}")

    # 2) 후보 수 = N + n_edit, 선택 비율 합 1
    r = info["select_ratio_with_residual"] + info["select_ratio_without_residual"]
    check("2 선택 비율 합 == 1", abs(r - 1) < 1e-6, f"{r}")
    check("2 후보 간 Q 분산 > 0 (critic 이 액션을 구분)", info["candidate_q_std"] > 0,
          f"{info['candidate_q_std']:.5f}")

    # 3) target_q 공식 — candidate_actions 를 고정 액션으로 바꿔치기해 정확히 대조
    b = batch(B)
    fixed = torch.zeros(B, spec.full_dim)
    L.candidate_actions = lambda vo, lat, st, pre=None: (
        fixed, {"select_ratio_with_residual": 0.0, "select_ratio_without_residual": 1.0,
                "candidate_q_std": 0.0})
    with torch.no_grad():
        nl = L.encode(b["next_obs"], stop_gradient=True)
        allq = L.target_critic(nl, b["next_state"], fixed)
        # 무작위 num_min_qs 개의 min 은 [전체 min, num_min_qs 번째로 큰 값] 사이에 있다
        lo = allq.min(dim=0).values
        hi = allq.topk(cfg.num_min_qs, dim=0, largest=True).values.min(dim=0).values
    g = cfg.discount ** replan
    exp_lo = float((b["reward"] + g * b["mask"] * lo).mean())
    exp_hi = float((b["reward"] + g * b["mask"] * hi).mean())
    got = L.update_critic(b)["target_q_mean"]
    check("3 target_q == reward + γ^replan · mask · min(Q_target)",
          exp_lo - 1e-4 <= got <= exp_hi + 1e-4,
          f"got={got:.5f} 범위=[{exp_lo:.5f}, {exp_hi:.5f}] γ^{replan}={g:.4f}")

    # 4) polyak: target 이 tau 만큼만 움직였나
    moved = [float((a - b_).abs().max()) for a, b_ in
             zip(L.target_critic.parameters(), tgt_before)]
    check("4 target_critic 이 조금씩만 움직임 (polyak tau=0.005)",
          0 < max(moved) < 1.0, f"최대 변화 {max(moved):.2e}")
    check("4 target_critic 에 gradient 없음",
          all(p.grad is None for p in L.target_critic.parameters()))

    # 5) temperature 방향. Adam 모멘텀이 방향 전환을 가리므로 각각 새 optimizer 로 본다.
    def temp_after(delta: float, steps: int = 3) -> tuple[float, float]:
        LL = EXPOLearner(DummyVLA(adim, 40), spec, adim, n_cams, replan, cfg, seed=2,
                         latency=latency)
        before = float(LL.temp().detach())
        for _ in range(steps):
            LL.update_temperature(LL.spec.target_entropy + delta)
        return before, float(LL.temp().detach())

    b0, up = temp_after(+10.0)
    b1, dn = temp_after(-10.0)
    check("5 entropy > target → temperature 감소", up < b0, f"{b0:.6f} → {up:.6f}")
    check("5 entropy < target → temperature 증가", dn > b1, f"{b1:.6f} → {dn:.6f}")

    # 6) 롤아웃 경로: 선택된 액션이 base 와 비활성 차원에서 같다
    L2 = EXPOLearner(DummyVLA(adim, 40, seed=1), spec, adim, n_cams, replan, cfg, seed=1,
                     latency=latency)
    bb = batch(4)
    chosen, sel = L2.act(bb["vla_obs"], bb["obs"], bb["state"])
    check("6 act() 출력 (B, full_dim)", chosen.shape == (4, spec.full_dim), str(tuple(chosen.shape)))
    check("6 Q/액션에 NaN 없음", bool(torch.isfinite(chosen).all()))

    mask = torch.zeros(spec.full_dim, dtype=torch.bool); mask[spec.index] = True
    with torch.no_grad():
        lat = L2.encode(bb["obs"], stop_gradient=True)
        base = L2.vla.sample(bb["vla_obs"], 1)[:, 0, :latency + replan].reshape(4, -1)
        edit, _ = L2.residual.sample(lat, bb["state"], base, cfg.edit_scale)
    check("6 편집은 활성 차원에만 — prefix 와 비탐색 그룹은 변화 0",
          bool((edit[:, ~mask] == 0).all()),
          f"비활성 {int((~mask).sum())}차원 (그중 prefix {L2.prefix_dim})")

    print(f"\n  [수치] γ^{replan}={cfg.discount**replan:.4f}  q={info['q']:.4f}  "
          f"critic_loss={info['critic_loss']:.4f}  temperature={info.get('temperature', 0):.4f}")
    print(f"  [수치] select_with_residual={info['select_ratio_with_residual']:.2f}  "
          f"entropy={info.get('entropy', 0):.1f} (target {spec.target_entropy})")
    # 7) 실험 yaml 의 expo 블록이 원본 기본값과 어긋나지 않는지
    import yaml as _yaml
    for name in ("openarm_rim", "fuji"):
        f = repo / "configs" / "exp" / f"{name}.yaml"
        if not f.is_file():
            continue
        d = _yaml.safe_load(f.read_text())
        c = ExpoConfig.from_dict(d.get("expo"))
        dev = c.deviations()
        check(f"7 {name}.yaml 의 expo 블록이 EXPO-FT 기본값과 동일", not dev,
              f"차이: {dev}" if dev else f"{len(_dc_fields(ExpoConfig))}개 항목 일치")

    # 8) θ₀ 재현성 — seed 가 파라미터 초기화까지 잡는지
    def theta(seed):
        lr = EXPOLearner(DummyVLA(adim, 40), spec, adim, n_cams, replan, cfg, seed=seed,
                         latency=latency)
        return torch.cat([p.flatten() for p in lr.critic.parameters()][:2] +
                         [p.flatten() for p in lr.residual.parameters()][:2])
    same = torch.equal(theta(0), theta(0))
    check("8 같은 seed → 같은 θ₀ (라운드 0 을 무엇으로 모았는지 기록하려면 필수)", same)
    check("8 다른 seed → 다른 θ₀", not torch.equal(theta(0), theta(1)))

    # 9) qvgm(cog-feature) critic 백엔드 — obs 자리에 표준화된 feature 가 온다
    dfeat = 96
    aidx = [t * adim + i for t in range(latency + replan)
            for i in range(3, 10)]                    # "오른팔 7관절" 흉내: 창 전체 x 7열
    qv = dict(dfeat=dfeat, action_index=aidx, latent=64, state_latent=16,
              hidden=(64, 32), bins=17, q_range=(0.0, 1.0))
    vq = DummyVLA(adim, 40, seed=3)
    Lq = EXPOLearner(vq, spec, adim, n_cams, replan, cfg, seed=0, latency=latency, qvgm=qv)

    def fbatch(n):
        b = batch(n)
        b["obs"] = torch.randn(n, dfeat)
        b["next_obs"] = torch.randn(n, dfeat)
        return b

    iq = Lq.update(fbatch(B * UTD), actor_batch=fbatch(B))
    check("9 qvgm: 1 update == utd_ratio critic 스텝", Lq.steps["critic"] == UTD, str(Lq.steps))
    check("9 qvgm: 후보간 Q 분산 > 0", iq["candidate_q_std"] > 0, f"{iq['candidate_q_std']:.5f}")
    check("9 qvgm: Q 가 support [0,1] 안", 0.0 <= iq["q"] <= 1.0, f"q={iq['q']:.4f}")
    check("9 qvgm: critic_loss 유한", math.isfinite(iq["critic_loss"]), str(iq["critic_loss"]))
    ch, _ = Lq.act({"batch_size": 4}, torch.randn(4, dfeat), torch.randn(4, adim))
    check("9 qvgm: act() 출력 (B, full_dim)", ch.shape == (4, spec.full_dim),
          str(tuple(ch.shape)))
    check("9 qvgm: 액션에 NaN 없음", bool(torch.isfinite(ch).all()))
    # action_index 밖의 열을 흔들어도 Q 가 안 변해야 한다 (critic 이 그 열을 안 본다)
    ftest = torch.randn(2, dfeat)
    stest, atest = torch.randn(2, adim), torch.rand(2, spec.full_dim) * 2 - 1
    amod = atest.clone()
    outside = [i for i in range(spec.full_dim) if i not in set(aidx)]
    amod[:, outside] += 10.0
    with torch.no_grad():
        dq = (Lq.critic(ftest, stest, atest) - Lq.critic(ftest, stest, amod)).abs().max()
    check("9 qvgm: action_index 밖 열은 Q 에 무영향", float(dq) == 0.0, f"Δ={float(dq):.2e}")

    print(f"\n{'전부 통과' if not fails else f'{len(fails)}개 실패: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_verify())
