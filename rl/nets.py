#!/usr/bin/env python3
"""EXPO-FT networks 를 torch 로 옮긴 것.

원본 대응 (expo-ft/expo_ft/):
    MLP                  networks/mlp.py
    ResNetV2Encoder      networks/encoders.py
    BatchEncoder         networks/pixel_multiplexer.py  (critic/edit 전용 비전 인코더)
    StateActionValue     networks/state_action_value.py
    Ensemble             networks/ensemble.py           (num_qs=10, 부분집합 min 은 num_min_qs=2)
    TanhNormal           distributions/tanh_normal.py
    Temperature          networks/temperature.py
    PixelMultiplexer     critic  래퍼 (이미지 latent + state latent [+ action])
    PixelEditMultiplexer residual actor 래퍼

원본과 의도적으로 다른 곳 두 군데 (둘 다 이유가 있다):

1. **critic 인코더 입력을 224x224 로 리사이즈한다.**
   ResNetV2Encoder 는 폭이 224 일 때만 앞단에서 7x7 stride2 + maxpool 로 4배 줄인다.
   우리 native 해상도(192x320)를 그냥 넣으면 flatten 이 491,520 이 되어 그 뒤 Dense 가
   251M 파라미터가 된다 (224 면 25,088 -> 12.8M). EXPO-FT 도 버퍼에 224x224 로 저장한다
   (pi05_resize_size=224). memmap 은 native 로 두고 배치 시점에 리사이즈한다 —
   embodiment 마다 해상도가 다르므로 (openarm 341x192, rby1m 256x256).

2. **residual actor 가 탐색 대상 차원만 출력한다.**
   원본은 full_action_dim 을 출력하고 마스킹으로 일부를 0 으로 만든다. 우리처럼 활성 비율이
   낮으면 (fuji: 105/510 = 20.6%) target_entropy = -full_action_dim/2 가 아무 효과 없는
   차원의 엔트로피로 채워지고, 정작 탐색해야 하는 차원이 ~10배 뾰족해진다.
   활성 차원만 출력하면 차원당 목표가 의도한 -0.5 로 유지된다.
   explore_groups 를 비우면 활성=전체가 되어 원본과 같아진다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_INPUT_SIZE = 224          # 위 1번 참고


def xavier_(m: nn.Module) -> nn.Module:
    """원본의 default_init = nn.initializers.xavier_uniform."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    return m


class MLP(nn.Module):
    """networks/mlp.py — Dense/(dropout)/(LayerNorm)/relu, activate_final 옵션."""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], activate_final: bool = False,
                 use_layer_norm: bool = False, dropout_rate: float | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for i, size in enumerate(hidden_dims):
            layers.append(xavier_(nn.Linear(d, size)))
            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate:
                    layers.append(nn.Dropout(dropout_rate))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(size))
                layers.append(nn.ReLU())
            d = size
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetV2Block(nn.Module):
    """encoders.py ResNetV2Block — pre-activation (GroupNorm→act→conv) x2 + skip."""

    def __init__(self, in_ch: int, filters: int, stride: int = 1, groups: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, eps=1e-5)
        self.conv1 = xavier_(nn.Conv2d(in_ch, filters, 3, stride, 1, bias=False))
        self.norm2 = nn.GroupNorm(groups, filters, eps=1e-5)
        self.conv2 = xavier_(nn.Conv2d(filters, filters, 3, 1, 1, bias=False))
        self.proj = (xavier_(nn.Conv2d(in_ch, filters, 1, stride, bias=False))
                     if (in_ch != filters or stride != 1) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.norm1(x)))
        y = self.conv2(F.relu(self.norm2(y)))
        return (self.proj(x) if self.proj is not None else x) + y


class ResNetV2Encoder(nn.Module):
    """encoders.py ResNetV2Encoder. 입력은 (B, C, 224, 224) 로 맞춰 들어온다."""

    def __init__(self, in_ch: int, stage_sizes: tuple[int, ...] = (3, 4, 6, 3),
                 num_filters: int = 64, input_size: int = ENCODER_INPUT_SIZE):
        super().__init__()
        self.input_size = input_size
        # 원본: 폭이 224 일 때 7x7 stride2 + maxpool, 아니면 3x3
        self.stem_downsample = input_size == 224
        if self.stem_downsample:
            self.stem = xavier_(nn.Conv2d(in_ch, num_filters, 7, 2, 3, bias=False))
            self.pool = nn.MaxPool2d(3, 2, padding=1)
        else:
            self.stem = xavier_(nn.Conv2d(in_ch, num_filters, 3, 1, 1, bias=False))
            self.pool = nn.Identity()

        blocks, ch = [], num_filters
        for i, n in enumerate(stage_sizes):
            filters = num_filters * 2 ** i
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                blocks.append(ResNetV2Block(ch, filters, stride))
                ch = filters
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.GroupNorm(4, ch, eps=1e-5)
        self.out_ch = ch

        s = input_size // (4 if self.stem_downsample else 1)
        for i in range(len(stage_sizes)):
            if i > 0:
                s = math.ceil(s / 2)
            self.out_dim = ch * s * s
        self.spatial = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stem(x))
        x = self.blocks(x)
        x = F.relu(self.norm(x))
        return x.flatten(1)


class BatchEncoder(nn.Module):
    """pixel_multiplexer.py BatchEncoder — encoder → Dense(latent) → LayerNorm → tanh.

    입력은 카메라를 채널로 이어붙인 uint8 (B, H, W, 3*n_cams). 224x224 로 리사이즈하고
    [-1, 1] 로 정규화한다 (원본 버퍼도 모델 입력 스케일로 저장된 이미지를 쓴다).
    """

    def __init__(self, in_ch: int, latent_dim: int = 512,
                 stage_sizes: tuple[int, ...] = (3, 4, 6, 3), num_filters: int = 64):
        super().__init__()
        self.encoder = ResNetV2Encoder(in_ch, stage_sizes, num_filters)
        self.proj = xavier_(nn.Linear(self.encoder.out_dim, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, obs_uint8: torch.Tensor, stop_gradient: bool = False) -> torch.Tensor:
        x = obs_uint8
        if x.dtype == torch.uint8:
            x = x.float().div_(127.5).sub_(1.0)          # [0,255] → [-1,1]
        if x.shape[-1] <= 16:                            # NHWC → NCHW
            x = x.permute(0, 3, 1, 2)
        if x.shape[-1] != ENCODER_INPUT_SIZE or x.shape[-2] != ENCODER_INPUT_SIZE:
            x = F.interpolate(x, size=(ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE),
                              mode="bilinear", align_corners=False)
        h = self.encoder(x)
        if stop_gradient:
            h = h.detach()
        return torch.tanh(self.norm(self.proj(h)))


class StateActionValue(nn.Module):
    """state_action_value.py — concat(obs, action) → MLP → Dense(1) → squeeze."""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...] = (256, 256, 256),
                 use_layer_norm: bool = True, dropout_rate: float | None = None):
        super().__init__()
        self.body = MLP(in_dim, hidden_dims, activate_final=True,
                        use_layer_norm=use_layer_norm, dropout_rate=dropout_rate)
        self.head = xavier_(nn.Linear(self.body.out_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


class CriticEnsemble(nn.Module):
    """ensemble.py Ensemble(StateActionValue, num=num_qs) + PixelMultiplexer.

    include_state 면 state 를 Dense(latent_dim_state)→LayerNorm→tanh 로 넣어
    이미지 latent 와 이어붙인다 (원본과 같은 순서).
    """

    def __init__(self, image_latent: int, state_dim: int, action_dim: int,
                 num_qs: int = 10, latent_dim_state: int = 64, include_state: bool = True,
                 hidden_dims: tuple[int, ...] = (256, 256, 256), critic_layer_norm: bool = True):
        super().__init__()
        self.include_state = include_state
        self.state_proj = xavier_(nn.Linear(state_dim, latent_dim_state)) if include_state else None
        self.state_norm = nn.LayerNorm(latent_dim_state) if include_state else None
        in_dim = image_latent + (latent_dim_state if include_state else 0) + action_dim
        self.qs = nn.ModuleList([
            StateActionValue(in_dim, hidden_dims, use_layer_norm=critic_layer_norm)
            for _ in range(num_qs)])
        self.num_qs = num_qs

    def _inputs(self, image_latent: torch.Tensor, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        x = image_latent
        if self.include_state:
            y = torch.tanh(self.state_norm(self.state_proj(state)))
            x = torch.cat([x, y], dim=-1)
        return torch.cat([x, action], dim=-1)

    def forward(self, image_latent: torch.Tensor, state: torch.Tensor, action: torch.Tensor,
                members: list[int] | None = None) -> torch.Tensor:
        """(len(members) or num_qs, B) — 원본 Ensemble 의 out_axes=0 과 같은 모양."""
        x = self._inputs(image_latent, state, action)
        idx = range(self.num_qs) if members is None else members
        return torch.stack([self.qs[i](x) for i in idx], dim=0)

    def subsample(self, num_min_qs: int, generator: torch.Generator | None = None) -> list[int]:
        """ensemble.py subsample_image_ensemble — REDQ 의 부분집합 (replace=False)."""
        perm = torch.randperm(self.num_qs, generator=generator)
        return perm[:num_min_qs].tolist()


class FuseProj(nn.Module):
    """Q-VGM 4.1 의 입력 융합: proj(RL token) 과 proj(proprio) 를 concat 한 뒤 LayerNorm.

    우리가 쓰던 방식과 두 군데 다르다:
      · state 를 raw 로 붙이지 않고 **투영**한다. raw 면 25차원이 512차원 latent 옆에
        스케일도 맞지 않은 채 놓여, MLP 첫 층이 사실상 latent 만 본다.
      · LayerNorm 을 **합친 뒤** 한 번 건다. 따로 걸면 두 갈래의 상대 크기가 고정되지
        않아, 학습이 진행되며 한쪽이 다른 쪽을 압도할 수 있다.

    액션은 여기 들어오지 않는다 — StepwiseQ 가 hidden 층마다 따로 concat 한다.
    """

    def __init__(self, dfeat: int, dstate: int, latent: int, state_latent: int):
        super().__init__()
        self.feat = xavier_(nn.Linear(dfeat, latent))
        self.state = xavier_(nn.Linear(dstate, state_latent))
        self.ln = nn.LayerNorm(latent + state_latent)
        self.out_dim = latent + state_latent

    def forward(self, f: torch.Tensor, st: torch.Tensor) -> torch.Tensor:
        return self.ln(torch.cat([self.feat(f), self.state(st)], -1))


# --------------------------------------------------------------------------- #
# Q-VGM 식 critic — 층마다 액션 재주입 + 청크 위치별(stepwise) 값
#
# 두 설계 모두 Q-VGM (arXiv 2606.08015) 4.1 절에서 왔고, 논문 ablation 이 각각의 값을
# 재 놓았다 (LIBERO 평균, 전체 92.5%):
#   · 층마다 액션 재주입 없음 → 88.2%.  latent 가 액션보다 훨씬 고차원이라 (우리는 4096 대
#     280) critic 이 액션을 무시하기 쉽다. ∇_A Q 를 쓰는 방법은 전부 이 민감도에 의존한다
#   · 단일 Q 헤드 → 90.1%.  헤드 2개의 min (clipped double Q)
#   · stepwise 아님 → 위 표에 없지만 본문이 "긴 지평 + 희소 보상에서 청크 전체에 값 하나는
#     너무 약한 지도 신호" 라고 명시한다. 우리는 219프레임에 보상이 끝 1프레임뿐이다
# --------------------------------------------------------------------------- #
class StepwiseQ(nn.Module):
    """(latent, action) → 청크 위치별 Q^(i). (B, n_steps)

    보통 MLP 와 다른 점은 **hidden 층마다 액션을 다시 concat** 한다는 것뿐이다.
    """

    def __init__(self, in_dim: int, action_dim: int, n_steps: int,
                 hidden_dims: tuple[int, ...] = (512, 512, 512), use_layer_norm: bool = True,
                 inject: bool = True, bins: int = 0):
        super().__init__()
        self.inject = inject
        # bins > 0 이면 위치마다 스칼라 대신 **bin logits** 를 낸다 (HL-Gauss 분포형).
        # support 를 [0,1] 로 고정하면 Q^(i) 가 그 밖으로 못 나가 발산이 구조적으로
        # 불가능해진다 — 스칼라 헤드에서 실패 Q 가 29 까지 폭발한 것이 이것 때문이다.
        self.bins = bins
        self.blocks = nn.ModuleList()
        d = in_dim + action_dim
        for h in hidden_dims:
            layer = [xavier_(nn.Linear(d, h))]
            if use_layer_norm:
                layer.append(nn.LayerNorm(h))
            layer.append(nn.GELU())
            self.blocks.append(nn.Sequential(*layer))
            d = h + (action_dim if inject else 0)      # 다음 층 입력에 액션을 다시 붙인다
        self.head = xavier_(nn.Linear(d - (action_dim if inject else 0),
                                      n_steps * bins if bins else n_steps))
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, action], -1)
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if self.inject and i < len(self.blocks) - 1:
                h = torch.cat([h, action], -1)
        o = self.head(h)
        return o.view(*o.shape[:-1], self.n_steps, self.bins) if self.bins else o


class StepwiseEnsemble(nn.Module):
    """StepwiseQ 를 num_qs 개. (num_qs, B, n_steps). Q-VGM 은 2개의 min 을 쓴다."""

    def __init__(self, in_dim: int, action_dim: int, n_steps: int, num_qs: int = 2,
                 hidden_dims: tuple[int, ...] = (512, 512, 512), use_layer_norm: bool = True,
                 inject: bool = True, bins: int = 0):
        super().__init__()
        self.qs = nn.ModuleList([
            StepwiseQ(in_dim, action_dim, n_steps, hidden_dims, use_layer_norm, inject, bins)
            for _ in range(num_qs)])
        self.bins = bins
        self.num_qs, self.n_steps = num_qs, n_steps

    def forward(self, x: torch.Tensor, action: torch.Tensor,
                members: list[int] | None = None) -> torch.Tensor:
        """members 를 주면 그 부분집합만 계산한다 (REDQ 의 subsample)."""
        qs = self.qs if members is None else [self.qs[k] for k in members]
        return torch.stack([q(x, action) for q in qs], 0)


class StepwiseV(nn.Module):
    """액션 없는 stepwise value. (B, n_steps). IQL expectile 회귀 대상."""

    def __init__(self, in_dim: int, n_steps: int, hidden_dims: tuple[int, ...] = (512, 512, 512),
                 use_layer_norm: bool = True):
        super().__init__()
        self.body = MLP(in_dim, hidden_dims, activate_final=True, use_layer_norm=use_layer_norm)
        self.head = xavier_(nn.Linear(self.body.out_dim, n_steps))
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class Temperature(nn.Module):
    """networks/temperature.py — exp(log_temp)."""

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(initial_temperature)))

    def forward(self) -> torch.Tensor:
        return self.log_temp.exp()


# --------------------------------------------------------------------------- #
# 탐색 대상 차원
# --------------------------------------------------------------------------- #
@dataclass
class ExploreSpec:
    """어느 액션 차원을 residual(edit) 로 탐색할지.

    modality 의 action 그룹 이름으로 선언한다 (configs/exp/<이름>.yaml 의 explore_groups).
    비어 있으면 전체 그룹 = EXPO-FT 원본과 동일.

    full_dim 은 **critic 과 같은 액션 벡터** 다: 청크 [0, latency+replan_steps) 를 평탄화한
    것. 앞 latency 스텝(prefix)은 이미 커밋되어 편집할 수 없으므로 index 에서 빠지고,
    scatter 가 그 자리에 0 을 넣는다 = 마스킹. 그래서 critic 과 edit policy 가 같은 입력을
    보면서도 편집은 실행 구간에만 들어간다.
    """

    index: torch.Tensor        # (active_dim*replan,) 액션 벡터 안의 편집 가능 위치
    active_dim: int            # 스텝당 활성 차원 수
    full_dim: int              # (latency + replan_steps) * action_dim
    replan_steps: int
    groups: tuple[str, ...]
    latency: int = 0

    @property
    def out_dim(self) -> int:
        return self.active_dim * self.replan_steps

    @property
    def target_entropy(self) -> float:
        """원본 -full_action_dim/2 를 활성 차원 기준으로. 차원당 -0.5 를 유지한다."""
        return -self.out_dim / 2

    def scatter(self, residual: torch.Tensor) -> torch.Tensor:
        """(B, out_dim) → (B, full_dim), 활성 위치에만 값을 넣고 나머지는 0."""
        out = residual.new_zeros(residual.shape[0], self.full_dim)
        out[:, self.index] = residual
        return out


def explore_spec(offsets: list[tuple[str, int, int]], groups: list[str] | tuple[str, ...],
                 action_dim: int, replan_steps: int, latency: int = 0) -> ExploreSpec:
    """offsets 는 rl.data.Modality.offsets("action") 결과.

    latency > 0 이면 액션 벡터가 prefix 를 포함해 (latency+replan)*action_dim 으로 넓어지고,
    편집 가능 위치는 스텝 [latency, latency+replan) 에만 잡힌다.
    """
    names = [n for n, _, _ in offsets]
    unknown = [g for g in groups if g not in names]
    if unknown:
        raise ValueError(f"모르는 action 그룹: {unknown} (가능: {names})")
    sel = list(groups) if groups else names
    per_step = [i for n, s, e in offsets if n in sel for i in range(s, e)]
    idx = [t * action_dim + i
           for t in range(latency, latency + replan_steps) for i in per_step]
    return ExploreSpec(index=torch.tensor(idx, dtype=torch.long), active_dim=len(per_step),
                       full_dim=(latency + replan_steps) * action_dim,
                       replan_steps=replan_steps, groups=tuple(sel), latency=latency)


class ResidualActor(nn.Module):
    """PixelEditMultiplexer + TanhNormal — base 액션을 보고 편집량 분포를 낸다.

    입력  : 이미지 latent, state, base action (full_dim)
    출력  : 활성 차원만의 TanhNormal (위 2번 참고). scatter 로 full_dim 에 배치한다.
    """

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

    def __init__(self, image_latent: int, state_dim: int, spec: ExploreSpec,
                 latent_dim_state: int = 64, include_state: bool = True,
                 hidden_dims: tuple[int, ...] = (256, 256, 256),
                 dropout_rate: float | None = None):
        super().__init__()
        self.spec = spec
        self.include_state = include_state
        self.state_proj = xavier_(nn.Linear(state_dim, latent_dim_state)) if include_state else None
        self.state_norm = nn.LayerNorm(latent_dim_state) if include_state else None
        in_dim = image_latent + (latent_dim_state if include_state else 0) + spec.full_dim
        self.body = MLP(in_dim, hidden_dims, activate_final=True, dropout_rate=dropout_rate)
        self.mean = xavier_(nn.Linear(self.body.out_dim, spec.out_dim))
        self.log_std = xavier_(nn.Linear(self.body.out_dim, spec.out_dim))

    def dist(self, image_latent: torch.Tensor, state: torch.Tensor,
             base_action: torch.Tensor) -> torch.distributions.Distribution:
        x = image_latent
        if self.include_state:
            y = torch.tanh(self.state_norm(self.state_proj(state)))
            x = torch.cat([x, y], dim=-1)
        h = self.body(torch.cat([x, base_action], dim=-1))
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        normal = torch.distributions.Normal(mean, log_std.exp())
        return torch.distributions.TransformedDistribution(
            torch.distributions.Independent(normal, 1),
            torch.distributions.TanhTransform(cache_size=1))

    def sample(self, image_latent: torch.Tensor, state: torch.Tensor, base_action: torch.Tensor,
               edit_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """(scatter 된 편집량 (B, full_dim), log_prob (B,)).

        log_prob 에는 원본과 같은 edit_scale 변수변환 보정을 넣는다
        (expo_ft.py: log_probs -= actions.shape[-1] * log(edit_scale)).
        """
        d = self.dist(image_latent, state, base_action)
        u = d.rsample()
        log_prob = d.log_prob(u) - self.spec.out_dim * math.log(edit_scale)
        return self.spec.scatter(u * edit_scale), log_prob


# --------------------------------------------------------------------------- #
def _verify() -> int:
    """CPU 에서 shape / 파라미터 수 / scatter / gradient 확인."""
    import json
    from pathlib import Path

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    repo = Path(__file__).resolve().parent.parent

    def offsets_of(mod_name: str) -> tuple[list[tuple[str, int, int]], int]:
        m = json.loads((repo / "modality" / mod_name / "modality.json").read_text())
        groups = sorted(((k, v["start"], v["end"]) for k, v in m["action"].items()),
                        key=lambda g: g[1])
        out, cum = [], 0
        for n, s, e in groups:
            out.append((n, cum, cum + (e - s)))
            cum += e - s
        return out, cum

    torch.manual_seed(0)
    B = 4

    for name, mod_name, groups, replan, latency in (
            ("openarm (전체 탐색, latency 2)", "openarm_lefthand", [], 8, 2),
            ("fuji (오른팔만)", "rby1m_rh56f1", ["right_arm_joints"], 15, 0)):
        print(f"\n=== {name} ===")
        offs, adim = offsets_of(mod_name)
        spec = explore_spec(offs, groups, adim, replan, latency)
        full = (latency + replan) * adim
        print(f"  action_dim={adim} replan={replan} full={full} "
              f"active={spec.out_dim} ({spec.out_dim/full:.1%}) target_entropy={spec.target_entropy}")
        check("spec.full_dim", spec.full_dim == full, str(spec.full_dim))
        check("spec.index 개수 == out_dim", len(spec.index) == spec.out_dim)
        check("spec.index 중복 없음", len(set(spec.index.tolist())) == spec.out_dim)
        check("target_entropy == -out_dim/2", spec.target_entropy == -spec.out_dim / 2)

        # scatter: 활성 위치에만 값, 나머지 0
        r = torch.arange(1, spec.out_dim + 1, dtype=torch.float32).repeat(B, 1)
        sc = spec.scatter(r)
        mask = torch.zeros(full, dtype=torch.bool)
        mask[spec.index] = True
        check("scatter 활성 위치에 값", bool((sc[:, mask] == r).all()))
        check("scatter 나머지는 0", bool((sc[:, ~mask] == 0).all()),
              f"비활성 {int((~mask).sum())}차원")

        n_cams = 3 if mod_name == "rby1m_rh56f1" else 2
        enc = BatchEncoder(in_ch=3 * n_cams)
        H, W = (256, 256) if mod_name == "rby1m_rh56f1" else (192, 320)
        obs = torch.randint(0, 255, (B, H, W, 3 * n_cams), dtype=torch.uint8)
        lat = enc(obs)
        check("BatchEncoder 출력 (B, 512)", lat.shape == (B, 512), str(tuple(lat.shape)))
        check("latent 이 tanh 범위 안", bool(lat.abs().max() <= 1.0))
        npar = sum(p.numel() for p in enc.parameters())
        check("인코더 파라미터 < 60M (224 리사이즈 확인)", npar < 60e6,
              f"{npar/1e6:.1f}M  flatten={enc.encoder.out_dim} spatial={enc.encoder.spatial}")

        critic = CriticEnsemble(512, adim, full, num_qs=10)
        state = torch.randn(B, adim)
        act = torch.randn(B, full)
        q = critic(lat, state, act)
        check("critic 출력 (num_qs, B)", q.shape == (10, B), str(tuple(q.shape)))
        members = critic.subsample(2)
        q2 = critic(lat, state, act, members=members)
        check("subsample 출력 (num_min_qs, B)", q2.shape == (2, B), f"members={members}")

        actor = ResidualActor(512, adim, spec)
        edit, logp = actor.sample(lat, state, act, edit_scale=0.2)
        check("residual 출력 (B, full_dim)", edit.shape == (B, full), str(tuple(edit.shape)))
        check("log_prob 출력 (B,)", logp.shape == (B,), str(tuple(logp.shape)))
        check("비활성 차원의 편집량은 0", bool((edit[:, ~mask] == 0).all()))
        check("|편집량| < edit_scale (tanh)", bool(edit.abs().max() < 0.2),
              f"max={edit.abs().max():.4f}")

        temp = Temperature(1.0)
        check("temperature 초기값 1.0", abs(float(temp()) - 1.0) < 1e-6)

        # gradient 가 흐르는지
        loss = q.mean() + logp.mean() + temp() * 0
        loss.backward()
        g_enc = sum(int(p.grad is not None) for p in enc.parameters())
        g_cri = sum(int(p.grad is not None) for p in critic.parameters())
        g_act = sum(int(p.grad is not None) for p in actor.parameters())
        check("gradient 가 인코더/critic/actor 전부에 흐름",
              g_enc > 0 and g_cri > 0 and g_act > 0, f"{g_enc}/{g_cri}/{g_act} 파라미터")

        print(f"  [수치] 파라미터 인코더 {npar/1e6:.1f}M / critic "
              f"{sum(p.numel() for p in critic.parameters())/1e6:.1f}M / residual "
              f"{sum(p.numel() for p in actor.parameters())/1e6:.1f}M")

    print(f"\n{'전부 통과' if not fails else f'{len(fails)}개 실패: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_verify())
