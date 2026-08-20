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

from rl.nets import BatchEncoder, CriticEnsemble, ExploreSpec, ResidualActor, Temperature


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


class EXPOLearner:
    """critic 앙상블 + residual(edit) actor + temperature + VLA."""

    def __init__(self, vla: VLA, spec: ExploreSpec, state_dim: int, n_cams: int,
                 replan_steps: int, cfg: ExpoConfig | None = None,
                 device: str | torch.device = "cpu", seed: int = 0, latency: int = 0):
        self.vla, self.spec, self.cfg = vla, spec, cfg or ExpoConfig()
        self.replan_steps, self.latency = replan_steps, latency
        if vla.action_horizon < latency + replan_steps:
            raise ValueError(f"action_horizon({vla.action_horizon}) < latency({latency}) + "
                             f"replan_steps({replan_steps})")
        self.device = torch.device(device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        c = self.cfg
        full = spec.full_dim

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
        # 단위라 하나로 묶어도 동일하다.
        self.opt_critic = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.encoder.parameters()), lr=c.critic_lr)
        self.opt_residual = torch.optim.Adam(self.residual.parameters(), lr=c.actor_lr)
        self.opt_temp = torch.optim.Adam(self.temp.parameters(), lr=c.temp_lr)
        self.steps = {"critic": 0, "actor": 0, "residual": 0, "temp": 0}

    # --- 공통 ---------------------------------------------------------------
    def encode(self, obs: torch.Tensor, stop_gradient: bool) -> torch.Tensor:
        return self.encoder(obs, stop_gradient=stop_gradient)

    def _members(self) -> list[int]:
        return self.critic.subsample(self.cfg.num_min_qs, self.gen)

    # --- 후보 생성 + Q argmax (원본 sample_batch_actions) --------------------
    def candidate_actions(self, vla_obs, latent: torch.Tensor, state: torch.Tensor,
                          ) -> tuple[torch.Tensor, dict]:
        """next_obs 에서 후보 N + n_edit 개를 만들고 target critic 으로 argmax.

        원본과 달리 base N 개를 잘라내지 않고 그대로 유지한 뒤 edit 을 덧붙인다
        (원본 롤아웃 경로는 N == n_edit_samples 를 가정하는 버그가 있다).
        """
        c = self.cfg
        B = latent.shape[0]
        with torch.no_grad():
            chunks = self.vla.sample(vla_obs, num_samples=c.N)              # (B,N,H,A)
        # RTC 지연: 앞 latency 개는 prefix 가 붙잡아 후보끼리 거의 같다. 실제로 실행되고
        # 후보끼리 다른 구간 [latency, latency+replan) 만 critic 에 넣는다.
        s = self.latency
        acts = chunks[:, :, s:s + self.replan_steps].reshape(B, c.N, -1)     # (B,N,full)

        if c.n_edit_samples > 0:
            k = min(c.n_edit_samples, c.N)
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
        with_edit = (best >= c.N).float()
        return chosen, {
            "select_ratio_with_residual": float(with_edit.mean()),
            "select_ratio_without_residual": float(1 - with_edit.mean()),
            "candidate_q_std": float(q.std(dim=1).mean()),   # 후보 간 Q 분산 (0 이면 critic 이 액션을 구분 못함)
        }

    # --- critic (원본 update_critic) ----------------------------------------
    def update_critic(self, b: dict) -> dict:
        c = self.cfg
        next_lat = self.encode(b["next_obs"], stop_gradient=True)
        next_action, sel = self.candidate_actions(b["vla_next_obs"], next_lat, b["next_state"])

        with torch.no_grad():
            members = self._members()
            next_qs = self.target_critic(next_lat, b["next_state"], next_action, members=members)
            next_q = next_qs.min(dim=0).values
            nan = torch.isnan(next_q)
            next_q = torch.where(nan, torch.zeros_like(next_q), next_q)
            target_q = b["reward"] + (c.discount ** self.replan_steps) * b["mask"] * next_q

        lat = self.encode(b["obs"], stop_gradient=c.freeze_critic_encoder)
        qs = self.critic(lat, b["state"], b["action"])
        loss = (((qs - target_q.unsqueeze(0)) ** 2) * b["valid"].unsqueeze(0)).mean()

        self.opt_critic.zero_grad(set_to_none=True)
        loss.backward()
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
    def act(self, vla_obs, obs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, dict]:
        lat = self.encode(obs, stop_gradient=True)
        return self.candidate_actions(vla_obs, lat, state)


def _slice(b: dict, sl: slice) -> dict:
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = v[sl]
        elif isinstance(v, dict) and k.startswith("vla_"):
            out[k] = {"batch_size": (sl.stop - sl.start), **{kk: vv for kk, vv in v.items()
                                                             if kk != "batch_size"}}
        else:
            out[k] = v
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
    spec = explore_spec(offs, ["right_arm_joints"], adim, replan)

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
    L.candidate_actions = lambda vo, lat, st: (fixed, {"select_ratio_with_residual": 0.0,
                                                       "select_ratio_without_residual": 1.0,
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
        base = L2.vla.sample(bb["vla_obs"], 1)[:, 0, latency:latency + replan].reshape(4, -1)
        edit, _ = L2.residual.sample(lat, bb["state"], base, cfg.edit_scale)
    check("6 편집은 활성 차원에만 (비활성 차원 변화 0)", bool((edit[:, ~mask] == 0).all()),
          f"비활성 {int((~mask).sum())}차원")

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

    print(f"\n{'전부 통과' if not fails else f'{len(fails)}개 실패: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_verify())
