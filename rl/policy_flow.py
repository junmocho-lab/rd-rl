#!/usr/bin/env python3
"""RLDX-1 의 action expert 를 **미분 가능한 velocity field** 로 노출한다.

RLDX-1 은 손대지 않는다. `get_action_with_features` 안의 지역 클로저 `_dit_forward`
(third_party/RLDX-1/rldx/model/core/rldx.py:623-640) 가 정확히 v_θ(x, τ, c) 인데 밖에서
부를 수 없어서, **같은 계산을 public 모듈 속성으로 재구성**한다:

    action_encoder → (+position_embedding) → concat(state_features) → MSAT → action_decoder

확인한 전제 (checkpoints/openarm_0818_..._30k_4gpu_mlxp):
  · `get_action_with_features` 에 @torch.no_grad() 가 없다 → gradient 가 흐른다
    (RTC guided 모드가 실제로 이 경로에서 액션 입력에 대한 VJP 를 돌린다, rldx.py:701)
  · use_physics=False → physics 토큰이 NoOp. 궤적 의존 상태가 없다
  · use_memory=False → 백본 출력이 프레임 독립
  · num_inference_timesteps=4 (배포). 학습은 K 를 더 크게 줄 수 있다
  · MSAT dropout 0.2 → v_base 가 확률적이면 타깃이 노이즈에 묻힌다. LoRA 를 켜면
    tune_diffusion_model 이 False 가 되어 set_frozen_modules_to_eval_mode 가 MSAT 를
    eval 로 내린다. 여기서는 명시적으로 한 번 더 eval 을 강제한다

v_base 는 **LoRA adapter 를 끈 같은 가중치**다 (peft BaseTunerLayer.enable_adapters(False)).
별도 사본을 들고 있지 않으므로 VRAM 이 늘지 않는다.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch


@contextlib.contextmanager
def adapters_disabled(module: torch.nn.Module):
    """이 블록 안에서 LoRA 를 끈다 → frozen base velocity field f_β."""
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:                                # LoRA 를 안 쓰는 경우
        yield
        return
    layers = [m for m in module.modules() if isinstance(m, BaseTunerLayer)]
    for m in layers:
        m.enable_adapters(False)
    try:
        yield
    finally:
        for m in layers:
            m.enable_adapters(True)


class FlowPolicy:
    """RLDXVLA 를 감싸 (조건 c, x, τ) → velocity 를 내주는 얇은 층."""

    def __init__(self, vla, flow_steps: int = 10):
        self.vla = vla
        self.model = vla.model                          # RLDX
        self.am = vla.model.action_model                # RLDXActionModel
        self.flow_steps = flow_steps
        # 추론 경로와 정확히 같은 공간에서 적분한다 (rldx.py:601-609):
        #   horizon = config.action_horizon (16),  차원 = config.max_action_dim (64)
        # DiT 는 패딩된 64차원 토큰을 먹고, 뒤쪽 36차원은 학습 때 노이즈→0 으로 수렴한다.
        # 여기서 0 으로 채우면 추론 때와 입력 분포가 달라진다 → 노이즈를 전 차원에 준다.
        self.horizon = int(self.am.config.action_horizon)
        self.dim = int(self.am.action_dim)                # = max_action_dim
        self.real_dim = int(vla.action_dim)               # modality 의 실제 관절 수
        cfg = self.am.config
        self.add_pos = bool(getattr(cfg, "add_pos_embed", False))
        assert not bool(getattr(cfg, "use_physics", False)), \
            "use_physics=True 는 샘플러가 궤적 의존이 되어 이 경로가 성립하지 않는다"

    # --- 조건부 컨텍스트 (백본 1회) ------------------------------------------
    @torch.no_grad()
    def context(self, obs: dict) -> dict:
        """관측 → {vl, state_features, embodiment_id, enc_mask}. 백본은 여기서 딱 1회 돈다."""
        collated = self.vla._collate(obs)
        inp = collated["inputs"] if "inputs" in collated else collated
        bi, ai = self.model.prepare_input(inp)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bo = self.model.backbone(bi)
            feats = self.am._encode_features(bo, ai)     # vlln + state_encoder
        m = bo.get("backbone_attention_mask", None)
        return {"vl": feats["backbone_features"], "sf": feats["state_features"],
                "emb": ai.embodiment_id,
                "mask": None if (m is None or bool(m.all())) else m}

    # --- velocity field ------------------------------------------------------
    def velocity(self, ctx: dict, x: torch.Tensor, tau: torch.Tensor,
                 base: bool = False) -> torch.Tensor:
        """(B,H,A) x 와 (B,) tau → (B,H,A) velocity. rldx.py:623-640 의 _dit_forward 와 동일.

        base=True 면 LoRA 를 끈 f_β. x 에 대한 미분이 가능하다 (adjoint/VJP 용).
        """
        am = self.am
        B, Hh = x.shape[0], x.shape[1]
        assert x.shape[-1] == self.dim, f"x 는 max_action_dim={self.dim} 이어야 한다 ({x.shape})"
        xin = x
        t_tok = tau.unsqueeze(1).expand(-1, Hh)
        ctxmgr = adapters_disabled(am.model) if base else contextlib.nullcontext()
        with ctxmgr, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            af = am.action_encoder(xin, t_tok, ctx["emb"])
            if self.add_pos:
                pos = torch.arange(af.shape[1], dtype=torch.long, device=af.device)
                af = af + am.position_embedding(pos).unsqueeze(0)
            sa = torch.cat((ctx["sf"], af), dim=1)
            phy = am.physics.build_tokens(
                am.physics.prepare_inference(None, B, af.device, af.dtype), tau)
            mo = am.model(hidden_states=sa, encoder_hidden_states=ctx["vl"], timestep=tau,
                          encoder_attention_mask=ctx["mask"], physics_embs=phy,
                          physics_attention_mask=None)
            ao = mo["action"] if isinstance(mo, dict) else mo
            v = am.action_decoder(ao, ctx["emb"])[:, -Hh:]
        return v.float()

    # --- 롤아웃 -------------------------------------------------------------
    def rollout(self, ctx: dict, x0: torch.Tensor | None = None, base: bool = False):
        """detach 된 Euler 롤아웃. (xs, taus) — xs[k] 는 스텝 k 의 상태.

        Q-VGM 은 현재 정책으로 stop-gradient 롤아웃한다 (Algorithm 1, 라인 17).
        """
        K = self.flow_steps
        B = ctx["vl"].shape[0]
        x = (torch.randn(B, self.horizon, self.dim, device=ctx["vl"].device)
             if x0 is None else x0)
        xs, taus = [], []
        with torch.no_grad():
            for k in range(K):
                t = torch.full((B,), k / K, device=x.device, dtype=torch.float32)
                xs.append(x)
                taus.append(t)
                x = x + (1.0 / K) * self.velocity(ctx, x, t, base=base)
        return xs, taus

    def eval_mode(self):
        """dropout 을 끈다. v_base 가 확률적이면 adjoint/velocity 타깃이 노이즈에 묻힌다."""
        self.am.model.eval()
        self.am.action_encoder.eval()
        self.am.action_decoder.eval()
        self.am.state_encoder.eval()
        if self.add_pos:
            self.am.position_embedding.eval()


def obs_from_frames(imgs, flat, mod, task: str, idx: np.ndarray) -> dict:
    """rl/data.py make_batch 의 vla_obs 와 같은 규약 (extract_cogfeat 와 동일)."""
    x = np.asarray(imgs[idx])
    return {"video": {name: x[:, c][:, None] for c, (name, _) in enumerate(mod.video)},
            "state": {name: flat.state[idx][:, None, s:e] for name, s, e in mod.offsets("state")},
            "language": {mod.task_key: [[task]] * len(idx)}}


def chunk_mask(spec, horizon: int, dim: int, real_dim: int, device) -> torch.Tensor:
    """ExploreSpec 의 평탄 인덱스를 (horizon, max_action_dim) 청크 마스크로.

    spec.index 는 (latency+replan, real_dim) 평탄 인덱스다. DiT 는 (horizon, max_action_dim)
    으로 적분하므로 좌표를 옮겨 심어야 한다 — 그냥 view(-1)[:full] 로 넣으면 관절이 어긋난다.
    """
    m = torch.zeros(horizon, dim, device=device)
    idx = spec.index.cpu().numpy()
    m[idx // real_dim, idx % real_dim] = 1.0
    return m
