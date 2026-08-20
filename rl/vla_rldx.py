#!/usr/bin/env python3
"""RLDX-1 을 EXPO-FT 학습 루프의 VLA 로 감싼 것.

원본 대응: expo-ft/expo_ft/agents/vla/pi05.py (Pi05Agent)

감추는 것:
  - 모델 액션 패딩 (RLDX max_action_dim=64) ↔ 환경 차원 (28/34/66)
  - 액션 공간 변환: raw LeRobot 절대값 ↔ 모델 공간 (상대 + percentile 정규화).
    변환은 RLDX 의 StateActionProcessor 를 그대로 쓴다 (우리가 재구현하지 않는다).
  - 백본 1회 + N 청크: get_action_with_features 를 감싸 features 를 N 배 확장한다.
    (측정: N=8 이 85ms, 단독 8회는 480ms)
  - autocast / grad 컨텍스트: policy.runtime._forward 를 그대로 쓴다.

실행은 pixi `rldx` 환경 (torch 2.8+cu128, python 3.10):
    cd third_party/RLDX-1 && PYTHONPATH="$PWD:<repo>" pixi run -e rldx python -m rl.vla_rldx
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from rl.data import Modality, rldx_layout
from rl.expo import VLA


class RLDXVLA(VLA):
    def __init__(self, model_path: Path | str, mod: Modality, rldx_root: Path | str,
                 rldx_config: str, device: str = "cuda", rtc_inference_mode: str = "none"):
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.policy.rldx_policy import RLDXPolicy

        # 등록 config 를 먼저 로드해야 태그가 존재한다 (같은 태그로 두 번 등록되지 않도록
        # rldx_layout 이 이미 로드했다면 그 결과를 재사용한다).
        tag = mod.embodiment_tag or rldx_layout(Path(rldx_root), rldx_config)[0]
        self.tag = tag
        self.mod = mod
        self.policy = RLDXPolicy(embodiment_tag=EmbodimentTag(tag), model_path=str(model_path),
                                 device=device, rtc_inference_mode=rtc_inference_mode)
        self.runtime = self.policy.runtime
        self.model = self.runtime.model
        self.proc = self.policy.processor.state_action_processor
        self.device = device

        self.action_dim = mod.action_dim
        self.action_horizon = int(self.model.config.action_horizon)
        self.max_action_dim = int(self.model.config.max_action_dim)
        self.max_action_horizon = int(getattr(self.policy.processor, "max_action_horizon",
                                             self.action_horizon))
        self._orig_gawf = self.model.action_model.get_action_with_features
        self.opt = None

    # --- 액션 공간 변환 ------------------------------------------------------
    def _split(self, x: np.ndarray, which: str) -> dict:
        """(..., dim) → {그룹명: (..., width)}  canonical 순서 기준."""
        return {n: x[..., s:e] for n, s, e in self.mod.offsets(which)}

    def _join(self, d: dict, which: str) -> np.ndarray:
        return np.concatenate([d[n] for n, _, _ in self.mod.offsets(which)], axis=-1)

    def normalize_actions(self, raw_chunk: np.ndarray, raw_state: np.ndarray) -> np.ndarray:
        """(B,H,A) 절대 액션 + (B,A) 상태 → (B,H,A) 모델 공간.

        상대 액션의 기준은 state[key][-1] 이므로 상태를 (B,1,A) 로 준다 = 그 프레임의 상태.
        """
        act = self._split(np.asarray(raw_chunk, np.float32), "action")
        st = self._split(np.asarray(raw_state, np.float32)[:, None, :], "state")
        out = self.proc.apply_action(act, self.tag, state=st)
        return self._join(out, "action")

    def denormalize_actions(self, norm_chunk: np.ndarray, raw_state: np.ndarray) -> np.ndarray:
        act = self._split(np.asarray(norm_chunk, np.float32), "action")
        st = self._split(np.asarray(raw_state, np.float32)[:, None, :], "state")
        out = self.proc.unapply_action(act, self.tag, state=st)
        return self._join(out, "action")

    # --- 샘플링 -------------------------------------------------------------
    def _collate(self, obs: dict) -> dict:
        from rldx.policy.step_request import decode_options_to_step_request

        b = len(next(iter(obs["video"].values())))
        req = decode_options_to_step_request(
            obs, {"session_ids": [f"rl{i}" for i in range(b)], "reset_memory": [True] * b})
        _, _, collated = self.runtime._prepare_inputs(req)
        return collated

    def sample(self, obs: dict, num_samples: int) -> torch.Tensor:
        """(B, num_samples, action_horizon, action_dim) 모델 공간 액션. 백본은 1회만 돈다."""
        collated = self._collate(obs)
        b = len(next(iter(obs["video"].values())))
        n = num_samples

        def rep(x):
            return x.repeat_interleave(n, dim=0) if torch.is_tensor(x) else x

        def expanding(backbone_features, state_features, embodiment_id, backbone_output,
                      action_input=None):
            bo = type(backbone_output)(data={k: rep(v) for k, v in backbone_output.items()})
            return self._orig_gawf(backbone_features=rep(backbone_features),
                                   state_features=rep(state_features),
                                   embodiment_id=rep(embodiment_id),
                                   backbone_output=bo, action_input=action_input)

        self.model.action_model.get_action_with_features = expanding
        try:
            out = self.runtime._forward(collated)
        finally:
            self.model.action_model.get_action_with_features = self._orig_gawf

        a = out["action_pred"]                                   # (B*N, H, max_action_dim)
        return a[..., :self.action_dim].reshape(b, n, self.action_horizon, self.action_dim)

    # --- 학습 (actor BC) ----------------------------------------------------
    def setup_training(self, lr: float = 3e-4, lora: bool = True) -> dict:
        """RL 단계의 학습 표면을 정한다.

        백본을 **완전히 동결**한다. BC 학습은 tune_top_llm_layers=4 로 상위 LLM 레이어를
        건드리지만, RL 에서 그대로 두면 (1) backbone_features 캐싱이 무효가 되어 계산량이
        폭증하고 (2) 라운드마다 그 레이어(1.54GB)를 전송해야 한다.

        action expert 는 LoRA 만 학습한다 (rank 16 → 약 4.8M 파라미터). 전체 미세조정은
        1.24B 파라미터에 Adam 모먼트까지 얹혀 로컬 32GB 에 안 들어간다.
        """
        am = self.model.action_model
        self.model.backbone.requires_grad_(False)
        if lora:
            am.config.action_model_use_lora = True
        am.set_trainable_parameters(tune_projector=False, tune_diffusion_model=False,
                                   tune_vlln=False)
        self.model.backbone.requires_grad_(False)          # LoRA 주입이 건드리지 않도록 재확인

        # flow matching 의 시간 샘플링을 fp32 로. autocast(bf16) 안에서 Beta 분포가
        # "dirichlet not implemented for BFloat16" 으로 죽는다. RLDX 소스(rldx.py:120-123)에
        # 같은 수정이 주석으로 남아 있다 — 학습 경로에서 실제로 밟는 문제라는 뜻.
        from torch.distributions import Beta
        am.beta_dist = Beta(torch.tensor(float(am.config.noise_beta_alpha)),
                            torch.tensor(float(am.config.noise_beta_beta)))
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(trainable, lr=lr)
        n_tr = sum(p.numel() for p in trainable)
        n_bb = sum(int(p.requires_grad) for p in self.model.backbone.parameters())
        return {"trainable_params": n_tr, "trainable_tensors": len(trainable),
                "backbone_trainable_tensors": n_bb}

    def _pad_actions(self, norm: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,H,A) 정규화 액션 → (B,maxH,maxA) + mask. processing_rldx.py 와 같은 규칙."""
        b, h, a = norm.shape
        out = np.zeros((b, self.max_action_horizon, self.max_action_dim), np.float32)
        out[:, :h, :a] = norm
        mask = np.zeros_like(out)
        mask[:, :h, :a] = 1.0
        dev = self.model.device
        return (torch.from_numpy(out).to(dev), torch.from_numpy(mask).to(dev))

    def train_step(self, obs: dict, raw_actions) -> dict:
        """flow matching BC 한 스텝. raw_actions 는 (B, H, action_dim) 절대 LeRobot 액션."""
        if self.opt is None:
            raise RuntimeError("setup_training() 을 먼저 호출할 것")
        raw_state = self._join({n: obs["state"][n][:, 0] for n, _, _ in self.mod.offsets("state")},
                               "state")
        norm = self.normalize_actions(np.asarray(raw_actions), raw_state)
        act, mask = self._pad_actions(norm)

        # runtime._prepare_inputs 는 {"inputs": {...}} 로 감싸서 준다 (_forward 가
        # model.get_action(**collated) 로 부르기 때문). forward 에는 안쪽 dict 을 넘긴다.
        collated = self._collate(obs)
        inp = collated["inputs"] if "inputs" in collated else collated
        inp["action"] = act
        inp["action_mask"] = mask
        # 백본은 파라미터가 전부 동결이고 입력도 grad 를 요구하지 않으므로 그래프가 생기지
        # 않는다 (activation 을 붙잡지 않는다). 그래프는 LoRA 가 붙은 action expert 에만.
        #
        # autocast 는 필요하다: 백본의 상위 LLM 레이어가 fp32 로 캐스팅되어 있어
        # (backbone_trainable_params_fp32=true) 없으면 flash-attn 이 fp32 입력에 죽는다.
        # Beta 분포 문제는 setup_training 에서 fp32 로 바꿔 해결한다.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.forward(inp)
        loss = out["loss"]

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {"actor_loss": float(loss.detach())}


# --------------------------------------------------------------------------- #
def _verify() -> int:
    """로컬 5090 + openarm base 정책 + 실제 데이터로 확인."""
    import json

    from rl.data import (build_flat, find_sessions, make_batch, open_images, resolve_modality)

    repo = Path(__file__).resolve().parent.parent
    rldx_root = repo / "third_party" / "RLDX-1"
    cfg_rel = "rldx/configs/data/openarm_inspire_config.py"
    ckpt = Path("/home/openarm14/ws/junmo_cho/checkpoints/0814-openarm-rh56f1-rldx-ptimg/"
                "openarm_0814_rh56f1_teleop_all200ep_egostereo_ptimg_framewt_drop03_rtc12tr_"
                "bs128_30k_4gpu_mlxp")
    data = repo / "rl-dataset/r0/0815_openarm_rh56f1_inference"
    mmp = Path("/tmp/claude-1000/-home-openarm14-ws-junmo-cho-rd-rl/"
               "66e43fa2-9640-4b3b-be95-636af3ec596a/scratchpad/img_full.mm")

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    mod, src = resolve_modality(data, None, rldx_root, cfg_rel, ckpt)
    print(f"  [modality] {src}")
    sessions = find_sessions(data)
    flat = build_flat(sessions, mod)
    imgs, meta = open_images(mmp)
    task = json.loads((sessions[0] / "meta" / "tasks.jsonl").read_text().splitlines()[0])["task"]
    print(f"  [task] {task!r}")

    t0 = time.time()
    vla = RLDXVLA(ckpt, mod, rldx_root, cfg_rel, device="cuda")
    print(f"  [load] {time.time()-t0:.1f}s  action_dim={vla.action_dim} "
          f"horizon={vla.action_horizon} max_action_dim={vla.max_action_dim} tag={vla.tag}")
    check("0 action_horizon 이 모델 config 와 일치", vla.action_horizon == 16, str(vla.action_horizon))

    B, N = 4, 8
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(flat) - 8, size=B)
    b = make_batch(flat, imgs, idx, mod, replan_steps=8, action_horizon=vla.action_horizon,
                   task=task, latency=2)

    # 1) 정규화 왕복
    norm = vla.normalize_actions(b["full_action"], b["state"])
    back = vla.denormalize_actions(norm, b["state"])
    check("1 정규화 shape 유지", norm.shape == b["full_action"].shape, str(norm.shape))
    err = float(np.abs(back - b["full_action"]).max())
    check("1 raw → 정규화 → raw 왕복 일치", err < 1e-3, f"최대 오차 {err:.2e}")
    check("1 정규화된 값이 대략 [-1,1] 안", float(np.abs(norm).max()) < 5,
          f"max|norm|={np.abs(norm).max():.3f}  max|raw|={np.abs(b['full_action']).max():.3f}")

    # 2) 샘플링
    for i in range(2):
        t = time.time()
        s = vla.sample(b["vla_obs"], N)
        dt = (time.time() - t) * 1000
    check("2 sample shape (B,N,H,action_dim)",
          tuple(s.shape) == (B, N, vla.action_horizon, vla.action_dim), str(tuple(s.shape)))
    sf = s.float().cpu().numpy()
    check("2 N개 샘플이 서로 다름", float(sf.std(axis=1).mean()) > 1e-4,
          f"샘플간 std {sf.std(axis=1).mean():.5f}")
    check("2 배치 항목끼리도 다름", float(sf[0].mean() - sf[1].mean()) != 0.0)
    print(f"  [속도] B={B} N={N} → {dt:.1f} ms ({dt/B:.1f} ms/obs), "
          f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # 3) 샘플과 정규화된 로그 액션이 같은 공간에 있나 (분포 비교)
    logged = norm[:, 2:10]                                     # latency=2 구간
    print(f"  [공간] 정규화된 로그 액션 std={logged.std():.3f} 범위=[{logged.min():.2f},"
          f"{logged.max():.2f}]")
    print(f"         모델 샘플        std={sf.std():.3f} 범위=[{sf.min():.2f},{sf.max():.2f}]")
    ratio = sf.std() / max(logged.std(), 1e-6)
    check("3 샘플과 로그 액션의 스케일이 같은 자리 (0.2~5배)", 0.2 < ratio < 5.0,
          f"std 비 {ratio:.2f}")

    # 4) 학습 표면
    info = vla.setup_training(lr=3e-4)
    print(f"  [학습] trainable {info['trainable_params']/1e6:.2f}M "
          f"({info['trainable_tensors']} 텐서), 백본 trainable 텐서 {info['backbone_trainable_tensors']}")
    check("4 백본에 학습 대상이 없음", info["backbone_trainable_tensors"] == 0)
    check("4 trainable 이 LoRA 규모 (1~20M)", 1e6 < info["trainable_params"] < 20e6,
          f"{info['trainable_params']/1e6:.2f}M")

    # 5) BC 스텝 — 손실이 유한하고 백본이 안 변해야 한다
    snap = {n: p.detach().clone() for n, p in
            list(vla.model.backbone.named_parameters())[:8]}
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        losses.append(vla.train_step(b["vla_obs"], b["full_action"])["actor_loss"])
    check("5 BC loss 가 유한", all(np.isfinite(losses)), f"{[round(x, 5) for x in losses]}")
    moved = max(float((p - snap[n]).abs().max())
                for n, p in list(vla.model.backbone.named_parameters())[:8])
    check("5 백본 파라미터가 안 변함 (특징 캐싱의 전제)", moved == 0.0, f"최대 변화 {moved:.2e}")
    print(f"  [학습] peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB, "
          f"loss {losses[0]:.5f} → {losses[-1]:.5f}")

    print(f"\n{'전부 통과' if not fails else f'{len(fails)}개 실패: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_verify())
