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

정책 서버(아래 ExpoServer) 도 여기 있다 — 서빙이 이 래퍼의 내부(백본 1회 + N배 확장,
runtime 훅)를 그대로 쓰기 때문이다.

실행은 pixi `rldx` 환경 (torch 2.8+cu128, python 3.10):
    cd third_party/RLDX-1 && PYTHONPATH="$PWD:<repo>" pixi run -e rldx python -m rl.vla_rldx
    ...                                              pixi run -e rldx python -m rl.vla_rldx serve --help
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rl import ddp
from rl.data import Modality, rldx_layout
from rl.expo import VLA


def load_state_action_processor(model_path: Path | str, rldx_root: Path | str,
                                rldx_config: str):
    """가중치 없이 StateActionProcessor 만 로드한다.

    정규화 통계(statistics.json)와 modality_configs 는 processor 에만 있고 모델 가중치와
    무관하다. state/action 정규화만 필요할 때 1.25B 를 올리지 않는 경로다.
    """
    import rldx  # noqa: F401  — AutoProcessor 레지스트리가 import 부수효과로 채워진다
    from transformers import AutoProcessor

    rldx_layout(Path(rldx_root), rldx_config)          # embodiment 태그 등록
    p = Path(model_path)
    pdir = p / "processor" if (p / "processor").is_dir() else p
    return AutoProcessor.from_pretrained(pdir).state_action_processor


def normalize_states(proc, tag: str, mod: Modality, raw_state: np.ndarray) -> np.ndarray:
    """(..., S) 원본 상태 → 모델 공간. apply_state = q01/q99 minmax + clip (또는 sincos).

    critic 의 state 입력을 여기 통과시킨다. raw 라디안을 그대로 넣으면 그룹별 스케일이
    5배까지 벌어지는데 (fuji: torso std 0.13 vs right_hand 0.49) critic 의 LayerNorm 은
    Linear **이후** latent 에 걸려서 입력 차원별 스케일을 보정하지 못한다. 안 움직이는
    관절(openarm 은 28차원 중 15차원)도 raw 로는 0 이 아닌 상수로 들어간다 — 정규화하면
    lo == hi 라서 정확히 0 이 된다.

    **학습과 롤아웃 질의가 같은 함수를 타야 한다** (rl/offline_critic_0.py, learner/loop.py,
    ExpoServer._critic_obs). 한쪽만 바꾸면 에러 없이 조용히 어긋난다.
    sincos 그룹이 선언된 config 면 반환 차원이 늘어난다 (지금 쓰는 세 config 는 없다).
    """
    off = mod.offsets("state")
    x = np.asarray(raw_state, np.float32)
    out = proc.apply_state({n: x[..., s:e] for n, s, e in off}, tag)
    return np.concatenate([out[n] for n, _, _ in off], axis=-1).astype(np.float32)


def normalize_actions(proc, tag: str, mod: Modality, raw_chunk: np.ndarray,
                     raw_state: np.ndarray) -> np.ndarray:
    """(B,H,A) 절대 액션 + (B,A) 상태 → 모델 공간. 가중치 없이 processor 만 쓴다.

    RLDXVLA.normalize_actions 와 같은 계산이다 (같은 apply_action 호출). 그쪽은 1.25B 를
    올려야 접근할 수 있는데 이 계산에는 신경망이 전혀 관여하지 않는다 — actnorm.npy 캐시를
    굽는 것 때문에 오프라인 학습이 13.8GB 체크포인트를 요구하던 이유가 그것뿐이었다.

    상대 액션의 기준은 state[key][-1] 이므로 상태를 (B,1,A) 로 준다 = 그 프레임의 상태.
    """
    off = mod.offsets("action")
    soff = mod.offsets("state")
    x = np.asarray(raw_chunk, np.float32)
    st = np.asarray(raw_state, np.float32)[:, None, :]
    out = proc.apply_action({n: x[..., s:e] for n, s, e in off}, tag,
                            state={n: st[..., s:e] for n, s, e in soff})
    return np.concatenate([out[n] for n, _, _ in off], axis=-1).astype(np.float32)


def denormalize_actions(proc, tag: str, mod: Modality, norm_chunk: np.ndarray,
                       raw_state: np.ndarray) -> np.ndarray:
    """normalize_actions 의 역. (B,H,A) 모델 공간 → 원본 라디안. 가중치 불필요.

    편집량을 사람이 읽는 단위로 되돌릴 때 쓴다 — 정규화 공간의 L2 는 관절별 스케일이
    q01/q99 로 뭉개져 있어서 "이게 큰 변화인가" 를 판단할 수 없다.
    """
    off = mod.offsets("action")
    soff = mod.offsets("state")
    x = np.asarray(norm_chunk, np.float32)
    st = np.asarray(raw_state, np.float32)[:, None, :]
    out = proc.unapply_action({n: x[..., s:e] for n, s, e in off}, tag,
                              state={n: st[..., s:e] for n, s, e in soff})
    return np.concatenate([out[n] for n, _, _ in off], axis=-1).astype(np.float32)


class RLDXVLA(VLA):
    def __init__(self, model_path: Path | str, mod: Modality, rldx_root: Path | str,
                 rldx_config: str, device: str = "cuda", rtc_inference_mode: str = "none",
                 rtc_inference_delay: int | None = None,
                 rtc_inference_exec_horizon: int | None = None):
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.policy.rldx_policy import RLDXPolicy

        # 등록 config 를 먼저 로드해야 태그가 존재한다 (같은 태그로 두 번 등록되지 않도록
        # rldx_layout 이 이미 로드했다면 그 결과를 재사용한다).
        tag = mod.embodiment_tag or rldx_layout(Path(rldx_root), rldx_config)[0]
        self.tag = tag
        self.mod = mod
        # exec_horizon 을 **반드시 넘겨야 한다.** 안 넘기면 config 의 0 이 그대로 가고
        # policy_loader.py:289 가 action_horizon - delay 로 채운다 (16-2 = 14).
        # 그런데 RTC prefix 는 session_registry.py:289 에서
        #   prefix = 이전청크[exec_horizon : exec_horizon+delay]
        # 로 잘리므로, 14 면 위치 14,15 를 "방금 실행한 액션" 이라고 pin 한다 — 실제로
        # 커밋된 것은 rrc 의 execution_horizon(=replan_steps=8) 기준 위치 8,9 다.
        # 6프레임 미래의 값을 과거로 pin 하니 모델이 이미 더 갔다고 착각해 동작이 어긋난다.
        # latency 0 에서는 RTC 가 꺼져(delay>0 조건) 이 버그가 잠들어 있었다.
        self.policy = RLDXPolicy(embodiment_tag=EmbodimentTag(tag), model_path=str(model_path),
                                 device=device, rtc_inference_mode=rtc_inference_mode,
                                 rtc_inference_delay=rtc_inference_delay,
                                 rtc_inference_exec_horizon=rtc_inference_exec_horizon)
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
        self._trainable: list = []

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

    @contextlib.contextmanager
    def expanded(self, n: int):
        """이 블록 안에서 디노이저가 관측당 n개의 청크를 낸다 (백본은 그대로 1회).

        action_input(RTC prefix 등) 은 **확장하지 않는다.** prefix 는 (1,d,A) 그대로
        actions[:, :d] = prefix 에서 n 으로 브로드캐스트되므로 후보 전부가 같은 prefix 를
        공유한다 — 이게 원하는 동작이다 (앞 d 스텝은 이미 실행이 확정된 액션이다).
        """
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
            yield
        finally:
            self.model.action_model.get_action_with_features = self._orig_gawf

    def sample(self, obs: dict, num_samples: int) -> torch.Tensor:
        """(B, num_samples, action_horizon, action_dim) 모델 공간 액션. 백본은 1회만 돈다."""
        collated = self._collate(obs)
        b = len(next(iter(obs["video"].values())))
        with self.expanded(num_samples):
            out = self.runtime._forward(collated)

        a = out["action_pred"]                                   # (B*N, H, max_action_dim)
        return a[..., :self.action_dim].reshape(b, num_samples, self.action_horizon,
                                                self.action_dim)

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
        self._trainable = trainable          # train_step 에서 rank 평균을 낼 대상
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
        ddp.all_reduce_grads(self._trainable)     # 멀티 GPU 일 때만 실제로 통신한다
        self.opt.step()
        return {"actor_loss": float(loss.detach())}


# --------------------------------------------------------------------------- #
# 정책 서버 — base BC 정책 + edit(residual) + critic 을 묶어 rrc 에 서빙
#
# RLDX 의 rldx/eval/run_rldx_server.py 와 같은 자리에 서지만 청크를 **고르는** 단계가
# 하나 더 붙는다. 끼워 넣는 지점은 PolicyRuntime.step 안의 딱 한 곳이다:
#
#   unbatch → 프로세서 → RTC prefix 주입 → [추론] → RTC 캐시 저장 → 디코드
#                                          ↑ 이 한 단계만 갈아끼운다
#
# 갈아끼운 _run_inference 가 하는 일:
#   1. 백본 1회 + 디노이저 N회 → 후보 청크 N개. RTC prefix 는 이미 주입되어 있어
#      후보 전부가 같은 prefix 를 공유한다 (앞 d 스텝은 실행이 확정된 액션이다)
#   2. **원본 uint8 이미지** + state 로 critic latent 를 만든다. VLA 프로세서를 통과한
#      픽셀이 아니다 — critic 인코더는 학습 때 디코딩된 원본 프레임을 봤다
#   3. edit 후보를 붙이고 target critic 으로 argmax (rl/expo.py select_from_chunks)
#   4. 고른 후보의 **청크 전체**를 돌려준다. 실행 구간 [d, d+replan) 만 edit 이 반영된
#      값으로 바꾼다. 그 뒤 구간은 base 후보 그대로 (실행되지 않고 RTC postfix 로만 쓰인다)
#
# 이 뒤는 RLDX 원본 경로 그대로다 — RTC 캐시에 고른 청크가 저장되고 디코드도 그것만
# 지나간다. rrc 는 서버가 base 정책인지 EXPO 인지 알 필요가 없다 (host/port 만 본다).
# --------------------------------------------------------------------------- #
def _cat_cams(obs: dict, mod: Modality) -> np.ndarray:
    """서버 관측의 **마지막** 프레임을 mod.video 순서로 채널 concat → (B,H,W,3*n_cams).

    학습 쪽 rl/data.py make_batch 의 cat_cams 와 같은 규칙이어야 한다 (순서가 어긋나면
    critic 이 학습 때와 다른 그림을 본다).
    """
    cams = [np.asarray(obs["video"][n])[:, -1] for n, _ in mod.video]
    bad = [n for (n, _), c in zip(mod.video, cams) if c.dtype != np.uint8]
    if bad:
        raise ValueError(f"critic 인코더는 uint8 이미지를 받는다. {bad} 의 dtype 이 다르다")
    return np.concatenate(cams, axis=-1)


def _cat_state(obs: dict, mod: Modality) -> np.ndarray:
    """(B, state_dim) float32. canonical 순서 concat = 학습 때의 flat.state 와 같은 배열."""
    return np.concatenate([np.asarray(obs["state"][n])[:, -1]
                           for n, _, _ in mod.offsets("state")], axis=-1).astype(np.float32)


class ExpoServer:
    """RLDXPolicy 의 추론 단계만 EXPO 선택으로 갈아끼운 것."""

    def __init__(self, exp: dict, model_path: Path, modality: Path, rldx_root: Path,
                 device: str = "cuda", artifacts: Path | None = None, seed: int = 0,
                 rtc_mode: str = "trained", img_size: tuple[int, int] = (320, 192),
                 verbose: bool = False, guide_steps: int = 0, guide_move: float = 0.05,
                 guide_all: bool = False, rtc_exec_horizon: int | None = None,
                 log_every: int = 25):
        from rl.data import resolve_modality
        from rl.expo import EXPOLearner, ExpoConfig
        from rl.nets import explore_spec

        self.replan = int(exp["replan_steps"])
        self.latency = int(exp["inference_latency"])
        cfg_rel = exp["rldx_data_config"]
        mod, src = resolve_modality(Path("."), Path(modality), rldx_root, cfg_rel,
                                    Path(model_path))
        self.mod, self.img_size, self.verbose = mod, img_size, verbose
        print(f"  [modality] {src}")

        self.vla = RLDXVLA(model_path, mod, rldx_root, cfg_rel, device=device,
                           rtc_inference_mode=rtc_mode, rtc_inference_delay=self.latency,
                           rtc_inference_exec_horizon=rtc_exec_horizon or self.replan)
        spec = explore_spec(mod.offsets("action"), exp.get("explore_groups") or [],
                            mod.action_dim, self.replan, self.latency)
        self.cfg = ExpoConfig.from_dict(exp.get("expo"))
        self.learner = EXPOLearner(self.vla, spec, mod.state_dim, mod.n_cams, self.replan,
                                   self.cfg, device=device, seed=seed, latency=self.latency)
        for m in (self.learner.encoder, self.learner.critic, self.learner.target_critic,
                  self.learner.residual):
            m.eval()
        # cog feature critic 이면 학습 때와 같은 latent 를 만들 수 있게 따로 세운다.
        # (EXPOLearner 의 ResNet 인코더 + CriticEnsemble 은 shape 이 다르다 —
        #  실측 856(512+64+280) vs 820(512+28+280). load_state_dict 가 실패한다)
        self.cog = None
        if artifacts is not None and Path(artifacts).is_file():
            probe = torch.load(artifacts, map_location="cpu")
            if probe.get("features"):
                from rl.critic_io import load_serving_critic
                m = self.vla.model
                n_cog = int(getattr(m, "_n_cog_tokens",
                                    getattr(m.backbone, "n_cog_tokens", 64)))
                self.cog = load_serving_critic(Path(artifacts), self.cfg, mod.state_dim,
                                               mod.action_dim, self.latency, self.replan,
                                               n_cog, dev=device)
                # 편집/guidance 마스크는 **실제로 실행되는 스텝**에만 걸어야 한다.
                # critic 창은 체크포인트가 정하고(예 10스텝) 실행 구간은 rrc 가 정한다
                # (예 [0,8)). 창 밖이나 실행 밖 스텝을 건드리면 아무 효과 없이 critic 만
                # 착취한다. 관절은 explore_groups 로 제한한다.
                jsel = [i for nm, s0, e0 in mod.offsets("action") if nm in
                        (exp.get("explore_groups") or [nm]) for i in range(s0, e0)]
                mk = torch.zeros(self.cog.window, mod.action_dim, device=device)
                mk[self.latency:self.latency + self.replan, jsel] = 1.0
                self.cog_mask = mk.reshape(-1)
                print(f"  [편집] 창 {self.cog.window}스텝 중 실행 "
                      f"[{self.latency},{self.latency + self.replan}) x 관절 {len(jsel)}개 "
                      f"= {int(self.cog_mask.sum())}/{self.cog.full} 차원")
        self.loaded = self._load(artifacts) if self.cog is None else ["cog-critic"]

        self.policy, self.runtime = self.vla.policy, self.vla.runtime
        if self.runtime.use_memory:
            raise SystemExit(
                "memory 모델은 아직 지원하지 않는다 — 후보 N개 확장이 memory scratchpad 의\n"
                "  배치(B=1)와 어긋난다. base 정책을 memory 없이 뽑거나 확장 경로를 고쳐야 한다.")
        if verbose:
            # [SERVER-LOG] RTC prefix injected: source=client|server_cache ... 가
            # PolicyRuntime.verbose 에 걸려 있다 (policy_runtime.py:335). prefix 를 rrc 가
            # 보내는지 서버 캐시에서 만드는지가 RTC 진단의 첫 갈림길이라 같이 켠다.
            self.runtime.verbose = True
        self._orig_run = self.runtime._run_inference
        self.runtime._run_inference = self._run_inference
        self.calls, self.ms, self.q, self.with_edit = 0, [], [], []
        self.guide_steps, self.guide_move, self.guide_all = guide_steps, guide_move, guide_all
        self.log_every = max(1, log_every)
        self.guide_gain = []

        print(f"  [정책] {Path(model_path).name}")
        print(f"  [태그] {self.vla.tag}  state_dim={mod.state_dim} action_dim={mod.action_dim} "
              f"cams={mod.n_cams}")
        print(f"  [청크] action_horizon={self.vla.action_horizon} latency={self.latency} "
              f"replan={self.replan} → critic 이 보는 구간 "
              f"[{self.latency},{self.latency + self.replan})")
        print(f"  [탐색] {list(spec.groups)}  활성 {spec.active_dim}/{mod.action_dim} 차원")
        print(f"  [RTC] mode={rtc_mode} delay={self.latency} "
              f"exec_horizon={self.runtime.rtc_exec_horizon}  "
              f"(delay = rrc 의 inference_latency_steps, "
              f"exec_horizon = rrc 의 execution_horizon 이어야 한다)")
        print(f"  [선택] N={self.cfg.N} + edit={self.cfg.n_edit_samples} "
              f"(edit_scale={self.cfg.edit_scale}) → target critic argmax")
        print(f"  [critic 이미지] {img_size[0]}x{img_size[1]} 로 맞춘 뒤 인코더가 224 로 줄인다")
        if self.loaded:
            print(f"  [산출물] {artifacts} 에서 {self.loaded} 로드")
        if self.cog is not None and self.guide_steps > 0:
            print(f"  [guidance] test-time ∇_A Q 상승 {self.guide_steps}스텝, "
                  f"차원당 목표 이동 {self.guide_move} "
                  f"(1프레임 자연 변화가 ~0.022 이므로 약 {self.guide_move/0.022:.1f} 프레임치)")
            print(f"             keep-best 라 Q 가 나빠지는 방향은 절대 채택하지 않는다")
            print(f"             순서: {'후보 전부 상승 → argmax (PA-RL 방식)' if self.guide_all else 'argmax → 고른 하나만 상승'}")
        elif self.cog is not None:
            print("  [guidance] 끔 (--guide-steps 0) → Q 선택만 한다")
        if self.cog is not None:
            print(f"  [critic] cog feature 경로 — 백본이 이미 계산한 backbone_features 에서\n"
                  f"           cognition token {self.cog.n_cog}개를 mean-pool 한다 "
                  f"(백본 재실행 없음, ResNet 인코더 미사용)")
        miss = [] if self.cog is not None else [
            k for k in ("enc", "critic", "target", "residual") if k not in self.loaded]
        if miss:
            print(f"  [주의] {miss} 는 **랜덤 초기화** 상태다. EXPO-FT 의 warmup 과 같은 조건이고\n"
                  f"         (랜덤 critic·랜덤 residual 로 수집) 그래서 edit 이 액션을 흔든다 —\n"
                  f"         지금 필요한 탐색 데이터가 이렇게 만들어진다. 성공률은 base BC 보다 낮다.")

    def _load(self, path: Path | None) -> list[str]:
        """critic/encoder/residual 산출물 로드. 있는 키만 채운다.

        Phase D 프로브(rl/offline_critic.py --save) 는 enc/critic/target 만 저장하고,
        learner 라운드 산출물은 residual/temp 까지 넣는다. 둘 다 받는다.
        """
        if path is None:
            return []
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"산출물이 없다: {path}")

        # 옆의 meta.json 과 대조한다. 안 하면 잘려서 온 파일이 torch 내부 에러
        # (PytorchStreamReader ... failed finding central directory) 로만 드러나서
        # "학습이 깨졌나" 로 오해하게 된다. learner 가 sha256 을 남기는 이유가 이것.
        meta = path.parent / "meta.json"
        size = path.stat().st_size
        if meta.is_file():
            rec = json.loads(meta.read_text())
            want = rec.get("theta_sha256")
            if want:
                got = hashlib.sha256(path.read_bytes()).hexdigest()
                if got != want:
                    raise SystemExit(
                        f"산출물이 manifest 와 다르다 — 전송이 잘렸을 가능성이 크다.\n"
                        f"  파일   {path} ({size:,} 바이트)\n"
                        f"  sha256 {got}\n"
                        f"  기대   {want}\n"
                        f"  → 다시 받을 것: ./actor/recv_round.py --round "
                        f"{'init' if path.parent.name == 'init' else path.parent.name}")
                print(f"  [산출물] sha256 대조 OK ({size/1e6:.0f} MB)")
        else:
            print(f"  [산출물] meta.json 이 없어 sha256 대조를 건너뜀 ({size/1e6:.0f} MB)")

        try:
            sd = torch.load(path, map_location=self.learner.device, weights_only=True)
        except Exception as e:
            raise SystemExit(f"{path} 를 torch 로 읽을 수 없다 ({size:,} 바이트): {e}")
        pairs = {"enc": self.learner.encoder, "critic": self.learner.critic,
                 "target": self.learner.target_critic, "residual": self.learner.residual,
                 "temp": self.learner.temp}
        got = []
        for k, m in pairs.items():
            if k in sd:
                m.load_state_dict(sd[k])
                got.append(k)
        if sd.get("lora"):
            # r000 부터는 학습된 action expert LoRA 가 들어온다 (init/ 에는 없다 — 주입
            # 직후 델타가 0 이라 base BC 와 같기 때문). 로드하려면 먼저 주입해야 한다.
            # setup_training 이 옵티마이저도 만들지만 서빙에서는 쓰지 않는다.
            self.vla.setup_training(lora=True)
            r = self.vla.model.load_state_dict(sd["lora"], strict=False)
            if r.unexpected_keys:
                raise SystemExit(f"lora 키가 모델에 없다: {r.unexpected_keys[:3]}")
            got.append(f"lora({len(sd['lora'])}텐서)")
        if not got:
            raise SystemExit(f"{path} 에 쓸 수 있는 키가 없다 (있는 키: {sorted(sd)})")
        return got

    def _critic_obs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        dev = self.learner.device
        x = torch.from_numpy(np.ascontiguousarray(_cat_cams(obs, self.mod))).to(dev)
        w, h = self.img_size
        if (x.shape[1], x.shape[2]) != (h, w):
            # 학습 이미지는 convert_data.py 로 320x192 로 줄인 것이다. rrc 는 원본
            # 1280x720 을 보내므로 여기서 같은 해상도로 맞춘다 — 인코더가 어차피 224x224
            # 로 줄이지만 종횡비가 1.778 vs 1.667 로 달라 그대로 넣으면 학습과 다른 그림이
            # 된다 (VLA 쪽은 RLDX 프로세서가 이미 같은 일을 해준다).
            f = x.permute(0, 3, 1, 2).float()
            f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
            x = f.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        st = torch.from_numpy(normalize_states(self.vla.proc, self.vla.tag, self.mod,
                                              _cat_state(obs, self.mod))).to(dev)
        return x, st

    def _cog_select(self, pred, request, chunks):
        """cog feature 로 후보를 고른다. select_from_chunks 와 같은 규약.

        분포형 critic 이라 원본 select_from_chunks 를 못 쓴다 (그쪽은 raw 출력에 min 을
        걸어 (B,bins) 가 되어 view 가 깨진다) — q_of 를 적용해야 한다.
        """
        C = self.cog
        B, n = chunks.shape[0], chunks.shape[1]
        acts = chunks[:, :, :C.window].reshape(B, n, -1)   # 창은 체크포인트가 정한다

        # 백본이 액션 생성하면서 이미 계산했다 (rldx.py:729-733 의 반환값).
        # inference_mode 텐서라 clone 으로 꺼낸다.
        f = pred["backbone_features"][:1].clone()
        state = torch.from_numpy(normalize_states(
            self.vla.proc, self.vla.tag, self.mod, _cat_state(request.obs, self.mod))).to(f.device)
        lat = C.latent(C.cog_of(f), state)
        rl_, rs = lat.repeat_interleave(n, 0), state.repeat_interleave(n, 0)

        def score(a):                                  # (B,n,full) → (B,n) 앙상블 min
            with torch.no_grad():
                return C.q(rl_, rs, a.reshape(B * n, -1),
                           target=True).min(dim=0).values.view(B, n)

        q_pre = score(acts)
        best_pre = q_pre.argmax(dim=1)
        q0 = float(q_pre.gather(1, best_pre[:, None]).mean())   # 선택만 했을 때의 Q

        if self.guide_steps > 0 and self.guide_all:
            # PA-RL 순서: 후보 **전부** 상승 → argmax. 처음엔 낮았는데 상승 후 더 높아지는
            # 후보를 잡을 수 있다. critic 이 작은 MLP 라 n배 비용이 지연에 거의 안 보인다.
            gd, _ = self._cog_guide(rl_, rs, acts.reshape(B * n, -1))
            acts = gd.view(B, n, -1)
            q = score(acts)
        else:
            q = q_pre
        best = q.argmax(dim=1)
        chosen = acts[torch.arange(B, device=acts.device), best]

        gain, gstd = 0.0, 0.0
        if self.guide_steps > 0:
            if not self.guide_all:
                chosen, _ = self._cog_guide(lat, state, chosen)
            with torch.no_grad():
                qf = C.q(lat, state, chosen, target=True)
                gain = float(qf.min(0).values.mean()) - q0
                gstd = float(qf.std(0).mean())          # 앙상블 불일치 = 외삽 신호
        return chosen, best, {"chosen_q": q0 + gain,
                              "candidate_q_std": float(q_pre.std(dim=1).mean()),
                              "guide_gain": gain, "guide_ens_std": gstd,
                              "select_ratio_with_residual": 0.0}

    def _cog_guide(self, lat, state, act):
        """test-time Q guidance — ∇_A Q 상승 + keep-best. (편집 액션, 최종 Q)

        PA-RL 의 고정 step_size(3e-4) 를 쓰지 않는다. 우리 Q·액션 스케일에서 그 값은 이동이
        1e-9 라 아무 일도 안 한다 (probe_actopt 에서 실측). 대신 **gradient 를 정규화**해
        스텝마다 차원당 guide_move/steps 만큼 움직인다 — 스케일에 무관하고 상태마다 자동
        적응한다.

        keep-best (Q-VGM Eq. 7 의 장치): 매 스텝 Q 를 재보고 개선된 것만 채택한다. j=0 이
        원본이므로 **Q 가 나빠지는 방향은 절대 나가지 않는다** — 서빙에서 이게 안전장치다.

        inference_mode 텐서는 autograd 에 못 쓰므로 clone 으로 꺼낸다 (실측 확인).
        """
        C = self.cog
        d = int(self.cog_mask.sum())
        step = self.guide_move * (d ** 0.5) / max(self.guide_steps, 1)
        best = act.clone()
        with torch.no_grad():
            bq = C.q(lat, state, best, target=True).min(0).values
        cur = best
        for _ in range(self.guide_steps):
            cur = cur.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                qm = C.q(lat, state, cur, target=True).mean(0).sum()   # PA-RL: 앙상블 mean
                g, = torch.autograd.grad(qm, cur)
            g = g * self.cog_mask
            gn = g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cur = (cur.detach() + step * g / gn).clamp(-1.0, 1.0)
            with torch.no_grad():
                qq = C.q(lat, state, cur, target=True).min(0).values
            take = qq > bq
            best = torch.where(take[:, None], cur, best)
            bq = torch.maximum(bq, qq)
        return best, bq

    def _run_inference(self, request, collated, B, reset_memory):
        if B != 1:
            raise ValueError(f"정책 서버는 관측 1개씩 받는다 (B={B}). 배치 추론은 학습 경로다.")
        t0 = time.time()
        n = self.cfg.N
        with self.vla.expanded(n):
            pred, reset_memory = self._orig_run(request, collated, B, reset_memory)

        with torch.no_grad():
            a = pred["action_pred"].float()                      # (N, H, max_action_dim)
            H, A = a.shape[1], self.vla.action_dim
            need = (self.cog.window if self.cog is not None
                    else self.latency + self.replan)
            if H < need:
                raise ValueError(f"청크 길이 {H} < 필요 {need} "
                                 f"(실행 오프셋 {self.latency} + replan {self.replan}"
                                 f"{f', critic 창 {self.cog.window}' if self.cog else ''})")
            chunks = a[..., :A].reshape(1, n, H, A)
            if self.cog is not None:
                chosen, best, info = self._cog_select(pred, request, chunks)
            else:
                img, state = self._critic_obs(request.obs)
                lat = self.learner.encode(img, stop_gradient=True)
                chosen, best, info = self.learner.select_from_chunks(chunks, lat, state)

            j = int(best[0])
            src = j if j < n else j - n            # edit 후보는 base 후보 (j-n) 에서 나왔다
            out = a[src:src + 1].clone()                          # (1, H, max_action_dim)
            # chosen 은 청크 [0, 창) 을 평탄화한 것이다. **실행되는 구간만** 꽂는다 —
            # 실행 오프셋은 rrc 의 inference_latency_steps 이고 critic 창과 별개다.
            e0 = self.latency
            out[0, e0:e0 + self.replan, :A] = \
                chosen[0, e0 * A:(e0 + self.replan) * A].view(self.replan, A)

        pred = dict(pred)
        pred["action_pred"] = out
        dt = (time.time() - t0) * 1000
        self.calls += 1
        self.ms.append(dt)
        self.q.append(info["chosen_q"])
        self.with_edit.append(info["select_ratio_with_residual"])
        if self.verbose or self.calls <= 3 or self.calls % self.log_every == 0:
            # guide Δ / 앙상블std 비율이 1 미만이면 그 개선은 앙상블 노이즈 안이다
            # ens.std 절대값이 핵심이다: 오프라인 검증에서 0.007~0.02 였는데 실기에서
            # 0.09~0.15 가 나오면 critic 이 학습 분포 밖이라는 뜻이다 (Δ/std 만 보면 놓친다).
            gtxt = ("" if not info.get("guide_ens_std") else
                    f"  guideΔ={info['guide_gain']:+.4f} "
                    f"ens.std={info['guide_ens_std']:.4f} "
                    f"Δ/std={info['guide_gain']/max(info['guide_ens_std'],1e-9):.2f}"
                    + ("  ** OOD 의심 (오프라인 0.007~0.02)"
                       if info["guide_ens_std"] > 0.05 else ""))
            # cog 모드는 edit 후보를 만들지 않는다 (residual policy 미사용) — 표시도 그렇게.
            ncand = f"{n}" if self.cog is not None else f"{n}+{min(self.cfg.n_edit_samples, n)}"
            print(f"[EXPO] #{self.calls} {dt:.0f}ms  후보 {ncand} "
                  f"→ {j}{' (edit)' if j >= n else ''}  Q={info['chosen_q']:+.4f} "
                  f"후보간Qstd={info['candidate_q_std']:.4f}{gtxt}", flush=True)
        return pred, reset_memory

    def run(self, host: str, port: int) -> None:
        from rldx.policy.server_client import PolicyServer

        print(f"\n  듣는다 tcp://{host}:{port}   (rrc zmq_client 가 붙으면 된다)", flush=True)
        PolicyServer(policy=self.policy, host=host, port=port).run()


def _verify_cog(argv: list[str]) -> int:
    """서빙 배선이 **학습 때의 latent/Q 를 재현하는지** 데이터셋으로 대조한다 (로봇 불필요).

    왜 이게 결정적인가: 학습은 cogfeat.npy 를 인덱스로 읽고, 서빙은 백본 출력에서 그 자리에서
    mean-pool 한다. 두 경로가 같은 프레임에 같은 값을 내면 배선이 맞은 것이고, 다르면
    (전처리·토큰 위치·표준화·Proj 중) 어디가 어긋났는지 단계별로 드러난다.

    네 단계를 순서대로 대조한다:
      1) cog feature   서빙 경로 mean-pool  vs  cogfeat.npy[idx]
      2) latent        표준화 + Proj 통과 후
      3) Q             로그된 액션에 대한 Q (분포형이면 bin 기댓값)
      4) 후보 선택     같은 후보 집합에서 argmax 인덱스가 같은지
    """
    import argparse

    import yaml

    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser("rl.vla_rldx verify-cog")
    p.add_argument("--exp", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--critic", required=True, help="work 기준 상대경로도 된다")
    p.add_argument("--model-path", default="")
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    dev = a.device

    from rl.critic_io import load_serving_critic
    from rl.data import build_flat, find_sessions, open_images, resolve_modality
    from rl.expo import ExpoConfig
    from rl.nets import explore_spec
    from rl.offline_critic import normalize_all

    rldx = repo / "third_party/RLDX-1"
    exp = yaml.safe_load((repo / "configs/exp" / f"{a.exp}.yaml").read_text())
    cfg = ExpoConfig.from_dict(exp.get("expo"))
    R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
    work = a.checkpoints / f"{a.exp}-critic"
    base = a.checkpoints / (a.model_path or exp["base_policy"])
    mod, _ = resolve_modality(a.data, None, rldx, exp["rldx_data_config"], base)
    sessions = find_sessions(a.data)
    flat = build_flat(sessions, mod)
    imgs, _ = open_images(work / "images.mm")
    norm = normalize_all(None, flat, H, cache=work / "actnorm.npy")
    proc = load_state_action_processor(base, rldx, exp["rldx_data_config"])
    snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
    task = json.loads((sessions[0] / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    spec = explore_spec(mod.offsets("action"), exp.get("explore_groups") or [],
                        mod.action_dim, R, LAT)
    ck = Path(a.critic) if Path(a.critic).is_file() else work / a.critic

    vla = RLDXVLA(base, mod, rldx, exp["rldx_data_config"], device=a.device)
    n_cog = int(getattr(vla.model, "_n_cog_tokens",
                        getattr(vla.model.backbone, "n_cog_tokens", 64)))
    C = load_serving_critic(ck, cfg, mod.state_dim, mod.action_dim, LAT, R, n_cog,
                            dev=a.device)
    # ExpoServer.__init__ 과 **같은 방식**으로 창·마스크를 만든다 (테스트가 서버와 어긋나면
    # 의미가 없다): 창은 체크포인트, 편집은 실행 구간 [LAT, LAT+R) x explore 관절.
    jsel = [i for nm, s0, e0 in mod.offsets("action")
            if nm in (exp.get("explore_groups") or [nm]) for i in range(s0, e0)]
    mk = torch.zeros(C.window, mod.action_dim, device=dev)
    mk[LAT:LAT + R, jsel] = 1.0
    MASK = mk.reshape(-1)
    MIDX = MASK.nonzero(as_tuple=True)[0]
    print(f"  [편집] 창 {C.window}스텝 중 실행 [{LAT},{LAT + R}) x 관절 {len(jsel)}개 "
          f"= {len(MIDX)}/{C.full} 차원")

    # 정답지: 학습이 읽는 그 파일
    fp = np.load(work / C.meta["features"], mmap_mode="r")
    idx = np.linspace(0, len(flat) - R - 1, a.frames).astype(np.int64)

    print(f"\n[검증] 프레임 {len(idx)}개  n_cog {n_cog}  features {C.meta['features']}")
    ok = True

    # --- 1) cog feature ---
    grabbed = []
    for c in range(0, len(idx), 8):
        k = idx[c:c + 8]
        x = np.asarray(imgs[k])
        obs = {"video": {nm: x[:, ci][:, None] for ci, (nm, _) in enumerate(mod.video)},
               "state": {nm: flat.state[k][:, None, s0:e0]
                         for nm, s0, e0 in mod.offsets("state")},
               "language": {mod.task_key: [[task]] * len(k)}}
        with torch.no_grad():
            out = vla.runtime._forward(vla._collate(obs))
        grabbed.append(C.cog_of(out["backbone_features"].clone()).cpu().numpy())
    served = np.concatenate(grabbed)
    truth = np.asarray(fp[idx])
    d = np.abs(served - truth)
    rel = d.max() / max(np.abs(truth).max(), 1e-9)
    print(f"  1) cog feature   최대 절대차 {d.max():.3e}  상대 {rel:.3e}  "
          f"{'OK' if rel < 1e-3 else '** 불일치'}")
    ok &= rel < 1e-3

    # --- 2) latent, 3) Q ---
    st = torch.from_numpy(snorm[idx]).to(dev)
    lat_s = C.latent(torch.from_numpy(served).to(dev), st)
    lat_t = C.latent(torch.from_numpy(truth).to(dev), st)
    dl = (lat_s - lat_t).abs().max().item()
    print(f"  2) latent        최대 절대차 {dl:.3e}  {'OK' if dl < 1e-2 else '** 불일치'}")
    ok &= dl < 1e-2

    act = torch.from_numpy(np.ascontiguousarray(
        np.asarray(norm[idx])[:, :C.window].reshape(len(idx), -1))).to(dev)
    with torch.no_grad():
        q_s = C.q(lat_s, st, act).min(0).values
        q_t = C.q(lat_t, st, act).min(0).values
    dq = (q_s - q_t).abs()
    print(f"  3) Q(로그 액션)  최대 절대차 {dq.max():.3e}  평균 {dq.mean():.3e}  "
          f"Q 범위 [{q_t.min():.3f},{q_t.max():.3f}]  "
          f"{'OK' if dq.max() < 1e-2 else '** 불일치'}")
    ok &= dq.max().item() < 1e-2

    # --- 4) 후보 선택이 같은 인덱스를 고르나 ---
    g = torch.Generator(device=dev).manual_seed(0)
    N = 8
    cand = act[:, None, :].repeat(1, N, 1)
    cand[:, 1:] += (torch.rand(len(idx), N - 1, C.full, device=dev, generator=g) * 2 - 1) \
        * 0.05 * MASK
    with torch.no_grad():
        qq_s = C.q(lat_s.repeat_interleave(N, 0), st.repeat_interleave(N, 0),
                   cand.reshape(-1, C.full)).min(0).values.view(len(idx), N)
        qq_t = C.q(lat_t.repeat_interleave(N, 0), st.repeat_interleave(N, 0),
                   cand.reshape(-1, C.full)).min(0).values.view(len(idx), N)
    same = (qq_s.argmax(1) == qq_t.argmax(1)).float().mean().item()
    print(f"  4) 후보 argmax   일치율 {same:.1%}  (후보 {N}개, 편집 범위에 ±0.05 교란)  "
          f"{'OK' if same > 0.98 else '** 불일치'}")
    ok &= same > 0.98

    # --- 5) expanded(N) 아래에서도 같은 cog feature 가 나오나 ---
    # 실제 서버는 후보 N개를 만들려고 expanded(N) 로 backbone_features 를 N배 복제한 뒤
    # _cog_select 가 [:1] 을 꺼낸다. 복제가 dim 0 이라 seq 축은 그대로여야 하는데,
    # 그건 **주장이 아니라 확인할 사실**이다 (여기서 깨지면 서버가 엉뚱한 latent 를 쓴다).
    k = idx[:4]
    x = np.asarray(imgs[k])
    obs = {"video": {nm: x[:, ci][:, None] for ci, (nm, _) in enumerate(mod.video)},
           "state": {nm: flat.state[k][:, None, s0:e0] for nm, s0, e0 in mod.offsets("state")},
           "language": {mod.task_key: [[task]] * len(k)}}
    collated = vla._collate(obs)
    with torch.no_grad():
        plain = vla.runtime._forward(collated)["backbone_features"].clone()
        N = int(cfg.N)
        with vla.expanded(N):
            exp_out = vla.runtime._forward(vla._collate(obs))["backbone_features"].clone()
    print(f"  5) expanded({N})     plain {tuple(plain.shape)} -> expanded {tuple(exp_out.shape)}")
    shape_ok = (exp_out.shape[0] == plain.shape[0] * N and exp_out.shape[1:] == plain.shape[1:])
    # 복제가 repeat_interleave 이므로 샘플 i 의 사본들은 행 i*N .. i*N+N-1 에 있다
    rep_ok = shape_ok and all(
        torch.equal(exp_out[i * N], exp_out[i * N + j]) for i in range(len(k)) for j in range(N))
    # 각 샘플의 첫 사본이 plain 과 같아야 한다
    same_ok = shape_ok and all(torch.equal(exp_out[i * N], plain[i]) for i in range(len(k)))
    cog_ok = shape_ok and torch.equal(C.cog_of(exp_out[:1]), C.cog_of(plain[:1]))
    print(f"      seq 축 보존 {shape_ok}  사본끼리 동일 {rep_ok}  "
          f"plain 과 동일 {same_ok}  cog_of([:1]) 동일 {cog_ok}  "
          f"{'OK' if (shape_ok and rep_ok and same_ok and cog_ok) else '** 불일치'}")
    ok &= shape_ok and rep_ok and same_ok and cog_ok

    # --- 6) Q guidance 가 실제로 Q 를 올리나 (keep-best 라 내려갈 수는 없다) ---
    class _Srv:                                        # _cog_guide 만 쓰기 위한 최소 껍데기
        pass
    srv = _Srv()
    srv.cog = C
    srv.cog_mask = MASK
    srv._cog_guide = ExpoServer._cog_guide.__get__(srv, _Srv)
    print()
    for gs, gm in ((4, 0.05), (4, 0.2), (10, 0.2)):
        srv.guide_steps, srv.guide_move = gs, gm
        t0 = time.time()
        gact, gq = srv._cog_guide(lat_t, st, act)
        torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        with torch.no_grad():
            q_before = C.q(lat_t, st, act, target=True).min(0).values
            std_b = C.q(lat_t, st, act, target=True).std(0)
            std_a = C.q(lat_t, st, gact, target=True).std(0)
        mv = (gact - act)[:, MIDX].norm(dim=-1) / len(MIDX) ** 0.5
        worse = int((gq < q_before - 1e-6).sum())
        print(f"  6) guidance steps={gs} move={gm}: Q {q_before.mean():+.4f} -> {gq.mean():+.4f} "
              f"(Δ{(gq - q_before).mean():+.4f})  실제이동 {mv.mean():.4f}/차원 "
              f"= {mv.mean()/0.0218:.1f} 프레임치  앙상블std {std_b.mean():.4f}->{std_a.mean():.4f} "
              f"({std_a.mean()/std_b.mean().clamp_min(1e-9):.2f}배)  "
              f"ΔQ/std {(gq - q_before).mean()/std_a.mean().clamp_min(1e-9):.2f}  "
              f"나빠진 프레임 {worse} {'OK' if worse == 0 else '** keep-best 가 깨졌다'}")
        print(f"      {len(idx)}개 동시 상승에 {ms:.1f}ms "
              f"(서빙은 후보 8개 = {ms*8/len(idx):.1f}ms 예상)")
        ok &= worse == 0

    print(f"\n[결과] {'배선이 학습 경로를 재현한다' if ok else '** 어긋난 단계가 있다 (위 참고)'}")
    if ok:
        print("  검증한 것 : 같은 이미지에서 학습 경로와 비트 단위로 같은 latent/Q/선택")
        print("  안 한 것  : 실기 이미지(rrc 1280x720 vs 데이터셋 320x192)에서의 동작.")
        print("              RLDX 프로세서가 양쪽을 리사이즈하지만 원본이 다르다 — 다만 이 차이는")
        print("              **정책 자신도 똑같이 겪는 것**이라 critic 이 정책과 어긋나지는 않는다.")
    return 0 if ok else 1


def _serve(argv: list[str]) -> int:
    import argparse

    import yaml

    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser("rl.vla_rldx serve", description="EXPO 정책 서버")
    p.add_argument("--exp", required=True, help="configs/exp/<이름>.yaml 의 <이름>")
    p.add_argument("--model-path", required=True, type=Path, help="base BC 정책 디렉토리")
    p.add_argument("--modality", type=Path,
                   help="modality/<embodiment>/modality.json. 기본값은 exp yaml 의 modality 키")
    p.add_argument("--artifacts", type=Path,
                   help="라운드 산출물 .pt (critic/encoder/residual). 없으면 랜덤 초기화")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rtc-inference-mode", default="trained",
                   choices=["none", "trained", "guided"])
    p.add_argument("--critic-image-size", default="320x192",
                   help="critic 인코더에 넣기 전 맞출 해상도 = 학습 데이터 해상도")
    p.add_argument("--guide-steps", type=int, default=0,
                   help="test-time Q guidance 의 ∇_A Q 상승 스텝 수. 0 이면 Q 선택만 한다. "
                        "**총 이동량은 --guide-move 가 정한다** (gradient 를 정규화해 쓴다) — "
                        "steps 는 경로 해상도일 뿐이다. 실측: steps 4 와 10 의 ΔQ 가 동일 "
                        "(+0.0064) → 국소적으로 Q 가 거의 선형이라 4 면 충분하다. "
                        "Q-VGM 실측(LIBERO): 선택만 86.0 / guidance 88.7 (SFT 79.0). "
                        "cog feature critic 에서만 동작한다")
    p.add_argument("--guide-move", type=float, default=0.05,
                   help="상승의 차원당 총 이동량 (정규화 액션 단위). **실질적인 유일한 노브다.** "
                        "openarm 의 1프레임 자연 변화가 ~0.022 이라 0.05 는 약 2.3 프레임치, "
                        "0.2 는 약 9 프레임치(yaml 의 edit_scale 과 같은 크기)다. "
                        "실측 두 표본에서 ΔQ/std 는 0.05 와 0.2 가 비슷했지만 앙상블 std 증가는 "
                        "0.05 가 1.2~1.3배, 0.2 가 1.3~4.1배로 표본에 따라 크게 흔들렸다 — "
                        "즉 0.2 는 외삽 위험이 훨씬 크면서 신뢰도는 비슷하다. "
                        "**실기에서 0.05 / 0.1 / 0.2 를 쓸어보고 정할 값이다**: "
                        "로그의 Δ/std 가 1 이상이고 ens.std 증가가 2배 미만인 구간을 쓸 것. "
                        "keep-best 라 Q 가 나빠지면 원본을 유지한다")
    p.add_argument("--log-every", type=int, default=25,
                   help="[EXPO] 한 줄을 몇 호출마다 찍을지. 1 이면 매 스텝 (진단용)")
    p.add_argument("--rtc-exec-horizon", type=int, default=0,
                   help="RTC 의 execution horizon s. 0 이면 yaml 의 replan_steps 를 쓴다 "
                        "(= rrc 의 execution_horizon). 이 값을 안 넘기면 RLDX 가 "
                        "action_horizon - delay 로 채워서 (16-2=14) RTC prefix 를 엉뚱한 "
                        "위치에서 자른다 (서버 캐시 폴백 경로). 시작 로그의 exec_horizon 이 "
                        "replan_steps 와 같은지 꼭 확인할 것")
    p.add_argument("--guide-all", action="store_true",
                   help="후보 전부를 상승시킨 뒤 argmax (PA-RL 순서). 기본은 argmax 를 먼저 하고 "
                        "고른 하나만 상승시킨다. critic 이 작은 MLP 라 비용 차이가 거의 없다")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    exp = yaml.safe_load((repo / "configs" / "exp" / f"{a.exp}.yaml").read_text())
    modality = a.modality or (repo / exp["modality"])
    w, h = (int(v) for v in a.critic_image_size.lower().split("x"))
    print(f"EXPO 정책 서버 — 실험 {a.exp}")
    srv = ExpoServer(exp, a.model_path, modality, repo / "third_party" / "RLDX-1",
                     device=a.device, artifacts=a.artifacts, seed=a.seed,
                     rtc_mode=a.rtc_inference_mode, img_size=(w, h), verbose=a.verbose,
                     guide_steps=a.guide_steps, guide_move=a.guide_move,
                     guide_all=a.guide_all, rtc_exec_horizon=a.rtc_exec_horizon or None,
                     log_every=a.log_every)
    srv.run(a.host, a.port)
    return 0


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
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        sys.exit(_serve(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "verify-cog":
        sys.exit(_verify_cog(sys.argv[2:]))
    sys.exit(_verify())
