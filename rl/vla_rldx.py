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

from rl.data import Modality, rldx_layout
from rl.expo import VLA


class RLDXVLA(VLA):
    def __init__(self, model_path: Path | str, mod: Modality, rldx_root: Path | str,
                 rldx_config: str, device: str = "cuda", rtc_inference_mode: str = "none",
                 rtc_inference_delay: int | None = None):
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.policy.rldx_policy import RLDXPolicy

        # 등록 config 를 먼저 로드해야 태그가 존재한다 (같은 태그로 두 번 등록되지 않도록
        # rldx_layout 이 이미 로드했다면 그 결과를 재사용한다).
        tag = mod.embodiment_tag or rldx_layout(Path(rldx_root), rldx_config)[0]
        self.tag = tag
        self.mod = mod
        self.policy = RLDXPolicy(embodiment_tag=EmbodimentTag(tag), model_path=str(model_path),
                                 device=device, rtc_inference_mode=rtc_inference_mode,
                                 rtc_inference_delay=rtc_inference_delay)
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
                 verbose: bool = False):
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
                           rtc_inference_mode=rtc_mode, rtc_inference_delay=self.latency)
        spec = explore_spec(mod.offsets("action"), exp.get("explore_groups") or [],
                            mod.action_dim, self.replan)
        self.cfg = ExpoConfig.from_dict(exp.get("expo"))
        self.learner = EXPOLearner(self.vla, spec, mod.state_dim, mod.n_cams, self.replan,
                                   self.cfg, device=device, seed=seed, latency=self.latency)
        for m in (self.learner.encoder, self.learner.critic, self.learner.target_critic,
                  self.learner.residual):
            m.eval()
        self.loaded = self._load(artifacts)

        self.policy, self.runtime = self.vla.policy, self.vla.runtime
        if self.runtime.use_memory:
            raise SystemExit(
                "memory 모델은 아직 지원하지 않는다 — 후보 N개 확장이 memory scratchpad 의\n"
                "  배치(B=1)와 어긋난다. base 정책을 memory 없이 뽑거나 확장 경로를 고쳐야 한다.")
        self._orig_run = self.runtime._run_inference
        self.runtime._run_inference = self._run_inference
        self.calls, self.ms, self.q, self.with_edit = 0, [], [], []

        print(f"  [정책] {Path(model_path).name}")
        print(f"  [태그] {self.vla.tag}  state_dim={mod.state_dim} action_dim={mod.action_dim} "
              f"cams={mod.n_cams}")
        print(f"  [청크] action_horizon={self.vla.action_horizon} latency={self.latency} "
              f"replan={self.replan} → critic 이 보는 구간 "
              f"[{self.latency},{self.latency + self.replan})")
        print(f"  [탐색] {list(spec.groups)}  활성 {spec.active_dim}/{mod.action_dim} 차원")
        print(f"  [선택] N={self.cfg.N} + edit={self.cfg.n_edit_samples} "
              f"(edit_scale={self.cfg.edit_scale}) → target critic argmax")
        print(f"  [critic 이미지] {img_size[0]}x{img_size[1]} 로 맞춘 뒤 인코더가 224 로 줄인다")
        if self.loaded:
            print(f"  [산출물] {artifacts} 에서 {self.loaded} 로드")
        miss = [k for k in ("enc", "critic", "target", "residual") if k not in self.loaded]
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
        st = torch.from_numpy(_cat_state(obs, self.mod)).to(dev)
        return x, st

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
            if H < self.latency + self.replan:
                raise ValueError(f"청크 길이 {H} < latency({self.latency})+replan({self.replan})")
            chunks = a[..., :A].reshape(1, n, H, A)
            img, state = self._critic_obs(request.obs)
            lat = self.learner.encode(img, stop_gradient=True)
            chosen, best, info = self.learner.select_from_chunks(chunks, lat, state)

            j = int(best[0])
            src = j if j < n else j - n            # edit 후보는 base 후보 (j-n) 에서 나왔다
            out = a[src:src + 1].clone()                          # (1, H, max_action_dim)
            out[0, self.latency:self.latency + self.replan, :A] = chosen[0].view(self.replan, A)

        pred = dict(pred)
        pred["action_pred"] = out
        dt = (time.time() - t0) * 1000
        self.calls += 1
        self.ms.append(dt)
        self.q.append(info["chosen_q"])
        self.with_edit.append(info["select_ratio_with_residual"])
        if self.verbose or self.calls <= 3 or self.calls % 25 == 0:
            print(f"[EXPO] #{self.calls} {dt:.0f}ms  후보 {n}+{min(self.cfg.n_edit_samples, n)} "
                  f"→ {j}{' (edit)' if j >= n else ''}  Q={info['chosen_q']:+.3f} "
                  f"후보간 Q std={info['candidate_q_std']:.3f}", flush=True)
        return pred, reset_memory

    def run(self, host: str, port: int) -> None:
        from rldx.policy.server_client import PolicyServer

        print(f"\n  듣는다 tcp://{host}:{port}   (rrc zmq_client 가 붙으면 된다)", flush=True)
        PolicyServer(policy=self.policy, host=host, port=port).run()


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
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    exp = yaml.safe_load((repo / "configs" / "exp" / f"{a.exp}.yaml").read_text())
    modality = a.modality or (repo / exp["modality"])
    w, h = (int(v) for v in a.critic_image_size.lower().split("x"))
    print(f"EXPO 정책 서버 — 실험 {a.exp}")
    srv = ExpoServer(exp, a.model_path, modality, repo / "third_party" / "RLDX-1",
                     device=a.device, artifacts=a.artifacts, seed=a.seed,
                     rtc_mode=a.rtc_inference_mode, img_size=(w, h), verbose=a.verbose)
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
    sys.exit(_verify())
