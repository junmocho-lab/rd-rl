#!/usr/bin/env python3
"""**오라클 선택** — 후보를 실제 시뮬레이터로 굴려보고 고른다. critic 이 필요 없다.

무엇에 답하는가: "test-time 후보 선택으로 성공률을 올릴 수 있는가" 의 **상한**.

지금까지 우리는 critic 이 후보를 구분하지 못하는 것을 반복 확인했다 (후보간 Qstd 0.0004,
sel32 70% vs BC 77%). 그런데 그것이 (a) critic 이 못 배운 것인지 (b) 애초에 고를 것이 없는
것인지 구분하지 못했다. 오라클은 그 구분을 데이터 수집 없이 짓는다:

  오라클 ~100%  ->  선택은 원리적으로 작동한다. 문제는 순전히 critic 이고 데이터를 늘릴 가치가 있다
  오라클  ~77%  ->  어떤 critic 도 못 한다. 이 태스크에서 청크 선택은 불가능하다

방법: 매 replan 마다 정책에서 후보 N 개를 받고, 각 후보를 실제로 20스텝 실행해 본 뒤
상태를 되돌리고, 점수가 가장 높은 후보를 실제로 커밋한다. 환경이 결정적이라 되돌리기가
정확하다 (mj_getState/mj_setState + 파이썬 쪽 카운터 _nail_depth).

점수(--score):
  depth   20스텝 동안의 못 삽입량 (탐욕적). 못을 박는 것이 유일한 진척이므로 자연스럽다
  oracle  20스텝 실행 후 **에피소드 끝까지** base 정책으로 이어 굴려 성공 여부를 본다
          (진짜 오라클이지만 N배 느리다)
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_raw_to_rldx import DEFAULT_CAMERA_KEYS, TASK_CAMERAS, TASK_PROMPTS  # noqa: E402
from rollout_dexjoco import (  # noqa: E402
    Config as RolloutConfig, PolicyClient, action_25_to_env, build_env,
    chunk_from_response, env_state_to_25, obs_to_request,
)


@dataclass
class Config:
    task: str = "hammer_nail"
    host: str = "127.0.0.1"
    port: int = 20421
    """후보를 N개 내주는 서버 (expo.N 이 후보 수)."""
    episodes: int = 10
    seed: int = 0
    fixed_scene: int = -1
    """>=0 이면 그 시드로 장면을 고정한다 (A·B 세팅). -1 이면 에피소드마다 랜덤 (C 세팅).
    공식은 rollout_dexjoco.py 와 같아야 페어 비교가 된다."""
    replan: int = 20
    rtc_delay: int = 5
    max_steps: int = 360
    n_cand: int = 32
    """오라클이 고를 후보 수."""
    sigma: str = "0,0.005,0.01,0.02"
    """후보를 만들 때 정책 청크에 더할 노이즈 표준편차 목록.

    서버는 argmax 된 청크 하나만 돌려주므로 (프로토콜에 후보 전체를 받는 창구가 없다)
    후보군을 여기서 만든다. 실측 근거: 정책의 실제 후보 산포는 차원당 0.0046~0.019 이고
    (RTC prefix 유무에 따라), 오픈루프 리플레이는 노이즈 0.005 에서 성공 40%, 0.01 에서
    10% 로 무너진다. sigma 를 훑으면 **선택이 이득을 내려면 얼마나 다양해야 하는지**가 나온다.
    sigma=0 은 선택이 없는 것과 같아 BC 재현이 되어야 한다 (검증용)."""
    out: Path = Path("/workspace/junmo_cho/dexjoco/rollout/fixed/oracle")


def snapshot(raw):
    n = mujoco.mj_stateSize(raw._model, mujoco.mjtState.mjSTATE_INTEGRATION)
    buf = np.empty(n, np.float64)
    mujoco.mj_getState(raw._model, raw._data, buf, mujoco.mjtState.mjSTATE_INTEGRATION)
    # ★ env_step 을 반드시 포함해야 한다. panda_hammer_nail_env.step 이
    #   self.env_step += 1 ; terminated = self.env_step >= 1000
    # 이므로, 후보 32개를 시험하면 결정 하나에 660 스텝이 올라 두 번째 결정에서
    # 환경이 영구 종료된다 (실측: sigma>0 에서 전부 0% / 깊이 0.0000 / 전부 동점).
    return (buf, float(raw._nail_depth), raw._data.mocap_pos.copy(),
            raw._data.mocap_quat.copy(), list(getattr(raw, "_vz_buf", [])),
            int(getattr(raw, "env_step", 0)))


def restore(raw, s):
    buf, depth, mp, mq, vz, estep = s
    mujoco.mj_setState(raw._model, raw._data, buf, mujoco.mjtState.mjSTATE_INTEGRATION)
    raw._nail_depth = depth
    raw._data.mocap_pos[:] = mp
    raw._data.mocap_quat[:] = mq
    if hasattr(raw, "_vz_buf"):
        raw._vz_buf.clear(); raw._vz_buf.extend(vz)
    raw.env_step = estep
    mujoco.mj_forward(raw._model, raw._data)


def main(cfg: Config) -> None:
    rc = RolloutConfig(task=cfg.task, image_size=256, replan=cfg.replan,
                       rtc_delay=cfg.rtc_delay, host=cfg.host, port=cfg.port)
    env = build_env(rc)
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    while not hasattr(raw, "_nail_depth"):
        raw = raw.env                              # 래퍼를 벗긴다
    client = PolicyClient(cfg.host, cfg.port)
    # 서버 기동(VLA 로딩)이 늦어도 기다린다. 고정 sleep 에 의존하면 3시간 세션이
    # 첫 몇 분에 죽는다 — rollout_dexjoco 와 같은 재시도 루프를 쓴다.
    client.wait_ready()
    cam_keys = dict(zip(DEFAULT_CAMERA_KEYS, TASK_CAMERAS[cfg.task]))
    prompt = TASK_PROMPTS[cfg.task]

    # 후보 시험 중에는 이미지를 쓰지 않는다. replan 당 후보 32개 x 20스텝 = 640 env.step 이고
    # 매 스텝 카메라를 렌더하면 그것이 비용의 대부분이다 — 캐시된 프레임을 돌려주는
    # 껍데기로 바꿔 끈다 (물리는 그대로 돈다). 실제 커밋 구간에서만 다시 켠다.
    _real_render = raw.render
    _cached = {"f": None}

    def _stub_render(*args, **kwargs):
        if _cached["f"] is None:
            _cached["f"] = _real_render()
        return _cached["f"]

    class NoRender:
        def __enter__(self):
            _cached["f"] = _real_render()
            raw.render = _stub_render

        def __exit__(self, *exc):
            raw.render = _real_render

    norender = NoRender()

    # 되돌리기가 정확한지 먼저 확인한다 — 틀리면 오라클 결과 전체가 무의미하다
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    obs, _ = env.reset()
    s0 = snapshot(raw)
    a0 = np.zeros(25); a0[:3] = env_state_to_25(obs["state"])[:3]
    for _ in range(5):
        env.step(action_25_to_env(env_state_to_25(obs["state"])))
    d_after = float(raw._nail_depth)
    restore(raw, s0)
    ok_restore = abs(float(raw._nail_depth) - s0[1]) < 1e-12
    print(f"[복원 검증] nail_depth {d_after:.5f} -> 되돌린 뒤 {float(raw._nail_depth):.5f}  "
          f"{'OK' if ok_restore else '** 실패'}")

    print(f"[기준] BC(같은 장면) 77/100 = 77.0%   |   오픈루프 리플레이 = 100%\n")
    print(f"{'sigma':>7} {'성공':>8} {'평균 깊이':>10} {'전부 동점인 replan':>18}")
    for sg in [float(x) for x in cfg.sigma.split(",")]:
        n_succ, depths, ties, calls = 0, [], 0, 0
        for ep in range(cfg.episodes):
            # ★ 씬 시드 공식을 rollout_dexjoco.py 와 **같게** 맞춘다. 다르면 오라클과
            # BC/eval 이 서로 다른 장면을 보게 되어 같은 장면 페어 비교(McNemar)를 쓸 수
            # 없다. fixed_scene >= 0 이면 그 값으로 고정하는 것도 같은 규약이다.
            sd = (cfg.fixed_scene if cfg.fixed_scene >= 0
                  else (cfg.seed * 1_000_003 + ep)) % (2 ** 31 - 1)
            random.seed(sd)
            np.random.seed(sd)
            obs, _ = env.reset()
            client.reset()
            rng = np.random.default_rng(1000 * ep + int(sg * 1e5))
            success, t = False, 0
            while t < cfg.max_steps and not success:
                req, _ = obs_to_request(obs, cam_keys, prompt, rc.image_size)
                action, _ = client.get_action(req)
                chunk = chunk_from_response(action)              # (H, 25)
                # 후보군: 정책 청크 + 노이즈 (실행 구간에만). sigma=0 이면 후보가 모두 같다
                cands = [chunk] + [
                    np.concatenate([
                        chunk[:cfg.rtc_delay],
                        chunk[cfg.rtc_delay:] + rng.normal(0, sg, chunk[cfg.rtc_delay:].shape),
                    ]) for _ in range(cfg.n_cand - 1)] if sg > 0 else [chunk]
                snap = snapshot(raw)
                best, best_sc, scores = None, -1e9, []
                norender.__enter__()
                for c in cands:
                    restore(raw, snap)
                    d0 = float(raw._nail_depth)
                    hit = False
                    for k in range(cfg.rtc_delay, cfg.rtc_delay + cfg.replan):
                        _o, _r, term, _tr, info = env.step(action_25_to_env(c[k]))
                        if bool(info.get("succeed", False)):
                            hit = True; break
                        if term: break
                    sc = (float(raw._nail_depth) - d0) + (1.0 if hit else 0.0)
                    scores.append(sc)
                    if sc > best_sc:
                        best_sc, best = sc, c
                norender.__exit__()
                calls += 1
                if len(set(np.round(scores, 8))) == 1:
                    ties += 1
                restore(raw, snap)
                for k in range(cfg.rtc_delay, cfg.rtc_delay + cfg.replan):
                    obs, _r, term, _tr, info = env.step(action_25_to_env(best[k]))
                    t += 1
                    if bool(info.get("succeed", False)):
                        success = True; break
                    if term: break
            n_succ += success
            depths.append(float(raw._nail_depth))
        print(f"{sg:7.3f} {n_succ:4d}/{cfg.episodes:<3} {np.mean(depths):10.4f} "
              f"{ties:>10}/{calls:<7}", flush=True)

    print("\n읽는 법: sigma=0 은 선택 없음(BC 재현). sigma 를 올릴수록 후보가 다양해진다.\n"
          "  오라클이 100% 에 접근하면 -> 선택은 원리적으로 작동한다. critic 을 고치면 된다.\n"
          "  오라클도 77% 근처면   -> 어떤 critic 도 못 한다. 청크 선택으로는 불가능하다.\n"
          "  '전부 동점' 비율이 높으면 그 sigma 에서는 후보들이 결과를 안 바꾼다는 뜻이다.")


if __name__ == "__main__":
    main(tyro.cli(Config))
