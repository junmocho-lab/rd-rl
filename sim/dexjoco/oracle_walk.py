#!/usr/bin/env python3
"""오라클 궤적 위를 걸으며 매 replan 마다 BC 후보 32개를 받아 저장한다 (1단계).

목적: "정책이 뽑는 후보들 중에 오라클(=성공 궤적)이 실제로 들어 있는가" 를 본다.
지금까지 우리는 critic 이 후보를 구분 못 한다는 것만 봤을 뿐, 후보 자체가 오라클과
얼마나 떨어져 있는지 잰 적이 없다. 후보 구름 안에 오라클이 없으면 무엇으로 고르든 소용없다.

방법:
  · 환경은 **오라클 에피소드의 액션을 그대로 실행**한다 (결정적이라 정확히 재현된다).
  · 매 replan 경계에서 정책 서버에 관측을 보낸다. RTC prefix 는 오라클이 방금 실행한
    d=5 개 액션을 그대로 준다 — 그래야 정책이 "오라클 궤적 위에 있는 로봇" 으로서
    다음 청크를 뽑는다.
  · 서버는 argmax 청크 하나만 돌려주므로, 후보 32개는 서버의 --dump-obs 가 npz 로 남긴다
    (cog / state / acts). Q 는 2단계에서 오프라인으로 계산한다.

  서버는 --dump-n 을 replan 횟수 이상으로 띄워 둘 것.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_raw_to_rldx import DEFAULT_CAMERA_KEYS, TASK_CAMERAS, TASK_PROMPTS  # noqa: E402
from rollout_dexjoco import (  # noqa: E402
    Config as RolloutConfig, PolicyClient, action_25_to_env, build_env,
    chunk_from_response, env_state_to_25, obs_to_request,
)


@dataclass
class Config:
    session: Path = Path("/workspace/junmo_cho/dexjoco/rollout/fixed/s0_n100")
    episode: int = -1
    """오라클로 쓸 에피소드. -1 이면 가장 빨리 성공한 것."""
    task: str = "hammer_nail"
    host: str = "127.0.0.1"
    port: int = 20421
    seed: int = 0
    replan: int = 20
    rtc_delay: int = 5
    image_size: int = 256
    out: Path = Path("/workspace/junmo_cho/dexjoco/oracle_walk/walk.npz")


def main(cfg: Config) -> None:
    eps = [json.loads(l) for l in (cfg.session / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    if cfg.episode < 0:
        pick = min((e for e in eps if e.get("success")), key=lambda e: e["length"])
    else:
        pick = next(e for e in eps if e["episode_index"] == cfg.episode)
    i = pick["episode_index"]
    df = pd.read_parquet(cfg.session / f"data/chunk-000/episode_{i:06d}.parquet",
                         columns=["action", "action_raw"])
    A_exec = np.vstack([np.asarray(v, np.float32) for v in df["action"]])
    A_raw = np.vstack([np.asarray(v, np.float32) for v in df["action_raw"]])
    T = len(A_raw)
    print(f"[오라클] ep{i}  {T}프레임  성공 {pick['success']}  깊이 {pick.get('final_nail_depth',0):.4f}")

    rc = RolloutConfig(task=cfg.task, image_size=cfg.image_size,
                       replan=cfg.replan, rtc_delay=cfg.rtc_delay)
    env = build_env(rc)
    client = PolicyClient(cfg.host, cfg.port)
    cam_keys = dict(zip(DEFAULT_CAMERA_KEYS, TASK_CAMERAS[cfg.task]))
    prompt = TASK_PROMPTS[cfg.task]

    random.seed(cfg.seed); np.random.seed(cfg.seed)
    obs, _ = env.reset()
    client.reset()

    calls = []                       # 각 replan 에서 (프레임 t, 서버 argmax 청크, 오라클 다음 20액션)
    t = 0
    while t + cfg.replan <= T:
        req, _ = obs_to_request(obs, cam_keys, prompt, cfg.image_size)
        opts = {}
        if t == 0:
            opts["reset_memory"] = [True]
        else:
            # 오라클이 방금 실행한 d 개 액션을 prefix 로 준다 (rollout_dexjoco 와 같은 규약:
            # 정규화 전 정책 출력 action_raw 를 그대로 보낸다)
            opts["action_prefix"] = A_raw[t - cfg.rtc_delay:t].astype(np.float32)[None]
        act_dict, _ = client.get_action(req, opts or None)
        full = chunk_from_response(act_dict)
        start = cfg.rtc_delay if t > 0 else 0
        served = full[start:start + cfg.replan]           # 서버가 고른 실행 구간 20스텝
        oracle = A_raw[t:t + cfg.replan]                  # 오라클이 실제로 할 20스텝
        calls.append((t, served.copy(), oracle.copy()))
        d = np.abs(served - oracle).mean()
        print(f"  t={t:>3}  |서버 argmax - 오라클| 차원당 {d:.5f}", flush=True)

        # 환경은 오라클 액션으로 진행한다 (궤적 위를 유지)
        for k in range(cfg.replan):
            obs, _r, term, _tr, info = env.step(action_25_to_env(A_raw[t + k]))
            if bool(info.get("succeed", False)) or term:
                break
        t += cfg.replan
        if bool(info.get("succeed", False)):
            print(f"  [오라클 재현] t={t} 에서 성공"); break

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.out,
                        frames=np.array([c[0] for c in calls]),
                        served=np.stack([c[1] for c in calls]),
                        oracle=np.stack([c[2] for c in calls]),
                        episode=i, length=T)
    print(f"\n[저장] {cfg.out}  ({len(calls)} 호출)")
    print(f"       서버가 덤프한 후보 32개는 --dump-obs 경로의 npz 에 있다")


if __name__ == "__main__":
    main(tyro.cli(Config))
