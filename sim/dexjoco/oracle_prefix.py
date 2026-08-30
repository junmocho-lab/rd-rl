#!/usr/bin/env python3
"""오라클의 **앞 K 프레임만** 실행하고 나머지는 정책에 넘겨 성공률을 잰다.

1단계(oracle_walk.py) 실측:
    t=  0  |정책 청크 - 오라클| 차원당 0.030   <- 크다 (RTC prefix 없음)
    t>=20  0.0022 ~ 0.0042                     <- 작다 (리플레이 붕괴 임계 0.005 미만)
즉 오라클 궤적 위에 있으면 정책은 오라클과 거의 같은 액션을 낸다. 차이는 첫 청크에 몰려 있다.
이것이 사실이면 **첫 청크만 오라클로 깔아줘도** 나머지는 정책이 알아서 따라가야 한다.

  K=0   : 순정 BC (기준선 77%)
  K=20  : 첫 replan 창만 오라클
  K=40  : 두 창
전부 성공하면 -> 승부는 첫 청크에서 갈린다. test-time 선택은 t=0 에 집중하면 된다.
안 오르면 -> 첫 청크 가설이 틀렸고, 이탈은 그 뒤에 누적된다.
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
    chunk_from_response, obs_to_request,
)


@dataclass
class Config:
    session: Path = Path("/workspace/junmo_cho/dexjoco/rollout/fixed/s0_n100")
    episode: int = -1
    task: str = "hammer_nail"
    host: str = "127.0.0.1"
    port: int = 20461
    episodes: int = 20
    seed: int = 0
    replan: int = 20
    rtc_delay: int = 5
    max_steps: int = 360
    image_size: int = 256
    oracle_steps: str = "0,20,40"
    """오라클 액션을 몇 프레임 깔아줄지 (쉼표 구분). 0 은 순정 BC 기준선."""


def main(cfg: Config) -> None:
    eps = [json.loads(l) for l in (cfg.session / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    pick = (min((e for e in eps if e.get("success")), key=lambda e: e["length"])
            if cfg.episode < 0 else next(e for e in eps if e["episode_index"] == cfg.episode))
    i = pick["episode_index"]
    A_raw = np.vstack([np.asarray(v, np.float32) for v in
                       pd.read_parquet(cfg.session / f"data/chunk-000/episode_{i:06d}.parquet",
                                       columns=["action_raw"])["action_raw"]])
    print(f"[오라클] ep{i}  {pick['length']}프레임  깊이 {pick.get('final_nail_depth',0):.4f}")
    print(f"[기준] 순정 BC(같은 장면) 77/100 = 77.0%\n")

    rc = RolloutConfig(task=cfg.task, image_size=cfg.image_size,
                       replan=cfg.replan, rtc_delay=cfg.rtc_delay)
    env = build_env(rc)
    client = PolicyClient(cfg.host, cfg.port)
    cam_keys = dict(zip(DEFAULT_CAMERA_KEYS, TASK_CAMERAS[cfg.task]))
    prompt = TASK_PROMPTS[cfg.task]

    print(f"{'오라클깔기':>10} {'성공':>10} {'평균깊이':>9} {'평균길이':>9}")
    for K in [int(x) for x in cfg.oracle_steps.split(",")]:
        n_ok, depths, lens = 0, [], []
        for ep in range(cfg.episodes):
            random.seed(cfg.seed); np.random.seed(cfg.seed)   # 장면 고정
            obs, _ = env.reset()
            client.reset()
            raw_hist = []                   # 정책 출력(raw) 이력 — RTC prefix 용
            success, t, depth = False, 0, 0.0
            first_call = True
            chunk, chunk_pos = None, 0
            while t < cfg.max_steps and not success:
                if t < K:
                    a25 = A_raw[t]                        # 오라클 액션을 그대로
                    chunk = None                          # 다음엔 정책에서 새로 받는다
                else:
                    if chunk is None or chunk_pos >= cfg.replan:
                        req, _ = obs_to_request(obs, cam_keys, prompt, cfg.image_size)
                        opts = {}
                        if first_call:
                            opts["reset_memory"] = [True]
                        if cfg.rtc_delay > 0 and len(raw_hist) >= cfg.rtc_delay:
                            opts["action_prefix"] = np.stack(
                                raw_hist[-cfg.rtc_delay:]).astype(np.float32)[None]
                        act_dict, _ = client.get_action(req, opts or None)
                        full = chunk_from_response(act_dict)
                        start = cfg.rtc_delay if (cfg.rtc_delay > 0 and
                                                  len(raw_hist) >= cfg.rtc_delay) else 0
                        chunk, chunk_pos, first_call = full[start:], 0, False
                    a25 = chunk[chunk_pos]
                    chunk_pos += 1
                raw_hist.append(np.asarray(a25, np.float32))
                obs, _r, term, _tr, info = env.step(action_25_to_env(a25))
                depth = float(info.get("nail_depth", depth))
                t += 1
                if bool(info.get("succeed", False)):
                    success = True
                elif term:
                    break
            n_ok += success
            depths.append(depth); lens.append(t)
        print(f"{K:>10} {n_ok:>4}/{cfg.episodes:<5} {np.mean(depths):9.4f} {np.mean(lens):9.0f}",
              flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
