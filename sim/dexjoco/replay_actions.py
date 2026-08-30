#!/usr/bin/env python3
"""로그된 액션 시퀀스를 **그대로 재생**해서 결과를 확인한다 (정책 없음, 오픈루프).

두 가지를 잰다:

  1) 재현성 — 고정 장면(--fixed-scene)에서 같은 액션을 다시 넣으면 같은 결과가 나오는가.
     나오면 "이 장면에서는 액션 시퀀스가 결과를 완전히 결정한다" 가 확인된다.

  2) **섭동 내성** — 그 액션에 노이즈를 섞으면 얼마나 큰 노이즈에서 성공이 깨지는가.
     이것이 환경이 주는 **참값 액션 민감도**다. critic 이 재현해야 할 대상이고,
     probe_actsens 가 critic 에서 재던 것의 정답지에 해당한다.
     정책 후보들의 산포가 차원당 0.018 이므로, 그 근처에서 성공이 깨지면 후보 선택에
     의미가 있고, 0.2 를 줘도 안 깨지면 선택으로 얻을 것이 없다는 뜻이다.

    python sim/dexjoco/replay_actions.py --session <롤아웃> --seed 0 \
        --episode best --noise 0,0.005,0.01,0.02,0.05,0.1 --trials 3
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_raw_to_rldx import TASK_CAMERAS  # noqa: E402
from rollout_dexjoco import Config as RolloutConfig, action_25_to_env, build_env  # noqa: E402


@dataclass
class Config:
    session: Path = Path("/workspace/junmo_cho/dexjoco/rollout/fixed/s0_n100")
    task: str = "hammer_nail"
    seed: int = 0
    """장면 시드. 수집 때 쓴 --fixed-scene 과 같아야 한다."""
    top: int = 10
    """가장 빨리 성공한 에피소드 상위 몇 개를 재생할지."""
    out: Path = Path("/workspace/junmo_cho/dexjoco/replay")
    """비디오 저장 경로. <out>/ep<원본번호>_n<노이즈>_t<시행>.mp4"""
    video: bool = True
    noise: str = "0,0.005,0.01,0.02,0.05,0.1"
    """액션에 더할 가우시안 노이즈 표준편차 목록 (raw 액션 단위)."""
    trials: int = 3
    """노이즈 수준별 반복 횟수 (노이즈 0 은 1회면 충분 — 결정적이면 같은 결과)."""
    max_steps: int = 400


def main(cfg: Config) -> None:
    import pandas as pd

    import imageio.v2 as imageio

    eps = [json.loads(l) for l in (cfg.session / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    ok = sorted([e for e in eps if e.get("success")], key=lambda e: e["length"])[:cfg.top]
    print(f"[선택] 가장 빨리 성공한 {len(ok)}개: "
          + ", ".join(f"ep{e['episode_index']}({e['length']})" for e in ok))

    rc = RolloutConfig(task=cfg.task, image_size=256)
    env = build_env(rc)
    cam0 = TASK_CAMERAS[cfg.task][0]           # front
    cfg.out.mkdir(parents=True, exist_ok=True)

    def run(A: np.ndarray, noise: float, trial: int, tag: str) -> tuple[bool, float, int]:
        # 장면 시드를 매번 같은 값으로 걸어 초기 조건을 고정한다 (env.reset 은 seed= 를 무시하고
        # 전역 RNG 에서 뽑는다 — rollout_dexjoco.py 의 --fixed-scene 과 같은 처리).
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        env.reset()
        rng = np.random.default_rng(10_000 * trial + int(noise * 1e6))
        depth, frames, succ = 0.0, [], False
        t = 0
        for t in range(min(len(A), cfg.max_steps)):
            a = A[t].astype(np.float64)
            if noise > 0:
                a = a + rng.normal(0.0, noise, size=a.shape)
            obs, _r, term, trunc, info = env.step(action_25_to_env(a))
            depth = float(info.get("nail_depth", depth))
            if cfg.video:
                frames.append(np.asarray(obs[cam0]))
            succ = bool(info.get("succeed", False))
            if succ or term:
                break
        if cfg.video and frames:
            dst = cfg.out / f"{tag}.mp4"
            w = imageio.get_writer(str(dst), fps=30, codec="libx264",
                                   pixelformat="yuv420p", macro_block_size=1,
                                   output_params=["-crf", "20"])
            for f in frames:
                w.append_data(f)
            w.close()
        return succ, depth, t + 1

    NS = [float(x) for x in cfg.noise.split(",")]
    tot = {ns: [0, 0] for ns in NS}
    print(f"\n{'원본':>8} {'길이':>5} " + " ".join(f"{ns:>8.3f}" for ns in NS))
    for e in ok:
        i = e["episode_index"]
        A = np.vstack([np.asarray(v, np.float32) for v in
                       pd.read_parquet(cfg.session / f"data/chunk-000/episode_{i:06d}.parquet",
                                       columns=["action"])["action"]])
        cells = []
        for ns in NS:
            n_try = 1 if ns == 0 else cfg.trials
            res = [run(A, ns, k, f"ep{i:03d}_n{ns:g}_t{k}") for k in range(n_try)]
            s = sum(r[0] for r in res)
            tot[ns][0] += s
            tot[ns][1] += n_try
            cells.append(f"{s}/{n_try}")
        print(f"{f'ep{i}':>8} {e['length']:>5} " + " ".join(f"{c:>8}" for c in cells), flush=True)
    print(f"{'합계':>8} {'':>5} " + " ".join(f"{tot[ns][0]}/{tot[ns][1]:<6}" for ns in NS))
    print(f"\n[비디오] {cfg.out}/ep<번호>_n<노이즈>_t<시행>.mp4")

    print("\n해석: 노이즈 0 에서 원본과 같은 결과가 나오면 이 장면은 결정적이다.\n"
          "      정책 후보들의 실제 산포는 차원당 0.018 이다 — 그 근처에서 성공이 깨지기\n"
          "      시작하면 후보 선택/편집에 의미가 있고, 0.1 을 줘도 안 깨지면\n"
          "      액션을 골라봐야 결과가 안 바뀐다는 뜻이다.")


if __name__ == "__main__":
    main(tyro.cli(Config))
