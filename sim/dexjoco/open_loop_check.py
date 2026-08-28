#!/usr/bin/env python3
"""Open-loop check: does the served policy reproduce its own training actions?

Separates two failure modes that a low rollout success rate cannot distinguish:

  * client wiring wrong (image/state encoding, group order, dtype, scaling)
      -> predictions are far off even on frames the model was trained on
  * policy genuinely weak / overfit / distribution-shifted
      -> predictions match on training frames but rollouts still fail

Feeds frames and states straight out of the BC LeRobot dataset (decoded from the
same mp4s training read) through the same request builder rollout_dexjoco.py uses,
and compares the returned chunk against the dataset's recorded action chunk.

    /workspace/junmo_cho/dexjoco/venv/bin/python sim/dexjoco/open_loop_check.py \
        --dataset ~/ws/dexjoco_dataset/hammer_nail_rand_obj --port 20200
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_raw_to_rldx import DEFAULT_CAMERA_KEYS, TASK_PROMPTS  # noqa: E402
from rollout_dexjoco import GROUPS, LANGUAGE_KEY, PolicyClient  # noqa: E402


@dataclass
class Config:
    dataset: Path = Path("/rlwrld2/home/junmo_cho/ws/dexjoco_dataset/hammer_nail_rand_obj")
    host: str = "127.0.0.1"
    port: int = 20200
    episodes: int = 3
    """How many dataset episodes to sample frames from."""
    frames_per_episode: int = 8
    """Query points spread evenly through each episode."""
    horizon: int = 32


def read_video(path: Path, indices: np.ndarray) -> np.ndarray:
    import imageio.v2 as imageio

    want = set(int(i) for i in indices)
    out: dict[int, np.ndarray] = {}
    reader = imageio.get_reader(str(path))
    try:
        for i, frame in enumerate(reader):
            if i in want:
                out[i] = np.asarray(frame, dtype=np.uint8)
            if len(out) == len(want):
                break
    finally:
        reader.close()
    return np.stack([out[int(i)] for i in indices])


def main(cfg: Config) -> None:
    info = json.loads((cfg.dataset / "meta/info.json").read_text())
    task = info.get("dexjoco_task") or cfg.dataset.name.replace("_rand_obj", "")
    prompt = json.loads((cfg.dataset / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    print(f"[i] {cfg.dataset.name}: {info['total_episodes']} ep, task {task}")
    print(f"[i] prompt: {prompt!r}")

    client = PolicyClient(cfg.host, cfg.port)
    client.ping()

    all_err, all_scale = [], []
    for ep in range(cfg.episodes):
        pq = cfg.dataset / f"data/chunk-000/episode_{ep:06d}.parquet"
        df = pd.read_parquet(pq)
        actions = np.stack(df["action"].values).astype(np.float32)
        states = np.stack(df["observation.state"].values).astype(np.float32)
        n = len(df)
        idx = np.linspace(0, max(n - cfg.horizon - 1, 0), cfg.frames_per_episode).astype(int)

        videos = {
            key: read_video(
                cfg.dataset / f"videos/chunk-000/observation.images.{key}/episode_{ep:06d}.mp4",
                idx,
            )
            for key in DEFAULT_CAMERA_KEYS
        }

        for j, t in enumerate(idx):
            request = {LANGUAGE_KEY: [prompt]}
            for key in DEFAULT_CAMERA_KEYS:
                request[f"video.{key}"] = videos[key][j][None, None]
            for name, lo, hi in GROUPS:
                request[f"state.{name}"] = states[t, lo:hi][None, None].astype(np.float32)

            action_dict, _ = client.get_action(request)
            pred = np.concatenate(
                [np.asarray(action_dict[f"action.{g[0]}"], np.float32)[0] for g in GROUPS], -1
            )
            gt = actions[t : t + cfg.horizon]
            k = min(len(pred), len(gt))
            err = np.abs(pred[:k] - gt[:k])
            all_err.append(err)
            all_scale.append(np.abs(gt[:k]))

    err = np.concatenate(all_err, 0)      # (N, 25)
    scale = np.concatenate(all_scale, 0)
    names = [("eef_position", 0, 3), ("eef_rotation", 3, 9), ("hand_joints", 9, 25)]
    print(f"\n{len(err)} predicted timesteps")
    print(f"{'group':14s} {'MAE':>9s} {'p95':>9s} {'max':>9s}   {'|gt| mean':>9s}  MAE/|gt|")
    for name, lo, hi in names:
        e, s = err[:, lo:hi], scale[:, lo:hi]
        print(
            f"{name:14s} {e.mean():9.4f} {np.percentile(e, 95):9.4f} {e.max():9.4f}   "
            f"{s.mean():9.4f}  {e.mean() / max(s.mean(), 1e-9):7.3f}"
        )
    print(
        "\n해석: 모델이 이 데이터를 100 epoch 넘게 봤으므로 배선이 맞다면 MAE/|gt| 가"
        " 작아야 한다 (수 % 수준). 0.5 를 넘으면 관측 인코딩이 어긋난 것이다."
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
