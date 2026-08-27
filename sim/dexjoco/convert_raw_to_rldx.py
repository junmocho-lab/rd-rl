#!/usr/bin/env python3
"""DexJoCo raw demos (zarr + per-episode mp4) -> LeRobot v2.1 dataset for RLDX-1.

Why a new converter instead of `dexjoco-dc-single-lerobot`:

  * dexjoco's own converter targets `lerobot.datasets` (LeRobot **v3.0**): one
    aggregated `data/chunk-000/file-000.parquet`, videos concatenated across
    episodes, `meta/episodes/*.parquet`, `meta/tasks.parquet`. RLDX-1's
    `LeRobotEpisodeLoader` wants **v2.1**: per-episode parquet under
    `data/chunk-{c:03d}/episode_{i:06d}.parquet`, per-episode mp4 under
    `videos/chunk-{c:03d}/{video_key}/episode_{i:06d}.mp4`, plus
    `meta/{info.json,episodes.jsonl,tasks.jsonl,modality.json,stats.json}`.
  * dexjoco's converter writes `action_rotvec` as the action target. That is
    **unusable** for chunked absolute-action regression here: the DexJoCo
    end-effector sits near a 180-degree rotation (state quat w ~ 0), so the
    rotation-vector representation sits on the +-pi antipode and flips sign.
    Measured on the published datasets: 216 / 21571 consecutive frames of
    hammer_nail (1.01%) and 212 / 27745 of water_plant (0.77%) jump by more
    than 3 rad, up to a full 2*pi flip, in 66% of episodes. The underlying
    quaternion is smooth (max step 0.023), so the discontinuity is purely an
    artefact of the representation.

    This converter therefore stores the orientation as **rot6d** (the first two
    rows of the rotation matrix flattened, matching
    `EndEffectorPose._matrix_to_rot6d` in RLDX-1). rot6d is continuous *and*
    free of the quaternion double cover (q vs -q), so no per-episode sign
    canonicalisation is needed - which matters because such canonicalisation is
    only consistent *within* an episode and would make identical visual states
    carry opposite BC targets across episodes.

Layout written (single-arm: 3 + 6 + 16 = 25 dims for both state and action):

    observation.state = [eef_xyz(3), eef_rot6d(6), allegro_qpos(16)]
    action            = [eef_xyz(3), eef_rot6d(6), allegro_target(16)]

    from raw state  [tcp_pos(3), tcp_quat_wxyz(4), allegro_qpos(16), <privileged>]
    from raw action [tgt_pos(3), tgt_quat_wxyz(4), allegro_target(16)]

`next.success` / `next.done` are True on the final frame of every episode (all
raw DexJoCo demos are successful teleop takes), `next.truncated` is always
False. Those three columns are what `rl/data.py` reads to build RL transitions,
so the demo set doubles as expert data for the offline critic.

Usage (needs the sim venv, which has zarr<3 / mujoco / imageio):

    /workspace/junmo_cho/dexjoco/venv/bin/python sim/dexjoco/convert_raw_to_rldx.py \
        --input  /workspace/junmo_cho/dexjoco/raw/dexjoco_raw_datasets/hammer_nail \
        --output /workspace/junmo_cho/dexjoco/lerobot/hammer_nail_rand_obj \
        --task hammer_nail

Then generate meta/stats.json from the RLDX-1 venv (training would do it too,
but doing it here surfaces layout errors before a GPU job starts):

    PYTHONPATH=third_party/RLDX-1 third_party/RLDX-1/.venv/bin/python -c \
      "from rldx.data.stats import generate_stats; generate_stats('<output>')"
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import tyro
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# NOTE: `zarr` is imported inside _load_episode, not here. Only the raw reader
# needs it, and keeping it lazy lets the meta-writing helpers (_features /
# _modality / write_meta) be imported from the RLDX-1 training venv, which has no
# zarr — see sim/dexjoco/test_format.py, which round-trips a synthetic dataset
# through LeRobotEpisodeLoader using exactly these functions.


ALLEGRO_JOINT_NAMES = [
    "ffj0", "ffj1", "ffj2", "ffj3",
    "mfj0", "mfj1", "mfj2", "mfj3",
    "rfj0", "rfj1", "rfj2", "rfj3",
    "thj0", "thj1", "thj2", "thj3",
]  # fmt: skip

# rot6d = first two ROWS of the rotation matrix, flattened - the convention of
# EndEffectorPose._matrix_to_rot6d / _rot6d_to_matrix in RLDX-1.
DIM_NAMES = (
    ["eef_x", "eef_y", "eef_z"]
    + ["eef_r11", "eef_r12", "eef_r13", "eef_r21", "eef_r22", "eef_r23"]
    + ALLEGRO_JOINT_NAMES
)
STATE_DIM = ACTION_DIM = len(DIM_NAMES)  # 25

# Prompts are copied verbatim from dexjoco's own configs so that a checkpoint
# trained here answers to the same instruction the benchmark's eval yaml sends
# (configs/rand_obj/*.yaml and dexjoco-data-converter/configs/*/language_instructions.yaml).
TASK_PROMPTS = {
    "hammer_nail": "Use the hammer to drive the nail into the wooden board.",
    "water_plant": "Grasp the watering can and apply water to the plant.",
    "click_mouse": "Move the mouse to the purple mouse pad and click the left mouse button.",
    "pick_bucket": "Place the boxed food into the bucket and then lift the bucket.",
    "pinch_tongs": "Grasp the tongs and perform three consecutive open-close motions.",
    "fold_glasses": "Fold the glasses and place them into the case.",
}

# RLDX-1 modality keys, assigned positionally to --cameras. Keeping them fixed
# across tasks means one modality config (dexjoco_panda_allegro_config.py) covers
# every single-arm task even where the third-person source camera differs.
# camera_front = the third-person view, camera_wrist = the wrist view.
DEFAULT_CAMERA_KEYS = ["camera_front", "camera_wrist"]

# Raw camera names (videos/<name>.mp4) per task, copied from dexjoco's own
# dexjoco-data-converter/configs/rand_obj/selected_data.yaml so a checkpoint
# trained here sees the same views the benchmark's eval yaml serves.
# click_mouse is the odd one out: the monitor occludes `front`, so upstream uses
# the ego_right view as the base camera. Only cameras the *policy-mode* env
# actually renders are usable — the raw demo dirs carry ego_left/ego_right for
# every task, but e.g. panda_water_plant_env only produces wrist + front at
# rollout time, so training water_plant on ego views would be unservable.
TASK_CAMERAS = {
    "hammer_nail": ["front", "wrist"],
    "water_plant": ["front", "wrist"],
    "pick_bucket": ["front", "wrist"],
    "pinch_tongs": ["front", "wrist"],
    "fold_glasses": ["front", "wrist"],
    "click_mouse": ["ego_right", "wrist"],
}

CHUNK_SIZE = 1000


@dataclass
class Config:
    input: Path
    """Raw dataset directory containing one subdirectory per demo (each with replay.zarr/ and videos/)."""
    output: Path
    """Destination LeRobot v2.1 dataset root. Must be absent or empty unless --overwrite."""
    task: str = "hammer_nail"
    """Task name; selects the language instruction."""
    prompt: str | None = None
    """Override the language instruction (default: TASK_PROMPTS[task])."""
    cameras: list[str] | None = None
    """Raw camera names to convert, in order (default: TASK_CAMERAS[task]).
    They map positionally onto DEFAULT_CAMERA_KEYS, so the first entry becomes
    camera_front regardless of which raw view it comes from."""
    image_size: int = 256
    """Output video resolution (square). 256 -> 256*256 = the processor's default image_max_area,
    and a multiple of image_resize_m=32, so RLDX-1's AspectAreaResizeAndCrop is a no-op."""
    fps: int = 30
    """Declared dataset fps. DexJoCo records videos and timestamps at 30."""
    drop_leading: int = 1
    """Frames dropped after the leading-static trim. The first recorded action of an episode is
    the teleop's initial jump away from the home pose (up to 1.1 m in one 1/30 s step) and, in 8
    of the 100 hammer_nail demos, is exactly zero (the env reads an all-zero pose as 'hold')."""
    max_episodes: int = -1
    """Convert at most this many demos (-1 = all). For smoke tests."""
    crf: int = 20
    """libx264 quality for the re-encoded videos."""
    overwrite: bool = False
    """Delete the output directory first if it exists."""


def _normalize_shape(a: np.ndarray) -> np.ndarray:
    """(T, 1, D) -> (T, D); dexjoco's recorder stores state with a singleton axis."""
    if a.ndim == 3 and a.shape[1] == 1:
        return a[:, 0]
    return a


def _first_non_static_frame(action: np.ndarray) -> int:
    """Index of the first action that differs from its successor.

    Same rule as dexjoco's own converter (episode_common.find_first_non_static_frame),
    kept identical so episode boundaries match the published LeRobot datasets.
    """
    for i in range(len(action) - 1):
        if not np.array_equal(action[i], action[i + 1]):
            return i
    raise ValueError("episode is entirely static")


def _pose_quat_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """(T, 4) wxyz -> (T, 6) rot6d (first two rows of R, flattened)."""
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    norms = np.linalg.norm(quat_xyzw, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError(f"{int((norms < 1e-8).sum())} zero-norm quaternions remain after trimming")
    mats = R.from_quat(quat_xyzw / norms[:, None]).as_matrix()  # (T, 3, 3)
    return mats[:, :2, :].reshape(len(mats), 6)


def _to_25(pose_and_hand: np.ndarray) -> np.ndarray:
    """[xyz(3), quat_wxyz(4), hand(16)] -> [xyz(3), rot6d(6), hand(16)]."""
    assert pose_and_hand.shape[1] >= 23, pose_and_hand.shape
    xyz = pose_and_hand[:, 0:3]
    rot6d = _pose_quat_to_rot6d(pose_and_hand[:, 3:7])
    hand = pose_and_hand[:, 7:23]
    return np.concatenate([xyz, rot6d, hand], axis=1).astype(np.float32)


def _load_episode(demo_dir: Path, drop_leading: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (state_25, action_25, start_index_into_the_raw_video)."""
    import zarr  # lazy: only the raw reader needs it — see the note at the imports

    root = zarr.open(str(demo_dir / "replay.zarr"), mode="r")
    data = root["data"]
    raw_action = _normalize_shape(np.asarray(data["action"][:], dtype=np.float64))
    raw_state = _normalize_shape(np.asarray(data["state"][:], dtype=np.float64))
    if len(raw_action) != len(raw_state):
        raise ValueError(f"{demo_dir.name}: action {len(raw_action)} != state {len(raw_state)}")

    start = _first_non_static_frame(raw_action)
    # Skip any all-zero pose commands, which the env interprets as "hold" rather
    # than as a target and which have no valid quaternion.
    while start < len(raw_action) and not np.any(raw_action[start, :7]):
        start += 1
    start += drop_leading
    if len(raw_action) - start < 2:
        raise ValueError(f"{demo_dir.name}: nothing left after trimming {start} frames")

    return _to_25(raw_state[start:]), _to_25(raw_action[start:]), start


def _open_writer(dst: Path, fps: int, crf: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(dst),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-crf", str(crf)],
    )


def _convert_video(src: Path, dst: Path, start: int, length: int, size: int, fps: int, crf: int) -> None:
    """Re-encode frames [start, start+length) of `src` to `size`x`size` at `dst`."""
    reader = imageio.get_reader(str(src))
    writer = _open_writer(dst, fps, crf)
    written = 0
    try:
        for i, frame in enumerate(reader):
            if i < start:
                continue
            if written >= length:
                break
            if frame.shape[0] != size or frame.shape[1] != size:
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            writer.append_data(frame)
            written += 1
    finally:
        writer.close()
        reader.close()
    if written != length:
        raise ValueError(f"{src}: wrote {written} frames, expected {length}")


def _write_video(frames: np.ndarray, dst: Path, fps: int, crf: int) -> None:
    """Encode an in-memory (T, H, W, 3) uint8 stack. Used by rollouts and the format test."""
    writer = _open_writer(dst, fps, crf)
    try:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
    finally:
        writer.close()


def video_path(cfg: Config, ep_idx: int, camera_key: str) -> Path:
    chunk = ep_idx // CHUNK_SIZE
    return (
        cfg.output
        / f"videos/chunk-{chunk:03d}/observation.images.{camera_key}/episode_{ep_idx:06d}.mp4"
    )


def _features(cfg: Config, camera_keys: list[str]) -> dict[str, Any]:
    feats: dict[str, Any] = {}
    for key in camera_keys:
        feats[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": [3, cfg.image_size, cfg.image_size],
            "names": ["channel", "height", "width"],
            "info": {
                "video.height": cfg.image_size,
                "video.width": cfg.image_size,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": cfg.fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    feats["observation.state"] = {"dtype": "float32", "shape": [STATE_DIM], "names": DIM_NAMES}
    feats["action"] = {"dtype": "float32", "shape": [ACTION_DIM], "names": DIM_NAMES}
    for key in ("next.success", "next.truncated", "next.done"):
        feats[key] = {"dtype": "bool", "shape": [1], "names": None}
    feats["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for key in ("frame_index", "episode_index", "index", "task_index"):
        feats[key] = {"dtype": "int64", "shape": [1], "names": None}
    return feats


def _modality(camera_keys: list[str]) -> dict[str, Any]:
    groups = {
        "eef_position": (0, 3),
        "eef_rotation": (3, 9),
        "hand_joints": (9, 25),
    }
    return {
        "state": {
            name: {"start": s, "end": e, "original_key": "observation.state"}
            for name, (s, e) in groups.items()
        },
        "action": {
            name: {"start": s, "end": e, "original_key": "action"}
            for name, (s, e) in groups.items()
        },
        "video": {key: {"original_key": f"observation.images.{key}"} for key in camera_keys},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def write_episode(
    cfg: Config,
    ep_idx: int,
    state: np.ndarray,
    action: np.ndarray,
    frames: dict[str, np.ndarray] | None,
    camera_keys: list[str],
    global_index: int,
    success: np.ndarray | None = None,
    done: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one episode's parquet (and videos, if `frames` is given); return its meta row.

    `success` / `done` / `truncated` default to "successful demo": done and success
    True on the final frame only, never truncated. A rollout writer passes the real
    per-frame flags instead — `rl/data.py` derives reward = next.success and
    mask = 1 - next.done from exactly these columns.
    """
    length = len(action)
    assert len(state) == length, (len(state), length)
    if frames is not None:
        for key in camera_keys:
            _write_video(frames[key], video_path(cfg, ep_idx, key), cfg.fps, cfg.crf)

    terminal = np.zeros(length, dtype=bool)
    terminal[-1] = True
    frame = pd.DataFrame(
        {
            "observation.state": list(np.asarray(state, dtype=np.float32)),
            "action": list(np.asarray(action, dtype=np.float32)),
            "next.success": terminal if success is None else np.asarray(success, dtype=bool),
            "next.truncated": (
                np.zeros(length, dtype=bool) if truncated is None
                else np.asarray(truncated, dtype=bool)
            ),
            "next.done": terminal if done is None else np.asarray(done, dtype=bool),
            "timestamp": np.arange(length, dtype=np.float32) / cfg.fps,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, ep_idx, dtype=np.int64),
            "index": np.arange(global_index, global_index + length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
        }
    )
    chunk = ep_idx // CHUNK_SIZE
    parquet_path = cfg.output / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)

    meta = {"episode_index": ep_idx, "tasks": [cfg.prompt or TASK_PROMPTS[cfg.task]], "length": length}
    meta.update(extra_meta or {})
    return meta


def write_meta(
    cfg: Config,
    episodes_meta: list[dict[str, Any]],
    camera_keys: list[str],
    prompt: str | None = None,
) -> None:
    prompt = prompt or cfg.prompt or TASK_PROMPTS[cfg.task]
    total_frames = sum(m["length"] for m in episodes_meta)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda_allegro",
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes_meta) * len(camera_keys),
        "total_chunks": (len(episodes_meta) - 1) // CHUNK_SIZE + 1,
        "chunks_size": CHUNK_SIZE,
        "fps": cfg.fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _features(cfg, camera_keys),
        # Provenance, not read by RLDX-1: which raw dexjoco camera fed each key.
        # Matters because click_mouse's camera_front is actually the ego_right view,
        # so a rollout client must map the env's cameras the same way.
        "dexjoco_task": cfg.task,
        "dexjoco_cameras": dict(zip(camera_keys, cfg.cameras or [])),
    }
    (cfg.output / "meta").mkdir(parents=True, exist_ok=True)
    (cfg.output / "meta/info.json").write_text(json.dumps(info, indent=4))
    with open(cfg.output / "meta/episodes.jsonl", "w") as f:
        for meta in episodes_meta:
            f.write(json.dumps(meta) + "\n")
    with open(cfg.output / "meta/tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": prompt}) + "\n")
    (cfg.output / "meta/modality.json").write_text(json.dumps(_modality(camera_keys), indent=4))


def main(cfg: Config) -> None:
    prompt = cfg.prompt or TASK_PROMPTS[cfg.task]
    demo_dirs = sorted(p.parent for p in cfg.input.glob("*/replay.zarr"))
    if cfg.max_episodes > 0:
        demo_dirs = demo_dirs[: cfg.max_episodes]
    if not demo_dirs:
        raise SystemExit(f"no demos with replay.zarr under {cfg.input}")

    if cfg.output.exists() and any(cfg.output.iterdir()):
        if not cfg.overwrite:
            raise SystemExit(f"{cfg.output} is not empty (pass --overwrite to replace)")
        shutil.rmtree(cfg.output)
    if cfg.cameras is None:
        cfg.cameras = list(TASK_CAMERAS[cfg.task])
    if len(cfg.cameras) != len(DEFAULT_CAMERA_KEYS):
        raise SystemExit(
            f"--cameras must have {len(DEFAULT_CAMERA_KEYS)} entries "
            f"(mapped positionally onto {DEFAULT_CAMERA_KEYS}), got {cfg.cameras}"
        )
    camera_keys = list(DEFAULT_CAMERA_KEYS)
    (cfg.output / "meta").mkdir(parents=True, exist_ok=True)

    episodes_meta: list[dict[str, Any]] = []
    global_index = 0
    for ep_idx, demo_dir in enumerate(tqdm(demo_dirs, desc=cfg.task)):
        state, action, start = _load_episode(demo_dir, cfg.drop_leading)
        length = len(action)

        # Videos are streamed straight from the source mp4 rather than decoded into
        # memory first: 100 episodes x 2 cameras x 640x640 would be ~50 GB of RAM.
        for raw_cam, key in zip(cfg.cameras, camera_keys):
            _convert_video(
                demo_dir / "videos" / f"{raw_cam}.mp4",
                video_path(cfg, ep_idx, key),
                start=start,
                length=length,
                size=cfg.image_size,
                fps=cfg.fps,
                crf=cfg.crf,
            )

        episodes_meta.append(
            write_episode(
                cfg, ep_idx, state, action, None, camera_keys, global_index,
                extra_meta={"raw_demo": demo_dir.name, "raw_start_frame": start},
            )
        )
        global_index += length

    write_meta(cfg, episodes_meta, camera_keys, prompt=prompt)

    lengths = np.array([m["length"] for m in episodes_meta])
    print(
        f"{cfg.output}: {len(episodes_meta)} episodes / {global_index} frames "
        f"(len min {lengths.min()} / mean {lengths.mean():.1f} / max {lengths.max()}), "
        f"cameras {camera_keys}"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
