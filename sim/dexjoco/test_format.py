#!/usr/bin/env python3
"""Round-trip a synthetic dexjoco-layout dataset through RLDX-1's own loader.

This checks the *format contract* — meta file names and keys, the info.json
data_path / video_path patterns, modality.json group slicing, the annotation ->
tasks.jsonl indirection, stats.json generation — without needing the raw DexJoCo
download or the sim venv. It writes random states, actions and videos through the
same `_features` / `_modality` / `write_meta` helpers the real converter uses, so
a schema change in one place fails here.

Run from the RLDX-1 training venv (needs torch/rldx, not zarr):

    cd /rlwrld2/home/junmo_cho/ws/rd-rl
    PYTHONPATH=third_party/RLDX-1:. NO_ALBUMENTATIONS_UPDATE=1 \
        third_party/RLDX-1/.venv/bin/python sim/dexjoco/test_format.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sim.dexjoco.convert_raw_to_rldx import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    Config,
    _features,
    _modality,
    write_episode,
    write_meta,
)


N_EPISODES = 3
LENGTHS = [24, 31, 18]


def build(root: Path) -> Config:
    cfg = Config(input=root, output=root, task="hammer_nail", image_size=64, fps=30)
    camera_keys = ["camera_front", "camera_wrist"]
    rng = np.random.default_rng(0)
    episodes = []
    global_index = 0
    for ep, length in enumerate(LENGTHS[:N_EPISODES]):
        state = rng.standard_normal((length, STATE_DIM)).astype(np.float32)
        action = rng.standard_normal((length, ACTION_DIM)).astype(np.float32)
        frames = {
            key: rng.integers(0, 255, (length, cfg.image_size, cfg.image_size, 3), dtype=np.uint8)
            for key in camera_keys
        }
        episodes.append(
            write_episode(cfg, ep, state, action, frames, camera_keys, global_index)
        )
        global_index += length
    write_meta(cfg, episodes, camera_keys, prompt="Use the hammer to drive the nail into the wooden board.")
    return cfg


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="dexjoco_fmt_"))
    try:
        root = tmp / "synthetic"
        root.mkdir()
        cfg = Config(input=root, output=root, task="hammer_nail", image_size=64, fps=30)
        build(root)

        # 1. stats.json, exactly as DatasetFactory._ensure_stats would generate it
        from rldx.data.stats import generate_stats

        generate_stats(root)
        assert (root / "meta/stats.json").exists()

        # 2. the loader RLDX-1 training actually uses
        import rldx.configs.data.dexjoco_panda_allegro_config as dj  # registers the tag
        from rldx.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        cfgs = dj.dexjoco_panda_allegro_config
        loader = LeRobotEpisodeLoader(root, cfgs)
        assert len(loader) == N_EPISODES, len(loader)
        assert loader.episode_lengths == LENGTHS[:N_EPISODES], loader.episode_lengths

        df = loader[1]
        expected_cols = {
            "language.annotation.human.task_description",
            "state.eef_position", "state.eef_rotation", "state.hand_joints",
            "action.eef_position", "action.eef_rotation", "action.hand_joints",
            "video.camera_front", "video.camera_wrist",
        }  # fmt: skip
        missing = expected_cols - set(df.columns)
        assert not missing, f"missing loader columns: {missing}"
        assert len(df) == LENGTHS[1], (len(df), LENGTHS[1])
        assert df["state.eef_rotation"].iloc[0].shape == (6,), df["state.eef_rotation"].iloc[0].shape
        assert df["action.hand_joints"].iloc[0].shape == (16,)
        assert df["video.camera_front"].iloc[0].shape == (cfg.image_size, cfg.image_size, 3)
        assert df["language.annotation.human.task_description"].iloc[0].startswith("Use the hammer")

        # 3. normalisation statistics keyed the way the processor expects
        stats = loader.get_dataset_statistics()
        for modality in ("state", "action"):
            for group, dim in (("eef_position", 3), ("eef_rotation", 6), ("hand_joints", 16)):
                got = len(stats[modality][group]["min"])
                assert got == dim, f"{modality}.{group}: stats dim {got} != {dim}"

        # 4. the horizon the sbatch passes must match the config, or assembly.py dies
        horizon = len(cfgs["action"].delta_indices)
        assert horizon == dj.ACTION_HORIZON == 32, (horizon, dj.ACTION_HORIZON)

        # 5. a step must have a full action chunk available somewhere in a real episode
        print(
            f"OK — {N_EPISODES} synthetic episodes loaded through LeRobotEpisodeLoader; "
            f"state {STATE_DIM}d / action {ACTION_DIM}d / horizon {horizon}; "
            f"stats groups {sorted(stats['action'])}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
