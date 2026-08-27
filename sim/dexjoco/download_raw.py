#!/usr/bin/env python3
"""Download DexJoCo raw (zarr + per-episode mp4) demo sets from the HF hub.

The *raw* repo is used rather than DexJoCo-Datasets-LeRobot because the latter is
LeRobot **v3.0** (one aggregated parquet + concatenated videos), while RLDX-1's
LeRobotEpisodeLoader wants v2.1 (per-episode parquet + per-episode mp4 +
meta/episodes.jsonl). Raw is already one directory per episode, so
convert_raw_to_rldx.py can write v2.1 straight out of it.

    python sim/dexjoco/download_raw.py hammer_nail water_plant
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "DexJoCo/DexJoCo-Datasets-Raw"
OUT = Path("/workspace/junmo_cho/dexjoco/raw")


def main(tasks: list[str], regime: str = "dexjoco_raw_datasets") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        print(f"=== {regime}/{task}", flush=True)
        snapshot_download(
            repo_id=REPO,
            repo_type="dataset",
            local_dir=str(OUT),
            allow_patterns=[f"{regime}/{task}/*"],
            max_workers=16,
        )
        n = len(list((OUT / regime / task).glob("*/replay.zarr")))
        print(f"    {n} episodes under {OUT / regime / task}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or ["hammer_nail"]
    main(args)
