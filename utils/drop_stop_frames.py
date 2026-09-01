#!/usr/bin/env python3
"""HiL 의 `is_stop` 정지 프레임을 데이터셋에서 실제로 제거한 새 데이터셋을 만든다.

왜 (fuji 0831 실측):
  HiL 은 정책이 실패한 지점에서 로봇을 멈추고 사람이 teleop 으로 인계받는데, 그
  **정지 구간 동안 비디오와 parquet 이 계속 기록된다**. 프레임당 |Δq| 중앙값이
  0.000001 로 정책 주행(0.001076)의 1/1000 — 물리적으로 멈춘 프레임이다.
  한 에피소드의 구조:
      -  [   0- 536]  537f  |Δq|=0.001803   정책 주행
      S  [ 537-1480]  944f  |Δq|=0.000025   실패 -> 정지 (사람 개입 대기)   <- 버린다
      I  [1481-1812]  332f  |Δq|=0.001461   teleop 개입                     <- 남긴다
      B  [1813-1849]   37f  |Δq|=0.000034   성공 후 정지                    <- 버린다
  critic 의 목표가 γ^(T-t) 라서 정지 구간에서도 목표만 계속 상승한다 → "안 움직이는
  상태 = 성공" 을 배우고, PA-RL 서빙에서 가만히 있는 후보를 선호하게 된다.
  실제로 critic_grid 에서 Q 가 요동친 ep20/ep34 가 각각 53% / 64% 정지 구간이었다.

`is_intervention` 은 버리지 않는다: |Δq| 중앙값 0.000383 으로 정책 주행과 같은 자릿수다
(실제로 움직이는 teleop 구간이고 HiL 의 핵심 데이터다). 정지이면서 개입인 프레임(B)은
is_stop 이 True 라 함께 빠진다. 즉 판정 기준은 **is_stop 하나**다.

종단 라벨: next.success / next.done 은 **에피소드 마지막 프레임에만** 붙는데 36/39
에피소드가 그 마지막이 정지 구간 안이다. 그냥 버리면 성공 라벨이 사라지므로, 버리기
전에 남는 마지막 프레임으로 옮긴다.

  PY=third_party/RLDX-1/.venv/bin/python
  PYTHONPATH=third_party/RLDX-1:. $PY utils/drop_stop_frames.py \
      --src rl-dataset/fuji/0831_fuji_all \
      --dst rl-dataset/fuji/0831_fuji_all_nostop

is_stop 컬럼이 없는 세션(teleop / inference)은 손대지 않고 그대로 복사한다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TERMINAL = ("next.success", "next.done", "next.truncated")


def has_stop(session: Path) -> bool:
    import pyarrow.parquet as pq
    f = sorted(session.glob("data/*/*.parquet"))
    return bool(f) and "is_stop" in pq.ParquetFile(f[0]).schema.names


def rewrite_episode(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, int]:
    """(새 df, 남긴 원본 frame_index, 옮긴 종단 라벨 수)."""
    keep = ~df["is_stop"].to_numpy(dtype=bool)
    if not keep.any():
        raise ValueError("전 프레임이 is_stop 이다")
    last_kept = int(np.flatnonzero(keep)[-1])
    moved = 0
    for c in TERMINAL:
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=bool)
        if v.any() and not v[last_kept]:          # 버려질 프레임에만 붙어 있었다
            df.iloc[last_kept, df.columns.get_loc(c)] = True
            if c == "next.success":
                moved += 1
    vidx = df["frame_index"].to_numpy(dtype=np.int64)[keep]
    out = df[keep].reset_index(drop=True)
    return out, vidx, moved


def reencode(src: Path, dst: Path, vidx: np.ndarray, fps: int) -> None:
    """src 의 vidx 프레임만 골라 dst 에 h264/yuv420p 로 다시 쓴다."""
    import imageio.v2 as imageio
    from rldx.utils.video_utils import get_frames_by_indices

    fr = get_frames_by_indices(str(src), vidx, video_backend="torchcodec")
    dst.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(dst), fps=fps, codec="libx264", pixelformat="yuv420p",
                           macro_block_size=1, output_params=["-crf", "18"])
    for f in fr:
        w.append_data(np.ascontiguousarray(f[..., :3]).astype(np.uint8))
    w.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--link-untouched", action="store_true",
                   help="is_stop 이 없는 세션을 복사 대신 심링크한다 (find_sessions 는 "
                        "심링크를 따라간다). 디스크를 아끼지만 원본을 지우면 깨진다")
    p.add_argument("--force", action="store_true", help="--dst 가 있으면 지우고 다시 만든다")
    a = p.parse_args()

    src, dst = a.src.resolve(), a.dst.resolve()
    if dst.exists():
        if not a.force:
            return print(f"이미 있다: {dst}  (--force 로 덮어쓸 것)") or 2
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    sessions = sorted(d for d in src.iterdir() if d.is_dir() and (d / "meta").is_dir())
    print(f"[원본] {src}  세션 {len(sessions)}")
    t0 = time.time()
    total_before = total_after = 0
    rewritten = []

    for s in sessions:
        info = json.loads((s / "meta/info.json").read_text())
        before = int(info["total_frames"])
        total_before += before
        if not has_stop(s):
            if a.link_untouched:
                (dst / s.name).symlink_to(s)
                how = "심링크"
            else:
                shutil.copytree(s, dst / s.name, symlinks=True)
                how = "복사"
            total_after += before
            print(f"  {s.name:56s} is_stop 없음 -> {how} ({before} 프레임)")
            continue

        # ── 재작성 ────────────────────────────────────────────────────────
        o = dst / s.name
        (o / "meta").mkdir(parents=True)
        for f in ("modality.json", "tasks.jsonl"):
            if (s / "meta" / f).exists():
                shutil.copy2(s / "meta" / f, o / "meta" / f)
        vkeys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
        fps = int(info["fps"])
        chunks = int(info["chunks_size"])

        eps_meta, moved_tot, kept_tot = [], 0, 0
        old_eps = {json.loads(l)["episode_index"]: json.loads(l)
                   for l in (s / "meta/episodes.jsonl").read_text().splitlines() if l.strip()}
        pq_files = sorted(s.glob("data/*/*.parquet"))
        gidx = 0
        for f in pq_files:
            df = pd.read_parquet(f)
            e = int(df["episode_index"].iloc[0])
            new, vidx, moved = rewrite_episode(df)
            moved_tot += moved
            kept_tot += len(new)
            # 인덱스 재부여: frame_index / timestamp / index 는 연속이어야 한다
            new["frame_index"] = np.arange(len(new), dtype=df["frame_index"].dtype)
            new["timestamp"] = (np.arange(len(new), dtype=np.float32) / fps).astype(
                df["timestamp"].dtype)
            new["index"] = np.arange(gidx, gidx + len(new), dtype=df["index"].dtype)
            gidx += len(new)
            op = o / f"data/chunk-{e // chunks:03d}/episode_{e:06d}.parquet"
            op.parent.mkdir(parents=True, exist_ok=True)
            new.to_parquet(op, index=False)
            for k in vkeys:
                reencode(s / f"videos/chunk-{e // chunks:03d}/{k}/episode_{e:06d}.mp4",
                         o / f"videos/chunk-{e // chunks:03d}/{k}/episode_{e:06d}.mp4",
                         vidx, fps)
            m = dict(old_eps.get(e, {"episode_index": e, "tasks": []}))
            m["length"] = len(new)
            eps_meta.append(m)
            print(f"    ep{e:3d}  {len(df):5d} -> {len(new):5d}", flush=True)

        (o / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in eps_meta))
        info2 = dict(info)
        info2["total_frames"] = kept_tot
        info2["total_videos"] = len(eps_meta) * len(vkeys)
        (o / "meta/info.json").write_text(json.dumps(info2, indent=4, ensure_ascii=False) + "\n")
        # episodes_stats.jsonl 은 이 레포의 어떤 코드도 읽지 않는다 (rl/ utils/ rldx/ 전부
        # grep 확인). 이미지 픽셀 통계까지 들어 있어 재계산이 비싸고, 낡은 것을 남기면
        # 함정이 되므로 아예 만들지 않는다. lerobot 툴링이 필요해지면 그때 생성할 것.
        total_after += kept_tot
        rewritten.append(o)
        print(f"  {s.name:56s} 재작성 {before} -> {kept_tot} 프레임, "
              f"종단 라벨 {moved_tot} 에피소드 이전")

    # ── stats.json 재생성 (lerobot_episode_loader 가 assert 하는 파일) ────────
    from rldx.data.stats import generate_stats
    for o in rewritten:
        print(f"  [stats] {o.name}")
        generate_stats(o)

    print(f"\n[완료] {total_before} -> {total_after} 프레임 "
          f"(제거 {total_before - total_after}, {100*(total_before-total_after)/total_before:.1f}%) "
          f"{time.time()-t0:.0f}s")
    print(f"[출력] {dst}")
    print("\n다음: configs/exp/<exp>.yaml 의 dataset: 을 새 경로로 바꾸고 "
          "cogfeat -> critic 을 다시 돌릴 것 (cogfeat.npy 는 프레임 수가 달라 무효다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
