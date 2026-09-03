#!/usr/bin/env python3
"""HiL 에피소드에서 **개입 시작 전** 구간을 잘라낸다.

HiL 롤아웃의 시간 구조 (2026-09-02 실측, |Δq| 는 프레임당 관절 변화 중앙):
    -  [   0- 510]  511f  |Δq|=0.001708   정책 주행          <- 잘라낸다
    S  [ 511- 763]  253f  |Δq|=0.000055   실패 -> 정지 대기   <- 잘라낸다
    I  [ 764-1048]  285f  |Δq|=0.001636   개입(teleop)       <- 여기부터 남긴다
    B  [1049-1230]  182f  |Δq|=0.000051   성공 후 정지        <- 그대로 둔다 (정상)

`is_intervention` 이 개입 구간이다 — 그 구간의 |Δq| 가 정책 주행과 같은 자릿수이고
앞의 `is_stop` 은 70배 작다 (물리적으로 멈춰 있다).

규칙:
  · 개입이 **있는** 에피소드 → 첫 is_intervention 프레임부터 끝까지 남긴다
  · 개입이 **없는** 에피소드 → 정책이 혼자 성공한 것이므로 **통째로 그대로** 둔다
  · 꼬리의 is_stop(성공 직후 정지)은 원래 그런 것이므로 건드리지 않는다

  PY=third_party/RLDX-1/.venv/bin/python
  PYTHONPATH=third_party/RLDX-1:. $PY utils/cut_before_hil.py \
      --src ~/ws/rby1m_rh56f1_hil_anchor_20260902_134259 \
      --dst rl-dataset/fuji/0902_fuji_rc_success/rby1m_rh56f1_hil_anchor_20260902_134259

frame_index / timestamp / index 는 0..N-1 로 다시 매기고 비디오도 같은 프레임만
재인코딩한다 — 그래야 build_images 의 arange(L) 경로가 그대로 맞는다.
next.success 등 종단 라벨은 마지막 프레임에 그대로 남으므로 옮길 필요가 없다
(자르는 것이 **앞부분**이기 때문이다).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


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
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    src = a.src.expanduser().resolve()
    dst = a.dst.expanduser().resolve()
    if dst.exists():
        if not a.force:
            return print(f"이미 있다: {dst}  (--force 로 덮어쓸 것)") or 2
        shutil.rmtree(dst)

    info = json.loads((src / "meta/info.json").read_text())
    fps = int(info["fps"])
    chunks = int(info["chunks_size"])
    vkeys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    (dst / "meta").mkdir(parents=True)
    for f in ("modality.json", "tasks.jsonl"):
        if (src / "meta" / f).exists():
            shutil.copy2(src / "meta" / f, dst / "meta" / f)
    old_eps = {json.loads(l)["episode_index"]: json.loads(l)
               for l in (src / "meta/episodes.jsonl").read_text().splitlines() if l.strip()}

    t0 = time.time()
    eps_meta, gidx = [], 0
    n_cut = n_whole = 0
    before = after = 0
    for f in sorted(src.glob("data/*/*.parquet")):
        df = pd.read_parquet(f)
        e = int(df["episode_index"].iloc[0])
        L = len(df); before += L
        iv = df["is_intervention"].to_numpy(dtype=bool)
        if iv.any():
            i0 = int(np.flatnonzero(iv)[0])       # 개입 시작
            new = df.iloc[i0:].reset_index(drop=True)
            n_cut += 1
            tag = f"자름 {L} -> {len(new)}  (앞 {i0} 프레임 제거)"
        else:
            i0 = 0
            new = df.reset_index(drop=True)
            n_whole += 1
            tag = f"유지 {L}  (개입 없음 — 정책이 혼자 성공)"
        after += len(new)
        vidx = df["frame_index"].to_numpy(dtype=np.int64)[i0:]

        # 인덱스 재부여 — 비디오도 같은 프레임만 남기므로 0..N-1 로 연속이 된다.
        new["frame_index"] = np.arange(len(new), dtype=df["frame_index"].dtype)
        new["timestamp"] = (np.arange(len(new), dtype=np.float32) / fps).astype(
            df["timestamp"].dtype)
        new["index"] = np.arange(gidx, gidx + len(new), dtype=df["index"].dtype)
        gidx += len(new)

        op = dst / f"data/chunk-{e // chunks:03d}/episode_{e:06d}.parquet"
        op.parent.mkdir(parents=True, exist_ok=True)
        new.to_parquet(op, index=False)

        for k in vkeys:
            sv = src / f"videos/chunk-{e // chunks:03d}/{k}/episode_{e:06d}.mp4"
            dv = dst / f"videos/chunk-{e // chunks:03d}/{k}/episode_{e:06d}.mp4"
            if i0 == 0:
                dv.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sv, dv)              # 안 자른 에피소드는 재인코딩 불필요
            else:
                reencode(sv, dv, vidx, fps)

        m = dict(old_eps.get(e, {"episode_index": e, "tasks": []}))
        m["length"] = len(new)
        eps_meta.append(m)
        print(f"  ep{e:3d}  {tag}", flush=True)

    (dst / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in eps_meta))
    info2 = dict(info)
    info2["total_frames"] = after
    info2["total_videos"] = len(eps_meta) * len(vkeys)
    (dst / "meta/info.json").write_text(json.dumps(info2, indent=4, ensure_ascii=False) + "\n")

    # stats.json 은 lerobot_episode_loader.py:187 이 assert 하는 파일이고, 프레임 분포가
    # 바뀌었으므로 반드시 다시 만든다. episodes_stats.jsonl 은 이 레포의 어떤 코드도
    # 읽지 않아 (rl/ utils/ rldx/ grep 확인) 만들지 않는다 — 낡은 것을 남기면 함정이 된다.
    from rldx.data.stats import generate_stats
    generate_stats(dst)

    print(f"\n[완료] {before} -> {after} 프레임 (제거 {before - after}, "
          f"{100*(before-after)/before:.1f}%)  {time.time()-t0:.0f}s")
    print(f"  에피소드 {len(eps_meta)}개 = 자름 {n_cut} + 유지 {n_whole}")
    print(f"[출력] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
