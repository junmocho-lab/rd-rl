#!/usr/bin/env python3
"""Downscale the videos of LeRobot datasets, copying everything else as-is.

inference든 dataset이든 일단 teleop이나 rollout으로 만들면 그날 만든 데이터들을 하나의 디렉토리에 일단 넣고
그 다음 그 디렉토리를 다음 명령어에 넣어주면 됨. 아래처럼 0815_openarm_rh56f1_inference

Usage:
    python convert_data.py ~/Code/rrc-release/data/junmo_cho/0815_openarm_rh56f1_inference/ -o ../dataset

Every child directory of SRC that looks like a LeRobot dataset (has data/, meta/,
videos/ and meta/info.json) is mirrored into OUT (default: <script dir>/dataset)
with its .mp4 files resized to 320x192 (cv2.INTER_LINEAR, re-encoded with libx264).
SRC's own name becomes a group directory under OUT, and each dataset inside it
gets the target size appended:

    OUT/0815_openarm_rh56f1_inference/openarm_rh56f1_inference_eval_..._320x192/

data/ and meta/ are copied byte-for-byte, except that the video resolution written
in meta/info.json is patched to the new size (--keep-info to leave it alone).

--modality PATH copies that file into every converted dataset as meta/modality.json
(RLDX needs it; rrc-release does not write one). It is installed even for datasets
that are already converted, so it can be added in a second pass:

    python convert_data.py <src> -o ./rl-dataset/r0 --modality modality/openarm_lefthand/modality.json

Re-running is incremental: datasets already converted at the same target size are
skipped, and a dataset that gained new episodes only converts the missing videos.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

DEFAULT_OUT = Path(__file__).resolve().parent / "dataset"
MARKER_NAME = ".convert_data.json"
REQUIRED_DIRS = ("data", "meta", "videos")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")


# --------------------------------------------------------------------------- #
# dataset discovery
# --------------------------------------------------------------------------- #
def is_lerobot_dataset(path: Path) -> bool:
    """Light sanity check: data/ meta/ videos/ + meta/info.json all present."""
    if not path.is_dir():
        return False
    if not all((path / d).is_dir() for d in REQUIRED_DIRS):
        return False
    return (path / "meta" / "info.json").is_file()


def find_datasets(src: Path) -> list[Path]:
    if is_lerobot_dataset(src):
        return [src]
    out = []
    for child in sorted(src.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if is_lerobot_dataset(child):
            out.append(child)
        else:
            missing = [d for d in REQUIRED_DIRS if not (child / d).is_dir()]
            reason = f"missing {'/'.join(missing)}" if missing else "no meta/info.json"
            print(f"  [skip] {child.name}: not a LeRobot dataset ({reason})")
    return out


def list_videos(ds: Path) -> list[Path]:
    """Video paths relative to the dataset root, sorted."""
    vids = [
        p.relative_to(ds)
        for p in (ds / "videos").rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    return sorted(vids)


# --------------------------------------------------------------------------- #
# video conversion
# --------------------------------------------------------------------------- #
def convert_video(job: dict) -> dict:
    """Decode with cv2, resize (INTER_LINEAR), re-encode through ffmpeg."""
    src, dst = Path(job["src"]), Path(job["dst"])
    width, height = job["width"], job["height"]
    tmp = dst.with_name(dst.name + ".tmp.mp4")
    res = {"src": str(src), "dst": str(dst), "ok": False, "frames": 0, "error": ""}

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        res["error"] = "cannot open video"
        return res

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # 0 / None / NaN
        fps = job["fallback_fps"]

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        job["ffmpeg"], "-y", "-loglevel", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps}",
        "-i", "pipe:0",
        "-an",
        "-c:v", job["vcodec"], "-pix_fmt", "yuv420p",
        "-crf", str(job["crf"]), "-preset", job["preset"],
        "-g", str(job["gop"]),
        str(tmp),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            proc.stdin.write(small.tobytes())
            res["frames"] += 1
        proc.stdin.close()
        stderr = proc.stderr.read().decode(errors="replace").strip()
        if proc.wait() != 0:
            res["error"] = f"ffmpeg failed: {stderr[:500]}"
        elif res["frames"] == 0:
            res["error"] = "no frames decoded"
        else:
            os.replace(tmp, dst)
            res["ok"] = True
    except BrokenPipeError:
        stderr = proc.stderr.read().decode(errors="replace").strip()
        proc.wait()
        res["error"] = f"ffmpeg died: {stderr[:500]}"
    except Exception as exc:  # noqa: BLE001 - reported back to the parent
        proc.kill()
        proc.wait()
        res["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cap.release()
        for pipe in (proc.stdin, proc.stderr):
            try:
                pipe.close()
            except Exception:
                pass
        if not res["ok"]:
            tmp.unlink(missing_ok=True)
    return res


# --------------------------------------------------------------------------- #
# non-video payload (data/, meta/, loose files)
# --------------------------------------------------------------------------- #
def copy_rest(src_ds: Path, dst_ds: Path) -> int:
    """Copy everything except videos/, skipping files already identical in size+mtime."""
    copied = 0
    for root, dirs, files in os.walk(src_ds):
        rel_root = Path(root).relative_to(src_ds)
        if rel_root == Path("."):
            dirs[:] = [d for d in dirs if d != "videos"]
        out_root = dst_ds / rel_root
        out_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name == MARKER_NAME:
                continue
            s, d = Path(root) / name, out_root / name
            if d.exists():
                ss, ds_ = s.stat(), d.stat()
                if ss.st_size == ds_.st_size and int(ss.st_mtime) == int(ds_.st_mtime):
                    continue
            shutil.copy2(s, d)
            copied += 1
    return copied


def patch_info_json(info_path: Path, width: int, height: int) -> None:
    """Rewrite the video resolution recorded in meta/info.json."""
    with open(info_path) as f:
        info = json.load(f)
    for feat in info.get("features", {}).values():
        if feat.get("dtype") != "video":
            continue
        names = feat.get("names") or []
        shape = feat.get("shape")
        if isinstance(shape, list) and len(shape) == len(names):
            for i, n in enumerate(names):
                if n == "height":
                    shape[i] = height
                elif n == "width":
                    shape[i] = width
        elif isinstance(shape, list) and len(shape) == 3:  # assume (C, H, W)
            shape[1], shape[2] = height, width
        vinfo = feat.get("info")
        if isinstance(vinfo, dict):
            if "video.height" in vinfo:
                vinfo["video.height"] = height
            if "video.width" in vinfo:
                vinfo["video.width"] = width
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def install_modality(dst_ds: Path, modality: Path, dry_run: bool) -> bool:
    """Drop modality.json into the converted dataset's meta/.

    Called for skipped datasets too, so `--modality` can be added on a later run
    without re-encoding anything.
    """
    dst = dst_ds / "meta" / "modality.json"
    if dst.exists() and dst.read_bytes() == modality.read_bytes():
        return False
    if dry_run:
        print(f"         would install meta/modality.json <- {modality}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(modality, dst)
    print(f"         installed meta/modality.json <- {modality}")
    return True


def read_fps(ds: Path, default: float = 30.0) -> float:
    try:
        with open(ds / "meta" / "info.json") as f:
            return float(json.load(f).get("fps", default))
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# per-dataset driver
# --------------------------------------------------------------------------- #
def load_marker(dst_ds: Path) -> dict | None:
    try:
        with open(dst_ds / MARKER_NAME) as f:
            return json.load(f)
    except Exception:
        return None


def convert_dataset(src_ds: Path, dst_ds: Path, args) -> str:
    """Returns one of: 'skipped', 'done', 'failed'."""
    src_videos = list_videos(src_ds)
    marker = load_marker(dst_ds)
    up_to_date = (
        marker is not None
        and not args.force
        and marker.get("width") == args.width
        and marker.get("height") == args.height
        and marker.get("num_videos") == len(src_videos)
    )
    if up_to_date:
        print(f"[skip] {src_ds.name}: already converted ({len(src_videos)} videos)")
        if args.modality:
            install_modality(dst_ds, args.modality, args.dry_run)
        return "skipped"

    todo = []
    for rel in src_videos:
        dst = dst_ds / rel
        if args.force or not dst.exists() or dst.stat().st_size == 0:
            todo.append(rel)

    tag = "convert" if marker is None else "update"
    print(
        f"[{tag}] {src_ds.name}: {len(todo)}/{len(src_videos)} videos "
        f"-> {args.width}x{args.height}"
    )
    if args.dry_run:
        return "done"

    dst_ds.mkdir(parents=True, exist_ok=True)
    n_copied = copy_rest(src_ds, dst_ds)
    if not args.keep_info:
        patch_info_json(dst_ds / "meta" / "info.json", args.width, args.height)
    if args.modality:
        install_modality(dst_ds, args.modality, args.dry_run)
    if n_copied:
        print(f"         copied {n_copied} non-video files")

    fallback_fps = read_fps(src_ds)
    jobs = [
        {
            "src": str(src_ds / rel),
            "dst": str(dst_ds / rel),
            "width": args.width,
            "height": args.height,
            "fallback_fps": fallback_fps,
            "ffmpeg": args.ffmpeg,
            "vcodec": args.vcodec,
            "crf": args.crf,
            "preset": args.preset,
            "gop": args.gop,
        }
        for rel in todo
    ]

    failures = []
    t0 = time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(convert_video, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                if not res["ok"]:
                    failures.append(res)
                    print(f"\n  [fail] {Path(res['src']).name}: {res['error']}")
                elapsed = time.time() - t0
                eta = elapsed / i * (len(jobs) - i)
                print(
                    f"\r         {i}/{len(jobs)} videos  "
                    f"({elapsed:6.1f}s elapsed, ~{eta:6.1f}s left)",
                    end="",
                    flush=True,
                )
        print()

    if failures:
        print(f"  {len(failures)} video(s) failed - marker not written, rerun to retry")
        return "failed"

    with open(dst_ds / MARKER_NAME, "w") as f:
        json.dump(
            {
                "source": str(src_ds),
                "width": args.width,
                "height": args.height,
                "num_videos": len(src_videos),
                "modality": str(args.modality) if args.modality else None,
                "interpolation": "cv2.INTER_LINEAR",
                "vcodec": args.vcodec,
                "crf": args.crf,
                "gop": args.gop,
                "converted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )
    return "done"


# --------------------------------------------------------------------------- #
def parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().replace("*", "x").split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad size {text!r}, expected WxH e.g. 320x192")
    return w, h


def main() -> int:
    p = argparse.ArgumentParser(
        description="Resize LeRobot dataset videos, copying data/ and meta/ unchanged.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("src", type=Path, help="directory holding the LeRobot datasets")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="output root")
    p.add_argument("-s", "--size", type=parse_size, default="320x192", help="target WxH")
    p.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 4),
                   help="parallel video conversions")
    p.add_argument("--force", action="store_true", help="re-convert everything")
    p.add_argument("--dry-run", action="store_true", help="only report what would run")
    p.add_argument("--modality", type=Path, default=None,
                   help="modality.json to install into each dataset's meta/")
    p.add_argument("--keep-info", action="store_true",
                   help="do not patch the resolution in meta/info.json")
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    p.add_argument("--vcodec", default="libx264")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--preset", default="medium")
    p.add_argument("--gop", type=int, default=2,
                   help="keyframe interval (2 matches the source datasets)")
    args = p.parse_args()

    args.width, args.height = (
        args.size if isinstance(args.size, tuple) else parse_size(args.size)
    )

    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not src.is_dir():
        print(f"error: {src} is not a directory", file=sys.stderr)
        return 1
    if args.modality is not None:
        args.modality = args.modality.expanduser().resolve()
        if not args.modality.is_file():
            print(f"error: modality file not found: {args.modality}", file=sys.stderr)
            return 1
        try:
            with open(args.modality) as f:
                json.load(f)
        except json.JSONDecodeError as exc:
            print(f"error: {args.modality} is not valid JSON ({exc})", file=sys.stderr)
            return 1
    if shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).exists():
        print(f"error: ffmpeg not found ({args.ffmpeg}), pass --ffmpeg /path/to/ffmpeg",
              file=sys.stderr)
        return 1

    # A collection directory is reproduced under OUT as a group of the same name;
    # pointing SRC straight at a single dataset writes into OUT itself.
    out_root = out if is_lerobot_dataset(src) else out / src.name
    print(f"src: {src}\nout: {out_root}\n")
    datasets = find_datasets(src)
    if not datasets:
        print("no LeRobot dataset found")
        return 1

    counts = {"done": 0, "skipped": 0, "failed": 0}
    suffix = f"_{args.width}x{args.height}"
    for ds in datasets:
        counts[convert_dataset(ds, out_root / (ds.name + suffix), args)] += 1

    print(
        f"\nconverted {counts['done']}, skipped {counts['skipped']}, "
        f"failed {counts['failed']}  (of {len(datasets)} datasets)"
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
