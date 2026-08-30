"""LeRobot v2.1 데이터셋에서 에피소드 부분집합을 만든다.

action distillation 의 arm 을 만들 때 쓴다:
  arm 1 (filtered BC)   원본 데이터셋 --eps success           → 성공 에피소드, 원본 액션
  arm 2 (distill)       relabel 출력  --eps success           → 성공 에피소드, 개선된 액션

parquet 은 다시 쓰고(에피소드/프레임 번호를 0..N-1 로 다시 매긴다) 비디오는 **심링크**한다.
원본 200 에피소드가 23GB 라 복사하면 낭비다.

  python3 rl/make_subset.py --data <src> --out <dst> --eps success
"""
import argparse, json, shutil
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--eps", default="success", choices=("all", "success", "fail"))
p.add_argument("--episodes", default="", help="쉼표로 나열한 원본 episode_index. --eps 보다 우선")
a = p.parse_args()

info = json.loads((a.data / "meta/info.json").read_text())
meta = [json.loads(l) for l in (a.data / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
CH = info["chunks_size"]

if a.episodes:
    want = [int(x) for x in a.episodes.replace(",", " ").split()]
elif a.eps == "all":
    want = [m["episode_index"] for m in meta]
else:
    want = [m["episode_index"] for m in meta if bool(m.get("success")) == (a.eps == "success")]
by = {m["episode_index"]: m for m in meta}
print(f"[선택] {len(want)}/{len(meta)} 에피소드  (--eps {a.eps})")
if not want:
    raise SystemExit("고른 에피소드가 없다")

a.out.mkdir(parents=True, exist_ok=True)
(a.out / "meta").mkdir(exist_ok=True)
for f in ("modality.json", "tasks.jsonl"):
    shutil.copy2(a.data / "meta" / f, a.out / "meta" / f)

vkeys = list(info.get("features", {}).keys())
vkeys = [k for k in vkeys if k.startswith("observation.images.")] or [
    d.name for d in sorted((a.data / "videos" / "chunk-000").iterdir()) if d.is_dir()]

rows, gidx, nframes = [], 0, 0
for new, old in enumerate(want):
    src = a.data / info["data_path"].format(episode_chunk=old // CH, episode_index=old)
    df = pd.read_parquet(src)
    n = len(df)
    # 번호를 0..N-1 로 다시 매긴다. index 는 데이터셋 전역 프레임 번호다.
    df["episode_index"] = new
    df["index"] = range(gidx, gidx + n)
    dst = a.out / info["data_path"].format(episode_chunk=new // CH, episode_index=new)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)

    for vk in vkeys:                                   # 비디오는 심링크
        vs = a.data / info["video_path"].format(episode_chunk=old // CH, video_key=vk,
                                                episode_index=old)
        vd = a.out / info["video_path"].format(episode_chunk=new // CH, video_key=vk,
                                               episode_index=new)
        vd.parent.mkdir(parents=True, exist_ok=True)
        if not vd.exists():
            vd.symlink_to(vs.resolve())

    m = dict(by[old]); m["episode_index"] = new; m["source_episode_index"] = old
    rows.append(m); gidx += n; nframes += n

(a.out / "meta/episodes.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
info = dict(info)
info["total_episodes"] = len(want)
info["total_frames"] = nframes
info["total_videos"] = len(want) * len(vkeys)
info["total_chunks"] = (len(want) - 1) // CH + 1
if "splits" in info:
    info["splits"] = {"train": f"0:{len(want)}"}
info["subset_of"] = str(a.data)
info["subset_filter"] = a.eps if not a.episodes else "explicit"
(a.out / "meta/info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
print(f"[출력] {a.out}  에피소드 {len(want)}  프레임 {nframes}  비디오 심링크 {len(want)*len(vkeys)}")
