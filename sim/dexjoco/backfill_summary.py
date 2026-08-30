"""옛 rollout_summary.json 에 길이 통계를 채워 넣는다.

success_mean_length 등은 나중에 추가된 필드라 그 전에 만든 롤아웃에는 없다.
meta/episodes.jsonl 이 length/success 를 그대로 들고 있으므로 전부 정확히 복원된다
(추정이 아니다). rtc_delay/fixed_scene 은 유도할 근거가 있을 때만 채우고,
채운 것은 _backfilled 에 출처를 남긴다.

  python3 sim/dexjoco/backfill_summary.py            # 확인만
  python3 sim/dexjoco/backfill_summary.py --write    # 실제로 쓴다
"""
import argparse, glob, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--root", default="rl-dataset/dexjoco")
p.add_argument("--write", action="store_true")
a = p.parse_args()

for f in sorted((REPO / a.root).glob("**/rollout_summary.json")):
    d = f.parent
    eps = d / "meta" / "episodes.jsonl"
    if not eps.is_file():
        print(f"[건너뜀] episodes.jsonl 없음: {d}")
        continue
    s = json.loads(f.read_text())
    m = [json.loads(l) for l in eps.read_text().splitlines() if l.strip()]
    L = np.array([x["length"] for x in m])
    ok = np.array([bool(x.get("success")) for x in m])
    sl = L[ok] if ok.any() else np.array([0])

    new, src = {}, []
    for k, v in (("mean_length", float(L.mean())), ("min_length", int(L.min())),
                 ("max_length", int(L.max())), ("success_mean_length", float(sl.mean())),
                 ("success_median_length", float(np.median(sl))),
                 ("success_min_length", int(sl.min())), ("success_max_length", int(sl.max()))):
        if k not in s:
            new[k] = v
    if new:
        src.append("길이통계 <- meta/episodes.jsonl")

    # rtc_delay / fixed_scene: 근거가 있을 때만
    cfg = d / "eval_config.json"
    if cfg.is_file():
        c = json.loads(cfg.read_text())
        if "rtc_delay" not in s and "rtc_delay" in c:
            new["rtc_delay"] = c["rtc_delay"]; src.append("rtc_delay <- eval_config.json")
        if "fixed_scene" not in s and "scene" in c:
            sc = c["scene"]
            if isinstance(sc, str) and sc.startswith("fixed seed"):
                new["fixed_scene"] = int(sc.split()[-1]); src.append("fixed_scene <- eval_config.json")

    else:
        # 수집 데이터셋 디렉토리. exp yaml 과 replan 이 일치하는 것만 근거로 삼는다
        # (summary 의 replan 은 원래부터 기록돼 있어 대조가 가능하다).
        for y in sorted((REPO / "configs/exp").glob("dexjoco_*.yaml")):
            t = y.read_text()
            def fld(k):
                for ln in t.splitlines():
                    if ln.startswith(k + ":"):
                        return ln.split(":", 1)[1].strip()
            if fld("dataset") != f"{a.root}/{d.name}":
                continue
            if s.get("replan") != int(fld("replan_steps")):
                print(f"          [경고] {y.name} 의 replan 이 summary 와 다르다 — 건너뜀")
                break
            if "rtc_delay" not in s:
                new["rtc_delay"] = int(fld("inference_latency"))
                src.append(f"rtc_delay <- {y.name} (replan 일치 확인)")
            if "fixed_scene" not in s:
                # _s0 접미사가 곧 --fixed-scene 0 으로 수집했다는 표시다
                new["fixed_scene"] = 0 if d.name.endswith("_s0") else -1
                src.append("fixed_scene <- 데이터셋 이름 규칙 (_s0 = 고정)")
            break

    if not new:
        print(f"[이미 완비] {d.relative_to(REPO)}")
        continue
    if src:
        new["_backfilled"] = "; ".join(src)
    s.update(new)
    print(f"[{'기록' if a.write else '예정'}] {d.relative_to(REPO)}")
    print(f"          성공 {int(ok.sum())}/{len(m)} = {100*ok.mean():.1f}%  "
          f"전체평균 {L.mean():.1f}  성공평균 {sl.mean():.1f}  성공최소 {int(sl.min())}  "
          f"채운 키 {len(new)-('_backfilled' in new)}개")
    if a.write:
        f.write_text(json.dumps(s, indent=2, ensure_ascii=False))
