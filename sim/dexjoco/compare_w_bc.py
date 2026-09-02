#!/usr/bin/env python3
"""eval arm 과 BC 를 **같은 장면**끼리 짝지어 나란히 놓은 비디오를 만든다.

씬이 시드로 재현되므로 eval 의 k번째 에피소드와 수집 데이터의 k번째는 같은 초기 배치다.
그래서 "왜 이겼나 / 왜 졌나" 를 같은 장면 위에서 볼 수 있다.

  win_*   arm 성공 & BC 실패   → critic 이 무엇을 고쳤나
  lose_*  arm 실패 & BC 성공   → critic 이 무엇을 망쳤나

  python3 sim/dexjoco/compare_w_bc.py --arm <eval 디렉토리>
"""
import argparse, json, os, shutil, subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--arm", type=Path, required=True, help="eval 디렉토리")
p.add_argument("--bc", type=Path, default=None, help="비교할 수집 데이터셋 (기본: exp yaml 의 dataset)")
p.add_argument("--cam", default="observation.images.camera_front")
p.add_argument("--max", type=int, default=8, help="win/lose 각각 최대 몇 쌍")
p.add_argument("--out", default="compare_w_bc")
a = p.parse_args()


def eps(d: Path):
    return [json.loads(x) for x in open(d / "meta/episodes.jsonl") if x.strip()]


def vid(d: Path, i: int, cam: str) -> Path:
    return d / f"videos/chunk-{i//1000:03d}/{cam}/episode_{i:06d}.mp4"


arm = a.arm
if a.bc is None:
    # eval_<exp>/<arm> 에서 exp 를 꺼내 yaml 의 dataset 을 찾는다
    exp = arm.parent.name.replace("eval_", "")
    ds = [l.split()[1] for l in open(REPO / f"configs/exp/{exp}.yaml") if l.startswith("dataset:")][0]
    a.bc = REPO / ds
A, B = eps(arm), eps(a.bc)
n = len(A)
ao = np.array([bool(e.get("success")) for e in A])
bo = np.array([bool(e.get("success")) for e in B])[:n]
win = np.flatnonzero(ao & ~bo)
lose = np.flatnonzero(~ao & bo)
print(f"[{arm.name}]  n={n}  arm {ao.sum()}/{n} = {100*ao.mean():.1f}%  "
      f"BC {bo.sum()}/{n} = {100*bo.mean():.1f}%")
print(f"  이김 {len(win)}개  짐 {len(lose)}개  (불일치 {len(win)+len(lose)})")

out = arm / a.out
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

rows = []
for kind, idxs in (("win", win), ("lose", lose)):
    for i in idxs[: a.max]:
        va, vb = vid(arm, int(i), a.cam), vid(a.bc, int(i), a.cam)
        if not (va.is_file() and vb.is_file()):
            print(f"  [건너뜀] ep {i}: 비디오 없음")
            continue
        dst = out / f"{kind}_ep{int(i):03d}.mp4"
        # 왼쪽 arm / 오른쪽 BC. drawtext 로 라벨을 박는다 (뭐가 뭔지 헷갈리지 않게).
        vf = ("[0:v]drawtext=text='ARM %s':x=8:y=8:fontsize=18:fontcolor=yellow:"
              "box=1:boxcolor=black@0.5[l];"
              "[1:v]drawtext=text='BC %s':x=8:y=8:fontsize=18:fontcolor=cyan:"
              "box=1:boxcolor=black@0.5[r];[l][r]hstack=inputs=2") % (
            "OK" if ao[i] else "FAIL", "OK" if bo[i] else "FAIL")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(va), "-i", str(vb),
                            "-filter_complex", vf, "-r", "30", str(dst)],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"  [실패] ep {i}: {r.stderr.strip()[:120]}")
            continue
        rows.append((kind, int(i), A[i]["length"], B[i]["length"],
                     A[i].get("final_nail_depth", 0), B[i].get("final_nail_depth", 0)))
        print(f"  {dst.name}  arm {A[i]['length']}f / BC {B[i]['length']}f")

# ── 정량 요약 ──────────────────────────────────────────────────────────────
def stat(sel, key, src):
    v = [ (A if src=='arm' else B)[i].get(key, 0) for i in sel ]
    return (np.mean(v), np.min(v), np.max(v)) if v else (0,0,0)

txt = [f"# {arm.name} vs BC — 같은 장면 비교\n",
       f"n={n},  arm {ao.sum()}/{n} = {100*ao.mean():.1f}%,  BC {bo.sum()}/{n} = {100*bo.mean():.1f}%",
       f"이김(arm 성공/BC 실패) {len(win)},  짐(arm 실패/BC 성공) {len(lose)},  "
       f"둘다성공 {int((ao&bo).sum())},  둘다실패 {int((~ao&~bo).sum())}\n",
       "## 에피소드 길이 / 최종 못 깊이 (성공 임계 0.04)\n",
       "| 구분 | ep | arm 길이 | BC 길이 | arm 깊이 | BC 깊이 |",
       "|---|---|---|---|---|---|"]
for kind, i, la, lb, da, db in rows:
    txt.append(f"| {kind} | {i} | {la} | {lb} | {da:.4f} | {db:.4f} |")
txt.append("")
for kind, sel in (("이김", win), ("짐", lose), ("둘다성공", np.flatnonzero(ao & bo)),
                  ("둘다실패", np.flatnonzero(~ao & ~bo))):
    if not len(sel):
        continue
    la = np.mean([A[i]["length"] for i in sel]); lb = np.mean([B[i]["length"] for i in sel])
    da = np.mean([A[i].get("final_nail_depth", 0) for i in sel])
    db = np.mean([B[i].get("final_nail_depth", 0) for i in sel])
    txt.append(f"- **{kind}** ({len(sel)}개): 길이 arm {la:.0f} / BC {lb:.0f}, "
               f"깊이 arm {da:.4f} / BC {db:.4f}")
(out / "README.md").write_text("\n".join(txt) + "\n")
print(f"\n[출력] {out}  (비디오 {len(rows)}개 + README.md)")
