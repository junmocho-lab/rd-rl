#!/usr/bin/env python3
"""actor → learner 라운드 전송.

    dataset/ 를 세션별로 올린 뒤 **맨 마지막에** READY 를 올린다.

READY 를 마지막에 따로 올리는 이유: kubectl cp 는 디렉토리를 원자적으로 옮기지 않는다.
learner 는 READY 안의 숫자를 실제 디스크와 대조해서 절반만 도착한 라운드를 걸러낸다
(learner/loop.py 의 validate). 그래서 여기서 세는 방식이 learner 와 **정확히 같아야** 한다
— 그래서 로직을 복제하지 않고 learner.loop.scan_dataset 을 그대로 import 한다.

usage:
    python3 actor/send_round.py --exp openarm_red_block --round 0 \\
        --dataset rl-dataset/r0/0815_openarm_rh56f1_inference/<session>_320x192 \\
        --collected-by base [--success 2] [--dry-run]

경로 기본값은 configs/paths.sh 에서 읽는다 (source 없이도 동작한다).

같은 라운드를 다시 보내면 원격 dataset/ 을 비우고 새로 올린다 (멱등).
단 learner 가 그 라운드를 **처리 중**일 때 다시 보내지 말 것 — 읽는 중인 데이터가 사라진다.
"""

from __future__ import annotations

import argparse
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    REPO, current_run, git_sha, kube, read_paths, remote_count, run,
)
from learner.loop import scan_sessions  # noqa: E402  (세는 방식의 단일 소스)


def main() -> int:
    paths = read_paths()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=current_run(paths.get("A_RUNS")),
                   help="run id. 생략하면 runs/CURRENT (start_learner.sh 가 씀)")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--dataset", required=True, nargs="+", type=Path,
                   help="올릴 LeRobot 세션 디렉토리들 (또는 그것들을 담은 부모 디렉토리)")
    p.add_argument("--collected-by", default="base",
                   help="이 데이터를 모은 정책 (base | rNNN). 라운드별 성공률 해석의 근거")
    p.add_argument("--success", type=int, default=None,
                   help="성공 에피소드 수 (선택). learner 는 검증하지 않고 기록만 한다")
    p.add_argument("--runs-root", default=paths.get("A_RUNS"), help="로컬 라운드 기록 위치")
    p.add_argument("--remote-runs-root", default=paths.get("L_RUNS"))
    p.add_argument("--namespace", default=paths.get("L_NS"))
    p.add_argument("--pod", default=paths.get("L_POD"))
    p.add_argument("--dry-run", action="store_true", help="스캔만 하고 아무것도 올리지 않는다")
    a = p.parse_args()

    if not a.exp:
        sys.exit("--exp 를 알 수 없다. ./actor/start_learner.sh 로 learner 를 띄우거나 --exp 를 직접 줄 것")
    for name in ("runs_root", "remote_runs_root", "namespace", "pod"):
        if not getattr(a, name):
            sys.exit(f"--{name.replace('_','-')} 를 알 수 없다 (configs/paths.sh 확인)")

    # 1) 올릴 세션 결정 — 부모 디렉토리를 주면 그 안의 세션들로 펼친다
    sessions: list[Path] = []
    for d in a.dataset:
        d = d.expanduser().resolve()
        if not d.is_dir():
            sys.exit(f"디렉토리 없음: {d}")
        if (d / "meta" / "info.json").is_file():
            sessions.append(d)
        else:
            children = [c for c in sorted(d.iterdir())
                        if c.is_dir() and (c / "meta" / "info.json").is_file()]
            if not children:
                sys.exit(f"LeRobot 세션을 찾을 수 없다: {d}")
            sessions.extend(children)

    # 2) learner 와 같은 함수로 센다 (learner/loop.py 의 scan_sessions)
    stats, problems = scan_sessions(sessions)
    if problems:
        for x in problems:
            print(f"  \u2717 {x}")
        sys.exit("로컬 데이터셋에 문제가 있다. 올리지 않는다.")

    rnd = f"r{a.round:03d}"
    remote_round = f"{a.remote_runs_root}/{a.exp}/{rnd}"
    remote_ds = f"{remote_round}/dataset"

    print(f"라운드 {rnd}  exp={a.exp}")
    print(f"  세션      {len(stats['sessions'])}개")
    for s in stats["sessions"]:
        print(f"              {s}")
    print(f"  에피소드  {stats['episodes']}")
    print(f"  프레임    {stats['frames']}")
    print(f"  파일      {stats['files']}")
    print(f"  크기      {stats['bytes']/1e6:.1f} MB")
    print(f"  원격      {remote_ds}")
    print()

    # 3) 원격을 먼저 비운다 — 같은 라운드를 다시 보내도 멱등하게.
    #
    # kubectl cp 는 대상 디렉토리가 이미 있으면 그 **안에** 복사한다 (tar 의미).
    # 그래서 같은 세션을 두 번 보내면 dataset/<세션>/<세션>/ 으로 중첩되고 파일 수가
    # 늘어 검증에서 걸린다. 지우고 새로 올려야 재실행이 깨끗하다.
    #
    # 안전장치: 지울 경로가 정말 이 라운드의 dataset 인지 확인한다.
    if not remote_ds.endswith(f"/{a.exp}/{rnd}/dataset") or len(a.exp) < 3:
        sys.exit(f"안전장치: 지우려는 경로가 예상과 다르다: {remote_ds}")
    # READY 를 먼저 없앤다 — 데이터가 없어진 라운드를 learner 가 집어가지 않도록.
    run(kube(a.namespace, "exec", a.pod, "--", "rm", "-f", f"{remote_round}/READY"), a.dry_run)
    run(kube(a.namespace, "exec", a.pod, "--", "rm", "-rf", remote_ds), a.dry_run)
    run(kube(a.namespace, "exec", a.pod, "--", "mkdir", "-p", remote_ds), a.dry_run)

    # 4) 세션별 업로드
    for s in sessions:
        run(kube(a.namespace, "cp", str(s), f"{a.pod}:{remote_ds}/{s.name}"), a.dry_run)

    # 5) READY 를 쓰기 전에 원격이 온전한지 확인 — 여기서 어긋나면 재시도하면 되고,
    #    FAILED 라운드를 만들지 않는다
    ready = {
        "round": a.round,
        "episodes": stats["episodes"],
        "success": a.success,
        "frames": stats["frames"],
        "files": stats["files"],
        "bytes": stats["bytes"],
        "sessions": sorted(stats["sessions"]),
        "collected_by": a.collected_by,
        "code": {"rd-rl": git_sha(REPO)},
    }
    if not a.dry_run:
        n, b = remote_count(a.namespace, a.pod, remote_ds)
        if (n, b) != (stats["files"], stats["bytes"]):
            sys.exit(f"원격 불일치 — READY 를 올리지 않는다.\n"
                     f"  로컬 files={stats['files']} bytes={stats['bytes']}\n"
                     f"  원격 files={n} bytes={b}\n"
                     f"  → 같은 명령을 다시 실행하면 kubectl cp 가 덮어쓴다")
        print(f"원격 확인 OK: files={n} bytes={b}")

    # 6) 로컬 기록 + READY 를 마지막에 올린다
    local_round = Path(a.runs_root) / a.exp / rnd
    local_round.mkdir(parents=True, exist_ok=True)
    ready_path = local_round / "READY"
    if a.dry_run:
        print(f"[dry-run] READY 내용:\n{json.dumps(ready, indent=2, ensure_ascii=False)}")
        return 0
    ready_path.write_text(json.dumps(ready, indent=2, ensure_ascii=False) + "\n")
    run(kube(a.namespace, "cp", str(ready_path), f"{a.pod}:{remote_round}/READY"), a.dry_run)

if __name__ == "__main__":
    sys.exit(main())
