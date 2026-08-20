#!/usr/bin/env python3
"""learner → actor 라운드 산출물 회수.

    <ckpt>/expo/<exp>/rNNN/DONE 이 나타날 때까지 기다린 뒤 그 디렉토리를 로컬로 내려받는다.

DONE 은 learner 가 **맨 마지막에** 쓰므로, 그게 보이면 산출물이 완성된 것이다.
FAILED 가 보이면 이유를 출력하고 종료한다 (무한 대기 방지).

로컬 목적지는 learner 와 **같은 상대경로**를 쓴다:
    learner  $L_CKPT/expo/<exp>/rNNN/
    actor    $A_CKPT/expo/<exp>/rNNN/
그래서 나중에 정책 서버가 "어느 라운드의 산출물"만 알면 경로가 유도된다.

usage:
    python3 actor/recv_round.py --round 1 [--exp <run id>] [--timeout 3600] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    count_files, current_run, kube, kube_quiet, read_paths, remote_count, run,
)


def remote_status(ns: str, pod: str, remote_round: str) -> str | None:
    """DONE / FAILED 중 무엇이 있는지. 폴링이라 조용히 실행한다."""
    # `; exit 0` 이 필수다. 마지막 [ -f ] 가 실패하면 bash 전체가 exit 1 로 끝나서
    # rc!=0 → None → "아직 안 됨" 으로 오해하고 DONE 이 있어도 영원히 기다린다.
    rc, out = kube_quiet(
        ns, "exec", pod, "--", "bash", "-lc",
        f"for f in DONE FAILED; do [ -f {remote_round}/$f ] && echo $f; done; exit 0",
    )
    if rc != 0:
        return None
    names = out.split()
    return names[0] if names else None


def remote_cat(ns: str, pod: str, path: str) -> dict | None:
    rc, out = kube_quiet(ns, "exec", pod, "--", "cat", path)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def main() -> int:
    paths = read_paths()
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--exp", default=current_run(paths.get("A_RUNS")),
                   help="run id. 생략하면 runs/CURRENT")
    p.add_argument("--local-ckpt", default=paths.get("A_CKPT"))
    p.add_argument("--remote-ckpt", default=paths.get("L_CKPT"))
    p.add_argument("--namespace", default=paths.get("L_NS"))
    p.add_argument("--pod", default=paths.get("L_POD"))
    p.add_argument("--timeout", type=float, default=0,
                   help="DONE 을 기다릴 최대 초. 0 = 무한")
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if not a.exp:
        sys.exit("--exp 를 알 수 없다. ./actor/start_learner.sh 로 learner 를 띄우거나 --exp 를 줄 것")
    for name in ("local_ckpt", "remote_ckpt", "namespace", "pod"):
        if not getattr(a, name):
            sys.exit(f"--{name.replace('_','-')} 를 알 수 없다 (configs/paths.sh 확인)")

    rnd = f"r{a.round:03d}"
    rel = f"expo/{a.exp}/{rnd}"
    remote_round = f"{a.remote_ckpt}/{rel}"
    local_round = Path(a.local_ckpt) / rel

    print(f"라운드 {rnd}  exp={a.exp}")
    print(f"  원격  {remote_round}")
    print(f"  로컬  {local_round}")
    print()

    # 1) DONE / FAILED 를 기다린다
    t0 = time.time()
    last_note = 0.0
    while True:
        st = remote_status(a.namespace, a.pod, remote_round)
        if st == "DONE":
            print(f"DONE 감지 ({time.time()-t0:.0f}초 대기)")
            break
        if st == "FAILED":
            rec = remote_cat(a.namespace, a.pod, f"{remote_round}/FAILED") or {}
            print("learner 가 이 라운드를 FAILED 로 표시했다:")
            print(f"  이유: {rec.get('reason')}")
            for x in rec.get("problems", []):
                print(f"    ✗ {x}")
            return 1
        if a.timeout and (time.time() - t0) > a.timeout:
            return print(f"타임아웃 ({a.timeout:.0f}초). learner 로그를 확인할 것") or 1
        now = time.time()
        if now - last_note >= 30:
            print(f"  대기 중... ({now-t0:.0f}초)")
            last_note = now
        time.sleep(a.poll_seconds)

    # 2) 무엇이 왔는지 먼저 본다
    meta = remote_cat(a.namespace, a.pod, f"{remote_round}/meta.json") or {}
    if meta.get("stub"):
        print("  [주의] stub 산출물이다 (실제 학습 아님)")
    if meta:
        ds = meta.get("dataset", {})
        print(f"  학습 데이터: 에피소드 {ds.get('episodes')} / 프레임 {ds.get('frames')}")
        print(f"  계획된 update(): {meta.get('planned_updates')}")
        print(f"  코드: {meta.get('code')}")
    n_remote, b_remote = remote_count(a.namespace, a.pod, remote_round)
    print(f"  원격 파일 {n_remote}개 / {b_remote/1e6:.1f} MB")
    print()

    if a.dry_run:
        print("[dry-run] 여기서 멈춘다")
        return 0

    # 3) 로컬을 비우고 내려받는다 — kubectl cp 는 대상이 있으면 그 안에 넣어 중첩된다
    if not str(local_round).endswith(f"/{a.exp}/{rnd}") or len(a.exp) < 3:
        sys.exit(f"안전장치: 지우려는 경로가 예상과 다르다: {local_round}")
    if local_round.exists():
        print(f"기존 로컬 디렉토리 제거: {local_round}")
        shutil.rmtree(local_round)
    local_round.parent.mkdir(parents=True, exist_ok=True)
    run(kube(a.namespace, "cp", f"{a.pod}:{remote_round}", str(local_round)))

    # 4) 대조
    n_local, b_local = count_files(local_round)
    if (n_local, b_local) != (n_remote, b_remote):
        sys.exit(f"불일치 — 다시 실행하면 새로 내려받는다.\n"
                 f"  원격 files={n_remote} bytes={b_remote}\n"
                 f"  로컬 files={n_local} bytes={b_local}")
    print(f"검증 OK: files={n_local} bytes={b_local}")
    print()
    print(f"산출물: {local_round}")
    for f in sorted(local_round.rglob("*")):
        if f.is_file():
            print(f"  {f.stat().st_size:>12,}  {f.relative_to(local_round)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
