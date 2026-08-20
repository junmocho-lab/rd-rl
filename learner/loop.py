#!/usr/bin/env python3
"""rd-rl learner loop — 라운드 메일박스를 감시하고 라운드를 하나씩 처리한다.

메일박스 규약:

    <runs>/<exp>/rNNN/dataset/<session>/{data,meta,videos}   actor 가 올린 LeRobot 데이터셋
    <runs>/<exp>/rNNN/READY                                  actor 가 **맨 마지막에** 올리는 신호(JSON)
    <ckpt>/expo/<exp>/rNNN/DONE                              learner 가 **맨 마지막에** 쓰는 신호
    <ckpt>/expo/<exp>/rNNN/FAILED                            검증 실패 (이유 포함)

READY 를 마지막에 따로 올리는 이유: kubectl cp 는 디렉토리를 원자적으로 옮기지 않아서,
절반만 복사된 데이터셋을 학습이 읽으면 조용히 망가진 걸로 학습한다. READY 안의 숫자를
실제 디스크와 대조하는 것이 그걸 막는 유일한 방법이다.

지금은 학습 자리가 stub 이다 (--stub-seconds 만큼 자고 더미 산출물을 쓴다).
프레임워크 왕복(데이터 감지 → 검증 → 산출물 → DONE)만 검증하는 단계.

usage:
    python3 learner/loop.py --exp <exp> --runs-root $L_RUNS --ckpt-root $L_CKPT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROUND_RE = re.compile(r"^r(\d{3,})$")
REQUIRED_DIRS = ("data", "meta", "videos")

_stop = False


def _on_term(signum, _frame):
    """SIGTERM(파드 축출) 과 크래시를 로그에서 구분할 수 있게 한다."""
    global _stop
    _stop = True
    print(f"[signal] {signal.Signals(signum).name} 수신 — 현재 라운드 후 종료", flush=True)


# --------------------------------------------------------------------------- #
# 로깅 — Job 로그는 잡을 지우면 사라지므로 DDN 파일에도 남긴다
# --------------------------------------------------------------------------- #
class Log:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}"
        print(line, flush=True)
        if self.path:
            with self.path.open("a") as f:
                f.write(line + "\n")


# --------------------------------------------------------------------------- #
# 라운드 탐색
# --------------------------------------------------------------------------- #
def round_num(path: Path) -> int | None:
    m = ROUND_RE.match(path.name)
    return int(m.group(1)) if m else None


def round_status(ckpt_round: Path) -> str | None:
    """이미 처리된 라운드인지. DONE/FAILED 둘 다 '처리됨' 으로 본다."""
    if (ckpt_round / "DONE").exists():
        return "done"
    if (ckpt_round / "FAILED").exists():
        return "failed"
    return None


def find_next_round(runs_exp: Path, ckpt_exp: Path) -> tuple[int, Path] | None:
    """READY 가 있고 DONE/FAILED 가 없는 가장 작은 번호. 재시작 시 이어받기가 이 규칙."""
    candidates = []
    if not runs_exp.is_dir():
        return None
    for d in runs_exp.iterdir():
        n = round_num(d) if d.is_dir() else None
        if n is None or not (d / "READY").is_file():
            continue
        if round_status(ckpt_exp / d.name) is None:
            candidates.append((n, d))
    return min(candidates) if candidates else None


# --------------------------------------------------------------------------- #
# 검증 — READY 의 숫자 vs 실제 디스크
# --------------------------------------------------------------------------- #
def scan_dataset(ds_dir: Path) -> tuple[dict, list[str]]:
    """dataset/ 아래를 훑어 (통계, 문제목록) 반환.

    files/bytes 는 dataset/ 아래 **모든 파일**이 기준이다 (actor 도 같은 정의로 세야 한다).
    """
    problems: list[str] = []
    stats = {"sessions": [], "episodes": 0, "frames": 0, "files": 0, "bytes": 0}

    if not ds_dir.is_dir():
        return stats, [f"dataset/ 없음: {ds_dir}"]

    for p in sorted(ds_dir.iterdir()):
        if not p.is_dir():
            continue
        missing = [d for d in REQUIRED_DIRS if not (p / d).is_dir()]
        info_path = p / "meta" / "info.json"
        if missing:
            problems.append(f"{p.name}: LeRobot 데이터셋 아님 (없는 디렉토리: {'/'.join(missing)})")
            continue
        if not info_path.is_file():
            problems.append(f"{p.name}: meta/info.json 없음")
            continue
        try:
            info = json.loads(info_path.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{p.name}: meta/info.json 파싱 실패 ({exc})")
            continue
        eps = int(info.get("total_episodes", 0))
        if eps == 0:
            problems.append(f"{p.name}: total_episodes=0 (빈 데이터셋은 로더가 죽는다)")
            continue
        stats["sessions"].append(p.name)
        stats["episodes"] += eps
        stats["frames"] += int(info.get("total_frames", 0))

    for f in ds_dir.rglob("*"):
        if f.is_file():
            stats["files"] += 1
            stats["bytes"] += f.stat().st_size

    return stats, problems


def validate(round_dir: Path, ready: dict) -> tuple[dict, list[str]]:
    stats, problems = scan_dataset(round_dir / "dataset")
    for key in ("episodes", "frames", "files", "bytes"):
        if key not in ready:
            problems.append(f"READY 에 '{key}' 없음")
            continue
        if int(ready[key]) != stats[key]:
            problems.append(
                f"{key} 불일치: READY={ready[key]} vs 실제={stats[key]} "
                "(전송이 덜 끝났거나 중간에 끊긴 것)"
            )
    if "sessions" in ready and sorted(ready["sessions"]) != sorted(stats["sessions"]):
        problems.append(
            f"sessions 불일치: READY={sorted(ready['sessions'])} vs 실제={sorted(stats['sessions'])}"
        )
    return stats, problems


# --------------------------------------------------------------------------- #
# 산출물
# --------------------------------------------------------------------------- #
def write_atomic(path: Path, obj: dict) -> None:
    """센티넬은 원자적으로 쓴다 — 반쯤 쓰인 DONE 을 actor 가 읽으면 안 된다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return ""


def code_provenance(repo: Path) -> dict:
    prov = {"rd-rl": git_sha(repo)}
    for name in ("RLDX-1", "expo-ft"):
        sha = git_sha(repo / "third_party" / name)
        if sha:
            prov[name] = sha
    return prov


def export_stub(ckpt_round: Path, log: Log, args, stats: dict, ready: dict) -> None:
    """실제 학습이 붙기 전의 자리. actor recv 를 시험할 수 있게 더미 페이로드를 쓴다."""
    payload_dir = ckpt_round / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    blob = payload_dir / "stub_payload.bin"
    n = int(args.stub_payload_mb * 1024 * 1024)
    with blob.open("wb") as f:
        f.write(os.urandom(n))
    log(f"  stub 페이로드 {blob.name} {n/1e6:.1f} MB")

    write_atomic(ckpt_round / "meta.json", {
        "stub": True,
        "round": ready.get("round"),
        "collected_by": ready.get("collected_by"),
        "dataset": {k: stats[k] for k in ("sessions", "episodes", "frames", "files", "bytes")},
        "planned_updates": args.updates_per_episode * stats["episodes"],
        "artifacts": ["payload/stub_payload.bin"],
        "code": code_provenance(Path(args.repo)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


# --------------------------------------------------------------------------- #
def process_round(n: int, round_dir: Path, ckpt_round: Path, log: Log, args) -> str:
    log(f"[r{n:03d}] 감지: {round_dir}")
    try:
        ready = json.loads((round_dir / "READY").read_text())
    except json.JSONDecodeError as exc:
        write_atomic(ckpt_round / "FAILED", {"round": n, "reason": f"READY JSON 파싱 실패: {exc}"})
        log(f"[r{n:03d}] FAILED — READY JSON 파싱 실패")
        return "failed"

    stats, problems = validate(round_dir, ready)
    if problems:
        write_atomic(ckpt_round / "FAILED", {"round": n, "reason": "검증 실패", "problems": problems})
        for p in problems:
            log(f"[r{n:03d}]   ✗ {p}")
        log(f"[r{n:03d}] FAILED — 학습하지 않음")
        return "failed"

    log(f"[r{n:03d}] 검증 통과: 세션 {len(stats['sessions'])}개 / 에피소드 {stats['episodes']} / "
        f"프레임 {stats['frames']} / {stats['bytes']/1e6:.1f} MB")

    # ── 실제 update() 가 들어갈 자리 ─────────────────────────────────────
    planned = args.updates_per_episode * stats["episodes"]
    log(f"[r{n:03d}] [STUB] 학습 생략. 실제로는 update() {planned}회 "
        f"(에피소드당 {args.updates_per_episode})")
    time.sleep(args.stub_seconds)
    # ────────────────────────────────────────────────────────────────────

    export_stub(ckpt_round, log, args, stats, ready)
    write_atomic(ckpt_round / "DONE", {"round": n, "stub": True,
                                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    log(f"[r{n:03d}] DONE → {ckpt_round}")
    return "done"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True, help="실험 이름 (메일박스 경로에 들어간다)")
    p.add_argument("--runs-root", required=True, type=Path, help="$L_RUNS")
    p.add_argument("--ckpt-root", required=True, type=Path, help="$L_CKPT")
    p.add_argument("--repo", default=".", help="rd-rl 체크아웃 (코드 SHA 기록용)")
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument("--heartbeat-seconds", type=float, default=300.0,
                   help="유휴 상태에서 살아있음을 알리는 주기 (상시 Job 생존 확인용)")
    p.add_argument("--once", action="store_true", help="라운드 하나 처리하고 종료")
    p.add_argument("--updates-per-episode", type=int, default=11,
                   help="에피소드당 update() 횟수. EXPO-FT parity 환산값 (지금은 로그만)")
    p.add_argument("--stub-seconds", type=float, default=5.0, help="stub 학습 소요 시간")
    p.add_argument("--stub-payload-mb", type=float, default=1.0)
    args = p.parse_args()

    runs_exp = args.runs_root / args.exp
    ckpt_exp = args.ckpt_root / "expo" / args.exp
    log = Log(runs_exp / "learner.log")

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    log("=" * 70)
    log(f"learner 시작  exp={args.exp}")
    log(f"  python   {sys.version.split()[0]}  pid={os.getpid()}")
    log(f"  runs     {runs_exp}")
    log(f"  ckpt     {ckpt_exp}")
    log(f"  code     {code_provenance(Path(args.repo))}")
    runs_exp.mkdir(parents=True, exist_ok=True)
    ckpt_exp.mkdir(parents=True, exist_ok=True)

    last_beat = 0.0
    processed = 0
    while not _stop:
        found = find_next_round(runs_exp, ckpt_exp)
        if found is None:
            now = time.time()
            if now - last_beat >= args.heartbeat_seconds:
                log(f"[idle] 대기 중 (처리한 라운드 {processed}개)")
                last_beat = now
            if args.once and processed:
                break
            time.sleep(args.poll_seconds)
            continue

        n, round_dir = found
        process_round(n, round_dir, ckpt_exp / round_dir.name, log, args)
        processed += 1
        last_beat = 0.0
        if args.once:
            break

    log(f"learner 종료 (처리한 라운드 {processed}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
