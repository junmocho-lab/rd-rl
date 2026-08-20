"""actor 쪽 공통 유틸 — send_round / recv_round 가 같은 정의를 쓰게 한다."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from learner.loop import count_files  # noqa: E402,F401  (세는 방식의 단일 소스)


def read_paths(path: Path | None = None) -> dict:
    """configs/paths.sh 의 KEY=value 를 읽는다. paths.sh 가 경로의 단일 소스이므로
    셸에서 source 하지 않고 실행해도 같은 값을 쓰게 한다."""
    path = path or REPO / "configs" / "paths.sh"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.replace("_", "").isalnum():
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def current_run(runs_root: str | None) -> str | None:
    """start_learner.sh 가 적어둔 현재 run id. --exp 를 매번 타이핑하지 않게 한다."""
    if not runs_root:
        return None
    p = Path(runs_root) / "CURRENT"
    return p.read_text().strip() if p.is_file() else None


def run(cmd: list[str], dry: bool = False, check: bool = True) -> str:
    print("$ " + " ".join(cmd))
    if dry:
        return ""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 and check:
        sys.exit(f"실패 (exit {res.returncode}): {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout


def kube(ns: str, *args: str) -> list[str]:
    return ["kubectl", "-n", ns, *args]


def kube_quiet(ns: str, *args: str) -> tuple[int, str]:
    """출력 없이 실행하고 (returncode, stdout). 폴링용."""
    res = subprocess.run(kube(ns, *args), capture_output=True, text=True)
    return res.returncode, res.stdout


def remote_count(ns: str, pod: str, remote_dir: str) -> tuple[int, int]:
    """원격 디렉토리의 (일반 파일 수, 총 바이트). learner/loop.py 의 count_files 와 같은 정의."""
    script = (
        f"find {remote_dir} -type f -printf '%s\\n' 2>/dev/null "
        "| awk '{s+=$1; n+=1} END {printf \"%d %d\\n\", n+0, s+0}'"
    )
    rc, out = kube_quiet(ns, "exec", pod, "--", "bash", "-lc", script)
    if rc != 0 or not out.split():
        sys.exit(f"원격 스캔 실패: {remote_dir}")
    n, b = out.split()
    return int(n), int(b)


def git_sha(repo: Path = REPO) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return ""
