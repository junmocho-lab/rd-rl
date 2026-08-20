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
import hashlib
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


def ready_fingerprint(round_dir: Path) -> str:
    """READY 파일 내용의 지문. '이 라운드' 가 아니라 '이 READY' 를 처리했는지로 판단한다."""
    p = round_dir / "READY"
    if not p.is_file():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def round_status(ckpt_round: Path, fingerprint: str) -> str | None:
    """이미 처리된 라운드인지.

    라운드 번호가 아니라 **처리한 READY 의 지문**으로 판단한다:
      - 같은 데이터가 다시 오면 건너뛴다 (Job 재시작 시 이어받기)
      - 같은 번호로 새 데이터가 오면 다시 처리한다 (재테스트가 수동 정리 없이 재현된다)
    지문이 없는 옛 DONE/FAILED 는 다시 처리한다.
    """
    for name in ("DONE", "FAILED"):
        p = ckpt_round / name
        if not p.exists():
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if rec.get("ready_sha") and rec["ready_sha"] == fingerprint:
            return name.lower()
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
        if round_status(ckpt_exp / d.name, ready_fingerprint(d)) is None:
            candidates.append((n, d))
    return min(candidates) if candidates else None


# --------------------------------------------------------------------------- #
# 검증 — READY 의 숫자 vs 실제 디스크
# --------------------------------------------------------------------------- #
def count_files(root: Path) -> tuple[int, int]:
    """(일반 파일 수, 총 바이트). actor 와 learner 가 같은 정의를 쓰게 하는 단일 소스."""
    files = nbytes = 0
    for f in root.rglob("*"):
        if f.is_file():
            files += 1
            nbytes += f.stat().st_size
    return files, nbytes


def scan_sessions(sessions: list[Path]) -> tuple[dict, list[str]]:
    """세션 디렉토리 목록을 검사하고 합계를 낸다.

    actor(보낼 세션들) 와 learner(도착한 dataset/ 아래 전부) 가 같은 함수를 쓴다.
    """
    problems: list[str] = []
    stats = {"sessions": [], "episodes": 0, "frames": 0, "files": 0, "bytes": 0}

    for p in sessions:
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
        f, b = count_files(p)
        stats["files"] += f
        stats["bytes"] += b

    return stats, problems


def scan_dataset(ds_dir: Path) -> tuple[dict, list[str]]:
    """도착한 dataset/ 를 훑는다 (learner 쪽).

    files/bytes 는 세션 디렉토리 합이 아니라 **dataset/ 아래 전부**로 덮어쓴다.
    세션 밖에 예상 못한 파일이 섞여 있으면 actor 가 보고한 숫자와 어긋나 검증에서 걸린다.
    """
    if not ds_dir.is_dir():
        return {"sessions": [], "episodes": 0, "frames": 0, "files": 0, "bytes": 0}, \
               [f"dataset/ 없음: {ds_dir}"]

    subdirs = [p for p in sorted(ds_dir.iterdir()) if p.is_dir()]
    stats, problems = scan_sessions(subdirs)
    stats["files"], stats["bytes"] = count_files(ds_dir)
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


def export_init(ckpt_exp: Path, log: Log, args) -> None:
    """learner 가 뜰 때 θ₀ 를 내보낸다. actor 가 이걸 받아 round 0 을 돈다.

    θ₀ = 비전 인코더 + critic 앙상블 + target critic + edit policy + temperature.

    **action expert LoRA 는 넣지 않는다.** PEFT 의 lora_B 가 0 초기화라 주입 직후
    델타가 정확히 0 이고 (실측: 주입 후 출력 변화 0.00e+00), 그래서 round 0 의 VLA
    출력은 base BC 와 **정확히 같다**. 13.8GB 백본을 올려 0 을 저장할 이유가 없다.
    r000 부터는 학습된 LoRA 가 산출물에 들어간다. actor 쪽 로더는 "있는 키만 채운다"
    라서 이 차이가 코드 경로를 나누지 않는다.

    VLA 자리에 DummyVLA 를 쓰는 이유: EXPOLearner 는 생성 시점에 vla 에서
    action_horizon 하나만 읽는다 (latency+replan 검사). 그래서 θ₀ 를 만들 때는
    base 정책을 로드할 필요가 없다.

    이미 init/DONE 이 있으면 아무것도 하지 않는다 — Job 이 재시작해도 actor 가
    롤아웃 중인 θ₀ 를 바꿔치기하면 안 된다.
    """
    out = ckpt_exp / "init"
    if (out / "DONE").exists():
        log(f"[init] θ₀ 이미 있다 → {out}")
        return
    if not args.exp_config:
        log("[init] --exp-config 가 없어 θ₀ 를 만들지 않는다 (프레임워크 stub 모드)")
        return
    try:
        import torch
        import yaml

        from rl.data import resolve_modality
        from rl.expo import DummyVLA, EXPOLearner, ExpoConfig
        from rl.nets import explore_spec
    except ImportError as e:
        log(f"[init] θ₀ 를 건너뛴다 — {e}")
        log("       L_PY 가 torch 있는 python 3.10 인지, PYTHONPATH 에 rd-rl 과 "
            "third_party/RLDX-1 이 있는지 확인할 것")
        return

    repo = Path(args.repo)
    cfg_path = Path(args.exp_config)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    exp = yaml.safe_load(cfg_path.read_text())
    log(f"[init] {cfg_path.name}  torch {torch.__version__}  python {sys.version.split()[0]}")

    # base 정책 체크포인트와 교차검증한다 — 등록 config 와 순서가 다르면 여기서 죽는다
    # (openarm 은 실제로 modality.json 의 start 순과 모델 순서가 다르다).
    base = args.ckpt_root / exp["base_policy"] if exp.get("base_policy") else None
    if base is not None and not base.is_dir():
        log(f"[init] base 정책이 없어 교차검증을 건너뜀: {base}")
        base = None
    mod, src = resolve_modality(repo, repo / exp["modality"], repo / "third_party" / "RLDX-1",
                                exp["rldx_data_config"], base)
    ecfg = ExpoConfig.from_dict(exp.get("expo"))
    spec = explore_spec(mod.offsets("action"), exp.get("explore_groups") or [],
                        mod.action_dim, int(exp["replan_steps"]))
    seed = int(args.seed)
    L = EXPOLearner(DummyVLA(mod.action_dim, int(exp["action_horizon"])), spec, mod.state_dim,
                    mod.n_cams, int(exp["replan_steps"]), ecfg, device="cpu", seed=seed,
                    latency=int(exp["inference_latency"]))

    out.mkdir(parents=True, exist_ok=True)
    theta = out / "theta.pt"
    torch.save({"enc": L.encoder.state_dict(), "critic": L.critic.state_dict(),
                "target": L.target_critic.state_dict(), "residual": L.residual.state_dict(),
                "temp": L.temp.state_dict()}, theta)
    sha = hashlib.sha256(theta.read_bytes()).hexdigest()
    n = sum(p.numel() for m in (L.encoder, L.critic, L.target_critic, L.residual, L.temp)
            for p in m.parameters())
    log(f"[init] θ₀ {theta.name} {theta.stat().st_size/1e6:.0f} MB  {n/1e6:.2f}M 파라미터")
    log(f"       sha256 {sha[:16]}  seed={seed}  탐색 {list(spec.groups)}")
    log(f"       {src}")

    write_atomic(out / "meta.json", {
        "kind": "init",
        "exp_config": str(cfg_path.relative_to(repo)) if cfg_path.is_relative_to(repo)
                      else str(cfg_path),
        "seed": seed,
        "theta_sha256": sha,
        "params": n,
        "artifacts": ["theta.pt"],
        "keys": ["enc", "critic", "target", "residual", "temp"],
        "lora": "zero-init (포함하지 않음 — 주입 직후 델타가 0 이라 base BC 와 동일)",
        "torch": torch.__version__,
        "expo_deviations": ecfg.deviations(),
        "explore_groups": list(spec.groups),
        "modality_source": src,
        "code": code_provenance(repo),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    write_atomic(out / "DONE", {"kind": "init", "theta_sha256": sha,
                               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    log(f"[init] DONE → {out}  (actor: ./actor/recv_round.py --round init)")


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
    fingerprint = ready_fingerprint(round_dir)

    # 같은 번호를 다시 처리하는 경우 이전 센티넬을 먼저 없앤다 — DONE 과 FAILED 가
    # 동시에 남아 상태가 모순되는 것을 막는다.
    for name in ("DONE", "FAILED"):
        stale = ckpt_round / name
        if stale.exists():
            log(f"[r{n:03d}] 이전 {name} 제거 (READY 가 바뀌어 재처리)")
            stale.unlink()

    try:
        ready = json.loads((round_dir / "READY").read_text())
    except json.JSONDecodeError as exc:
        write_atomic(ckpt_round / "FAILED", {"round": n, "ready_sha": fingerprint,
                                             "reason": f"READY JSON 파싱 실패: {exc}"})
        log(f"[r{n:03d}] FAILED — READY JSON 파싱 실패")
        return "failed"

    stats, problems = validate(round_dir, ready)
    if problems:
        write_atomic(ckpt_round / "FAILED", {"round": n, "ready_sha": fingerprint,
                                             "reason": "검증 실패", "problems": problems})
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
    write_atomic(ckpt_round / "DONE", {"round": n, "stub": True, "ready_sha": fingerprint,
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
    p.add_argument("--exp-config", default="",
                   help="configs/exp/<이름>.yaml. 주면 시작할 때 θ₀ 를 init/ 로 내보낸다")
    p.add_argument("--seed", type=int, default=0, help="θ₀ 초기화 seed (manifest 에 기록된다)")
    p.add_argument("--stub-seconds", type=float, default=5.0, help="stub 학습 소요 시간")
    p.add_argument("--stub-payload-mb", type=float, default=1.0)
    args = p.parse_args()

    runs_exp = args.runs_root / args.exp
    ckpt_exp = args.ckpt_root / "expo" / args.exp
    # 로그는 runs/<exp>/ 밖에 둔다 — 라운드 디렉토리를 지워도 남아야 한다.
    log = Log(args.runs_root / f"{args.exp}.learner.log")

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

    # 폴링 전에 θ₀ 를 내보낸다 — actor 가 이걸 받아야 round 0 을 돌 수 있다.
    export_init(ckpt_exp, log, args)

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
