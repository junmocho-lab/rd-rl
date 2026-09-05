#!/usr/bin/env python3
"""dexjoco sim 온라인 RL 오케스트레이터 — 롤아웃과 learner 라운드를 한 노드에서 왕복시킨다.

fuji 온라인 루프와 같은 라운드 메일박스 규약(learner/loop.py 머리말)을 쓰되, actor 가
같은 파일시스템에 있으므로 kubectl 대신 rename 으로 전달한다. 프로세스 배치:

    GPU 0   정책 서버   rl.vla_rldx serve --artifacts theta_live.pt --artifacts-watch
                        (이 스크립트가 자식 프로세스로 띄운다. 에피소드 경계에서 θ 리로드)
    GPU 1   learner     learner/loop.py (sbatch 가 별도로 띄운다 — 메일박스 폴링)
    CPU+EGL 롤아웃      sim/dexjoco/rollout_dexjoco.py (dexjoco venv, 라운드마다 재실행)

라운드 r 의 흐름:
    1. 롤아웃 k 에피소드 → runs/<run>/rollouts/rollout_rNNN (LeRobot 세션, next.success 포함)
    2. 세션을 runs/<run>/rNNN/dataset/ 로 rename + READY (성공 수 자동 집계)
    3. learner 가 ingest + updates_per_episode x k 회 학습 → ckpt/expo/<run>/rNNN/{theta.pt,DONE}
    4. theta.pt → theta_live.pt 원자 교체 → 서버가 다음 에피소드부터 새 θ
    5. 온라인 에피소드 누적이 round.total_online_episodes 에 닿으면 종료

재시작(선점/타임아웃): 모든 상태가 디스크에 있다 — DONE 라운드는 건너뛰고, READY 만 있는
라운드는 learner 완료를 기다리고, 세션이 반쯤 구른 라운드는 rollout --resume 이 이어 받는다.
같은 라운드는 같은 --seed 를 받아 장면이 재현된다.

usage (sbatch/dexjoco/online/run.sbatch 가 부른다):
    python sim/dexjoco/online_driver.py --exp <run id> \
        --exp-config configs/exp/dexjoco_hammer_nail_d5r20_online.yaml \
        --runs-root runs --ckpt-root checkpoints --port 21500 \
        --sim-py /workspace/junmo_cho/dexjoco/venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import learner.loop as lloop                                   # noqa: E402
from learner.loop import Log, count_files, scan_sessions, write_atomic  # noqa: E402

_server: subprocess.Popen | None = None
_stop = False


def _on_term(signum, _f):
    global _stop
    _stop = True
    print(f"[signal] {signal.Signals(signum).name} — 현재 라운드 후 종료", flush=True)


def episode_stats(session: Path) -> tuple[list[int], list[int]]:
    """에피소드별 (성공, 프레임 수). 파일 순서 = episode_index 순서."""
    import pandas as pd
    succ, frames = [], []
    for f in sorted(session.glob("data/chunk-*/episode_*.parquet")):
        s = pd.read_parquet(f, columns=["next.success"])["next.success"].to_numpy()
        succ.append(int(bool(s.any())))
        frames.append(int(len(s)))
    return succ, frames


def update_curve(runs_exp: Path, hz: float, window: int, log: Log) -> None:
    """train_success_curve.png 재생성 — 디스크의 READY 들이 정본이라 매번 전체를 다시 그린다.

    x축 = **sim 시간(분)** 누적 (에피소드 프레임 합 / hz; 시드 teleop/BC 는 애초에 없다 —
    READY 는 온라인 롤아웃만 담는다). y = 최근 window 에피소드의 성공률 (러닝 평균,
    window 가 차기 전 구간은 그리지 않는다). 라운드가 끝날 때마다 한 번씩 갱신된다.
    """
    succ, frames = [], []
    for d in sorted(runs_exp.glob("r[0-9]*")):
        ready = d / "READY"
        if not ready.is_file():
            continue
        rec = json.loads(ready.read_text())
        es, ef = rec.get("episode_success"), rec.get("episode_frames")
        if es is None:
            continue
        succ += [int(x) for x in es]
        frames += [int(x) for x in (ef or [0] * len(es))]
    if len(succ) < window:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t_min = np.cumsum(frames) / hz / 60.0
    s = np.asarray(succ, dtype=float)
    roll = np.convolve(s, np.ones(window) / window, mode="valid")   # i = window-1 부터
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # 클러스터에 한글 폰트가 없어 라벨은 영어로 (tofu 방지)
    ax.plot(t_min[window - 1:], roll, lw=2, color="tab:blue",
            label=f"success rate (rolling {window} ep)")
    ax.scatter(t_min, s, s=10, alpha=0.35,
               c=np.where(s > 0, "tab:green", "tab:red"),
               label="episode success(1)/fail(0)")
    ax.set_xlabel("sim time (min)")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{runs_exp.name} — online training success "
                 f"({len(s)} eps, overall {s.mean():.0%})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = runs_exp / "train_success_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log(f"[곡선] {out.name} 갱신 — 최근 {window}ep 성공률 {roll[-1]:.0%} "
        f"(sim {t_min[-1]:.1f}분)")


def publish_theta(src: Path, live: Path, log: Log) -> None:
    """theta_live 원자 교체 — 서버는 다음 에피소드 경계(reset)에서 mtime 을 보고 리로드한다."""
    tmp = live.with_name(live.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, live)
    log(f"[theta] {src.parent.name}/{src.name} → {live.name} "
        f"({src.stat().st_size / 1e6:.0f} MB)")


def wait_for(pred, what: str, log: Log, poll: float = 5.0, timeout: float = 7200.0,
             beat: float = 300.0):
    t0 = last = time.time()
    while True:
        r = pred()
        if r:
            return r
        if _stop:
            raise SystemExit(f"[중단] {what} 대기 중 종료 신호")
        if _server is not None and _server.poll() is not None:
            raise SystemExit(f"[오류] 정책 서버가 죽었다 (exit {_server.returncode}) — "
                             f"{what} 대기 중. 서버 로그를 볼 것")
        now = time.time()
        if now - t0 > timeout:
            raise SystemExit(f"[오류] {what} 타임아웃 ({timeout:.0f}s)")
        if now - last >= beat:
            log(f"[대기] {what} ({now - t0:.0f}s 경과)")
            last = now
        time.sleep(poll)


def start_server(args, exp_name: str, live: Path, log: Log) -> subprocess.Popen:
    py = REPO / "third_party/RLDX-1/.venv/bin/python"
    if not py.is_file():
        py = Path(sys.executable)
    cmd = [str(py), "-u", "-m", "rl.vla_rldx", "serve",
           "--exp", exp_name,
           "--model-path", str(args.ckpt_root / args.base_policy),
           "--artifacts", str(live), "--artifacts-watch",
           "--rtc-inference-mode", "trained", "--rtc-exec-horizon", str(args.replan),
           "--sim-wrapper", "--host", "127.0.0.1", "--port", str(args.port),
           "--log-every", "25"]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO / 'third_party/RLDX-1'}:{REPO}"
    if args.server_gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.server_gpu
    slog = (args.runs_root / f"{args.exp}.server.log").open("a")
    log(f"[서버] 기동 (GPU {args.server_gpu or 'default'}, port {args.port}) → "
        f"{args.exp}.server.log")
    return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=slog, stderr=subprocess.STDOUT)


def run_rollout(args, out_dir: Path, episodes: int, seed: int, log: Log) -> None:
    cmd = [str(args.sim_py), "-u", str(REPO / "sim/dexjoco/rollout_dexjoco.py"),
           "--task", args.task, "--episodes", str(episodes),
           "--port", str(args.port), "--replan", str(args.replan),
           "--rtc-delay", str(args.latency),
           "--max-episode-steps", str(args.max_episode_steps),
           "--seed", str(seed), "--log-every", "5",
           "--output", str(out_dir)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "third_party/dexjoco/dexjoco")
    env.setdefault("MUJOCO_GL", "egl")
    log(f"[롤아웃] {out_dir.name}: {episodes}ep seed={seed} (재개 포함)")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[오류] 롤아웃 실패 (exit {r.returncode})")
    log(f"[롤아웃] 완료 {time.time() - t0:.0f}s")


def send_round(runs_exp: Path, n: int, session: Path, collected_by: str, log: Log) -> dict:
    """세션을 rNNN 메일박스로 옮기고 READY 를 쓴다 (fuji actor/send_round.py 의 로컬판)."""
    mb = runs_exp / f"r{n:03d}"
    ds = mb / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    dst = ds / session.name
    if session.is_dir():                      # 재실행이면 이미 옮겨져 있을 수 있다
        session.rename(dst)
    if not dst.is_dir():
        raise SystemExit(f"[오류] 세션이 없다: {session} / {dst}")

    stats, problems = scan_sessions([dst])
    if problems:
        raise SystemExit(f"[오류] 세션 검증 실패: {problems}")
    stats["files"], stats["bytes"] = count_files(ds)
    succ, frames = episode_stats(dst)
    ready = {"round": n, "collected_by": collected_by,
             "sessions": stats["sessions"], "episodes": stats["episodes"],
             "frames": stats["frames"], "files": stats["files"], "bytes": stats["bytes"],
             "success": int(sum(succ)), "episode_success": succ,
             "episode_frames": frames,
             "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_atomic(mb / "READY", ready)
    log(f"[r{n:03d}] READY — {stats['episodes']}ep / {stats['frames']}프레임 / "
        f"성공 {sum(succ)}/{len(succ)} {succ}")
    return ready


def main() -> int:
    global _server
    import yaml

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True, help="run id (메일박스/wandb/산출물 이름)")
    p.add_argument("--exp-config", required=True)
    p.add_argument("--runs-root", type=Path, default=REPO / "runs")
    p.add_argument("--ckpt-root", type=Path, default=REPO / "checkpoints")
    p.add_argument("--port", type=int, default=21500)
    p.add_argument("--sim-py", type=Path,
                   default=Path("/workspace/junmo_cho/dexjoco/venv/bin/python"))
    p.add_argument("--task", default="hammer_nail")
    p.add_argument("--server-gpu", default="0",
                   help="서버가 쓸 CUDA_VISIBLE_DEVICES (빈 문자열이면 상속)")
    p.add_argument("--seed-base", type=int, default=1000,
                   help="라운드 r 의 롤아웃 seed = seed_base + 시작 에피소드 번호 — "
                        "장면이 에피소드 전역 번호에 결정적으로 묶인다")
    p.add_argument("--expected-hw", default="256x256")
    p.add_argument("--learn-timeout", type=float, default=7200.0)
    args = p.parse_args()

    w, h = (int(v) for v in args.expected_hw.lower().split("x"))
    lloop.EXPECTED_HW = (h, w)

    exp = yaml.safe_load((REPO / args.exp_config).read_text())
    rc = exp.get("round") or {}
    args.replan = int(exp["replan_steps"])
    args.latency = int(exp["inference_latency"])
    args.base_policy = exp["base_policy"]
    args.max_episode_steps = int(rc.get("max_episode_steps", 360))
    first = int(rc.get("episodes_first_round", 10))
    per = int(rc.get("episodes_per_round", 2))
    total = int(rc.get("total_online_episodes", 100))
    hz = float(rc.get("control_hz", 50))
    window = int(rc.get("success_window", 20))

    runs_exp = args.runs_root / args.exp
    ckpt_exp = args.ckpt_root / "expo" / args.exp
    live = runs_exp / "theta_live.pt"
    (runs_exp / "rollouts").mkdir(parents=True, exist_ok=True)
    log = Log(args.runs_root / f"{args.exp}.driver.log")
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    log("=" * 70)
    log(f"driver 시작  exp={args.exp}  config={args.exp_config}")
    log(f"  라운드 크기: 첫 {first}ep, 이후 {per}ep — 총 {total}ep 까지")
    log(f"  replan={args.replan} latency={args.latency} "
        f"max_steps={args.max_episode_steps} task={args.task}")
    if not args.sim_py.is_file():
        raise SystemExit(f"[오류] dexjoco venv python 이 없다: {args.sim_py}")

    # ── 재시작 상태 복원: DONE 라운드의 에피소드 수를 세고, 다음 라운드 번호를 정한다 ──
    done_eps, next_round, pending = 0, 0, None
    for d in sorted(runs_exp.glob("r[0-9]*")):
        n = lloop.round_num(d)
        if n is None or not (d / "READY").is_file():
            continue
        eps = int(json.loads((d / "READY").read_text()).get("episodes", 0))
        if (ckpt_exp / d.name / "DONE").exists():
            done_eps += eps
            next_round = max(next_round, n + 1)
        elif (ckpt_exp / d.name / "FAILED").exists():
            raise SystemExit(f"[오류] r{n:03d} 이 FAILED 다 — 원인 해결 후 해당 라운드 "
                             f"디렉토리를 지우고 재시작할 것")
        else:
            pending = (n, eps)
            next_round = max(next_round, n + 1)
    if done_eps or pending:
        log(f"[재개] 완료 {done_eps}ep, 대기 중 라운드 {pending}, 다음 r{next_round:03d}")

    # ── learner 의 θ₀ 를 기다렸다가 최신 θ 를 live 로 게시하고 서버를 띄운다 ────────
    wait_for(lambda: (ckpt_exp / "init" / "DONE").exists(),
             "learner θ₀ (init/DONE)", log, timeout=3600)
    thetas = sorted(ckpt_exp.glob("r*/theta.pt"))
    latest = thetas[-1] if thetas else ckpt_exp / "init" / "theta.pt"
    publish_theta(latest, live, log)
    _server = start_server(args, exp["name"], live, log)

    # ── 대기 중이던 라운드(READY 만 있음)를 learner 가 끝내길 기다린다 ───────────────
    if pending is not None:
        n, eps = pending
        wait_for(lambda: (ckpt_exp / f"r{n:03d}" / "DONE").exists()
                 or (ckpt_exp / f"r{n:03d}" / "FAILED").exists(),
                 f"r{n:03d} 학습", log, timeout=args.learn_timeout)
        if (ckpt_exp / f"r{n:03d}" / "FAILED").exists():
            raise SystemExit(f"[오류] r{n:03d} 학습 실패 — learner 로그를 볼 것")
        done_eps += eps
        publish_theta(ckpt_exp / f"r{n:03d}" / "theta.pt", live, log)

    update_curve(runs_exp, hz, window, log)     # 재시작 시 기존 라운드들로 곡선 복원

    # ── 라운드 루프 ──────────────────────────────────────────────────────────────
    n = next_round
    while done_eps < total and not _stop:
        k = first if done_eps == 0 else min(per, total - done_eps)
        sess = runs_exp / "rollouts" / f"rollout_r{n:03d}"
        moved = runs_exp / f"r{n:03d}" / "dataset" / sess.name
        collected_by = f"r{n-1:03d}" if n > 0 else "init"

        if not moved.is_dir():
            run_rollout(args, sess, k, args.seed_base + done_eps, log)
        else:
            log(f"[r{n:03d}] 세션이 이미 메일박스에 있다 — 롤아웃 생략 (재시작 이어받기)")
        send_round(runs_exp, n, sess, collected_by, log)

        wait_for(lambda rn=n: (ckpt_exp / f"r{rn:03d}" / "DONE").exists()
                 or (ckpt_exp / f"r{rn:03d}" / "FAILED").exists(),
                 f"r{n:03d} 학습", log, timeout=args.learn_timeout)
        if (ckpt_exp / f"r{n:03d}" / "FAILED").exists():
            raise SystemExit(f"[오류] r{n:03d} 학습 실패 — learner 로그를 볼 것")
        publish_theta(ckpt_exp / f"r{n:03d}" / "theta.pt", live, log)
        update_curve(runs_exp, hz, window, log)

        done_eps += k
        log(f"[진행] 온라인 에피소드 {done_eps}/{total}")
        n += 1

    log(f"driver 종료 — 온라인 에피소드 {done_eps}개 / 라운드 {n}개")
    if _server is not None and _server.poll() is None:
        _server.terminate()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if _server is not None and _server.poll() is None:
            _server.terminate()
