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
    """rank 0 만 기록한다.

    torchrun 아래에서는 4개 프로세스가 같은 라운드를 같은 순서로 처리하므로 4벌을 남기면
    로그가 4배가 될 뿐 새 정보가 없다. rank 별로 갈리는 상황(한 rank 만 죽는다)은 예외
    traceback 이 stderr 로 나오고 torchrun 이 거기에 rank 를 붙여준다.
    """

    def __init__(self, path: Path | None, enabled: bool = True):
        self.path = path if enabled else None
        self.enabled = enabled
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        if not self.enabled:
            return
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
                        mod.action_dim, int(exp["replan_steps"]),
                        int(exp["inference_latency"]))
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


class Trainer:
    """라운드 학습. VLA(13.8GB)·학습기·버퍼는 프로세스 수명 동안 한 번만 만든다.

    버퍼는 $L_RUNS/<exp>/buffer/ 에 둔다:
        sessions.json     ingest 순서의 세션 경로 — **이 순서가 memmap 의 프레임 순서다**
        images.mm(.json)  디코딩된 uint8 프레임 (라운드마다 뒤에만 붙는다)
        actnorm.npy       모델 공간으로 정규화된 액션 청크 (critic 입력)

    액션 공간 규약이 제일 중요하다:
        critic / residual 의 action  → **모델 공간**. vla.sample() 이 내는 후보와 같은
                                       공간이어야 Q argmax 가 의미를 갖는다
        actor BC 의 target           → **raw LeRobot**. vla.train_step 이 안에서 정규화한다
    뒤집으면 에러 없이 조용히 학습이 망가진다.

    학습 때 RTC 는 끈다 (rtc_inference_mode="none"). EXPO-FT 의 타깃 계산은 prefix 없이
    후보를 뽑고, critic 이 보는 구간은 어차피 [latency, latency+replan) 로 같다.
    """

    def __init__(self, args, log: Log, device=None):
        self.args, self.log = args, log
        self.buf = args.runs_root / args.exp / "buffer"
        self.ckpt_exp = args.ckpt_root / "expo" / args.exp
        self.built = False
        self.flat = self.imgs = self.actnorm = self.statenorm = None
        self.task = ""
        # rank 마다 자기 GPU. 단일 프로세스면 cuda:0.
        self.device = str(device) if device is not None else "cuda"

    # --- 1회 생성 -----------------------------------------------------------
    def _build(self) -> None:
        if self.built:
            return
        import torch
        import numpy as np
        import yaml

        from rl import ddp
        from rl.data import resolve_modality
        from rl.expo import EXPOLearner, ExpoConfig
        from rl.nets import explore_spec
        from rl.vla_rldx import RLDXVLA

        repo = Path(self.args.repo)
        cfg_path = Path(self.args.exp_config)
        if not cfg_path.is_absolute():
            cfg_path = repo / cfg_path
        exp = yaml.safe_load(cfg_path.read_text())
        self.exp = exp
        self.repo = repo
        self.replan = int(exp["replan_steps"])
        self.latency = int(exp["inference_latency"])
        self.horizon = int(exp["action_horizon"])
        self.rnd_cfg = exp.get("round") or {}
        self.rldx_root = repo / "third_party" / "RLDX-1"

        base = self.args.ckpt_root / exp["base_policy"]
        if not base.is_dir():
            raise SystemExit(f"base 정책이 없다: {base}  (configs/exp 의 base_policy 확인)")
        mod, src = resolve_modality(repo, repo / exp["modality"], self.rldx_root,
                                    exp["rldx_data_config"], base)
        self.mod = mod
        self.cfg = ExpoConfig.from_dict(exp.get("expo"))
        spec = explore_spec(mod.offsets("action"), exp.get("explore_groups") or [],
                            mod.action_dim, self.replan, self.latency)
        self.log(f"[학습] {src}")
        self.log(f"[학습] 정책 로드 {base.name}")
        t0 = time.time()
        self.vla = RLDXVLA(base, mod, self.rldx_root, exp["rldx_data_config"],
                           device=self.device, rtc_inference_mode="none")
        info = self.vla.setup_training(lr=float((exp.get("vla") or {}).get("lora_lr", 3e-4)))
        self.L = EXPOLearner(self.vla, spec, mod.state_dim, mod.n_cams, self.replan, self.cfg,
                             device=self.device, seed=int(self.args.seed), latency=self.latency)
        self.log(f"[학습] {time.time()-t0:.0f}s  LoRA {info['trainable_params']/1e6:.2f}M "
                 f"(백본 trainable {info['backbone_trainable_tensors']})  "
                 f"탐색 {list(spec.groups)} {spec.active_dim}/{mod.action_dim}차원")
        self._load_theta()

        # rank 0 의 파라미터로 전부 맞춘다. 여기서 어긋난 채 시작하면 gradient 만 평균되고
        # 파라미터는 영원히 갈라진 상태로 학습된다 (rl/ddp.py 머리말 참고).
        ddp.broadcast_params([self.L.encoder, self.L.critic, self.L.target_critic,
                              self.L.residual, self.L.temp, self.vla.model])

        # 배치 추출만 rank 마다 달라야 한다 — 같으면 4장이 똑같은 미니배치를 돌아서
        # 실효 배치가 늘지 않는다. 스트림이 겹치지 않게 rank 마다 멀리 떨어뜨린다.
        self.rng = np.random.default_rng(int(self.args.seed) + 1_000_003 * ddp.rank())
        if ddp.enabled():
            self.log(f"[학습] 분산 world={ddp.world_size()}  "
                     f"critic 미니배치 {self.cfg.batch_size}/rank "
                     f"→ 실효 {self.cfg.batch_size * ddp.world_size()}")
        self.updates = 0
        self._wandb_init(exp, spec, base)
        self.built = True

    def _wandb_init(self, exp: dict, spec, base: Path) -> None:
        """WANDB_API_KEY 가 있으면 붙는다 (없으면 조용히 로그 파일만).

        id=run id / resume="allow" 로 두는 이유: Job 이 재시작해도 같은 wandb run 에
        이어 붙어야 라운드 간 추세가 끊기지 않는다 (버퍼·산출물도 run id 로 이어받는다).
        """
        from rl import ddp
        self.wb = None
        if not ddp.is_main():
            return                    # 4개 rank 가 같은 run id 로 붙으면 서로 덮어쓴다
        if self.args.no_wandb or not os.environ.get("WANDB_API_KEY"):
            self.log("[wandb] WANDB_API_KEY 가 없어 붙지 않는다 (로그 파일만)")
            return
        try:
            import wandb
        except ImportError:
            self.log("[wandb] wandb 가 설치돼 있지 않다 — 건너뛴다")
            return
        self.wb = wandb.init(
            project=self.args.wandb_project, id=self.args.exp, name=self.args.exp,
            resume="allow", config={
                "exp": exp.get("name"), "robot": exp.get("robot"),
                "base_policy": exp.get("base_policy"), "run_id": self.args.exp,
                "action_horizon": self.horizon, "replan_steps": self.replan,
                "inference_latency": self.latency,
                "explore_groups": list(spec.groups), "active_dim": spec.active_dim,
                "target_entropy": spec.target_entropy,
                "seed": int(self.args.seed),
                # expo.batch_size 는 **rank 당** 값이다. 실제로 한 critic 스텝이 보는
                # 표본 수는 아래 effective_batch_size — GPU 수를 바꾸면 이게 바뀐다.
                "world_size": ddp.world_size(),
                "effective_batch_size": self.cfg.batch_size * ddp.world_size(),
                **{f"expo.{k}": v for k, v in vars(self.cfg).items()},
                **{f"round.{k}": v for k, v in self.rnd_cfg.items()},
            })
        self.log(f"[wandb] {self.args.wandb_project}/{self.args.exp} → {self.wb.url}")

    def _latest_theta(self) -> Path | None:
        """가장 최근 산출물. 프로세스가 살아있는 동안은 메모리 상태가 최신이지만, Job 이
        재시작하면 여기서 이어받아야 한다."""
        cands = sorted(self.ckpt_exp.glob("r*/theta.pt"))
        if cands:
            return cands[-1]
        p = self.ckpt_exp / "init" / "theta.pt"
        return p if p.is_file() else None

    def _load_theta(self) -> None:
        import torch
        path = self._latest_theta()
        if path is None:
            self.log("[학습] 이어받을 산출물이 없다 — 방금 만든 θ 로 시작한다")
            return
        sd = torch.load(path, map_location=self.L.device, weights_only=True)
        pairs = {"enc": self.L.encoder, "critic": self.L.critic, "target": self.L.target_critic,
                 "residual": self.L.residual, "temp": self.L.temp}
        got = [k for k in pairs if k in sd]
        for k in got:
            pairs[k].load_state_dict(sd[k])
        if "lora" in sd:
            missing = self.vla.model.load_state_dict(sd["lora"], strict=False)
            got.append(f"lora({len(sd['lora'])}텐서)")
            if missing.unexpected_keys:
                raise SystemExit(f"lora 키가 모델에 없다: {missing.unexpected_keys[:3]}")
        self.log(f"[학습] 이어받기 {path.parent.name}/{path.name} → {got}")

    # --- 버퍼 ---------------------------------------------------------------
    def _seed_records(self) -> list[dict]:
        """시드 데모(teleop) 세션 기록. 버퍼가 비어 있을 때 **맨 앞에** 한 번만 들어간다.

        EXPO-FT 의 `--num_data=10` 자리다 (train_pi_robo.py → BatchProcessor 가
        offline_ratio=0 일 때 `replay_buffer.insert_dataset(dataset)` 으로 온라인 버퍼에
        바로 섞는다). 원본은 성공 데모 디렉토리를 통째로 받지만 우리 teleop 세션은
        성공·실패가 섞여 있어서 여기서 고른다.

        맨 앞에 고정해야 하는 이유: images.mm 은 세션 순서대로 이어붙인 memmap 이고
        이어받기가 **접두사 일치**로 판정된다. 시드가 중간에 끼면 접두사가 깨져 매 라운드
        수십 GB 를 다시 디코딩한다.
        """
        from rl.data import find_sessions, select_seed_episodes

        rc = self.rnd_cfg
        root = rc.get("seed_dataset")
        if not root:
            return []
        n = int(rc.get("seed_teleop_episodes", 0))
        success_only = bool(rc.get("seed_success_only", True))
        path = Path(root)
        if not path.is_absolute():
            path = self.repo / path
        if not path.is_dir():
            raise SystemExit(f"round.seed_dataset 이 없다: {path}")

        out = []
        for s in find_sessions(path):
            eps = select_seed_episodes(s, n, success_only)
            if not eps:
                self.log(f"[시드] {s.name}: 담을 에피소드가 없다 "
                         f"(success_only={success_only}) — 건너뛴다")
                continue
            out.append({"path": str(s), "kind": "seed", "episodes": eps})
            self.log(f"[시드] {s.name}: {len(eps)}개 담는다 {eps}"
                     + (f"  (요청 {n}개, 성공만)" if success_only and len(eps) < n else ""))
        return out

    def ingest(self, round_dir: Path, round_no: int) -> dict:
        import numpy as np
        from rl import ddp
        from rl.data import build_flat, build_images, find_sessions, open_images
        from rl.vla_rldx import normalize_states
        from rl.offline_critic import normalize_all

        manifest = self.buf / "sessions.json"
        records = new = None
        if ddp.is_main():
            self.buf.mkdir(parents=True, exist_ok=True)
            records = json.loads(manifest.read_text()) if manifest.is_file() else []
            # 예전 형식(문자열 목록)을 만나면 online 으로 읽는다.
            records = [r if isinstance(r, dict) else {"path": r, "kind": "online"}
                       for r in records]
            if not records:
                records = self._seed_records()
            have = {Path(r["path"]).name for r in records}
            new = [{"path": str(s), "kind": "online", "round": round_no}
                   for s in find_sessions(round_dir / "dataset") if s.name not in have]
            records += new
            new = len(new)
        # 매니페스트는 여기서 쓰지 않는다 — 이 세션들이 실제로 읽히는지(해상도·카메라·
        # 파손 여부) 아래 build_flat/build_images 를 통과해야 안다. 먼저 커밋하면 못 쓰는
        # 세션이 버퍼에 영구히 등록돼서 **이후 모든 라운드가 같은 파일에서 계속 실패한다.**
        records, new = ddp.broadcast_object((records, new))

        paths = [Path(r["path"]) for r in records]
        keep = {Path(r["path"]).name: r["episodes"] for r in records if r.get("episodes")}
        gone = [p for p in paths if not p.is_dir()]
        if gone:
            raise SystemExit(f"버퍼에 등록된 세션이 사라졌다 (순서가 깨지면 memmap 이 어긋난다): "
                             f"{[p.name for p in gone][:3]}")
        n_seed_s = sum(1 for r in records if r["kind"] == "seed")
        self.log(f"[버퍼] 새 세션 {new}개 / 누적 {len(paths)}개 "
                 f"(시드 {n_seed_s} / 온라인 {len(paths)-n_seed_s})")

        self.flat = build_flat(paths, self.mod, keep=keep)

        # 비디오 디코딩과 memmap·캐시 쓰기는 rank 0 만 (4개가 같은 파일에 쓰면 깨진다).
        # 여기서 나는 예외를 그냥 올리면 rank 0 만 빠져나가고 나머지 3개는 아래 집합통신에서
        # 영원히 기다린다. 그래서 잡아서 전 rank 에 알린 뒤 **같이** 실패한다.
        err = None
        if ddp.is_main():
            try:
                build_images(paths, self.flat, self.buf / "images.mm", self.mod, resume=True)
                normalize_all(self.vla, self.flat, self.horizon, cache=self.buf / "actnorm.npy")
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        errs = [e for e in ddp.gather_object(err) if e]
        if errs:
            raise RuntimeError(f"버퍼 준비 실패 (매니페스트에 등록하지 않는다): {errs[0]}")

        # 여기까지 왔으면 이 세션들은 실제로 읽힌다. 그때 비로소 버퍼 등록을 확정한다.
        if ddp.is_main():
            manifest.write_text(json.dumps(records, indent=2) + "\n")

        self.imgs, _ = open_images(self.buf / "images.mm")
        self.actnorm = normalize_all(self.vla, self.flat, self.horizon,
                                     cache=self.buf / "actnorm.npy")
        # state 도 모델 공간으로. 벡터 연산 한 번이라 캐시하지 않는다 (액션 청크와 달리
        # 프레임당 python 루프가 없다). rl/vla_rldx.normalize_states 의 주석 참고.
        self.statenorm = normalize_states(self.vla.proc, self.vla.tag, self.mod, self.flat.state)
        tasks = json.loads((paths[0] / "meta" / "tasks.jsonl").read_text().splitlines()[0])
        self.task = tasks["task"]

        self.succ_idx = np.nonzero(self.flat.is_success)[0]
        n_ep = len(self.flat.ep_length)
        n_succ_ep = int(sum(self.flat.ep_success))
        # 시드는 게이트 계산에서 빼야 한다. EXPO-FT 의 can_update 는 training_log.ep_count
        # (온라인 롤아웃만; insert_dataset 은 이걸 올리지 않는다) 를 본다.
        n_seed_ep = sum(len(r["episodes"]) for r in records
                        if r["kind"] == "seed" and r.get("episodes"))
        n_online_ep = n_ep - n_seed_ep
        self.log(f"[버퍼] 프레임 {len(self.flat)}  에피소드 {n_ep} "
                 f"(시드 {n_seed_ep} + 온라인 {n_online_ep}, 성공 {n_succ_ep})  "
                 f"성공 프레임 {len(self.succ_idx)}")
        return {"sessions": len(paths), "new_sessions": new,
                "frames": len(self.flat), "episodes": n_ep,
                "seed_episodes": n_seed_ep, "online_episodes": n_online_ep,
                "success_episodes": n_succ_ep}

    # --- 배치 ---------------------------------------------------------------
    def _batch(self, n: int, success_only: bool = False) -> dict:
        import torch
        import numpy as np
        from rl.data import make_batch

        # 마지막 replan_steps 프레임은 뽑지 않는다 — nstep 이 t+replan 까지 읽으므로
        # 버퍼 끝을 넘어간다 (rl/offline_critic.py 와 rl/expo.py 의 표본 추출도 같은 규칙).
        hi = len(self.flat) - self.replan
        if hi <= 0:
            raise SystemExit(f"버퍼가 너무 작다: {len(self.flat)} 프레임 <= replan {self.replan}")
        if success_only:
            pool = self.succ_idx[self.succ_idx < hi]
            if len(pool) == 0:
                raise SystemExit("actor_success_only 인데 쓸 수 있는 성공 프레임이 0개다 "
                                 "(round.seed_teleop_episodes 로 성공 시연을 시드할 것)")
            idx = pool[self.rng.integers(0, len(pool), size=n)]
        else:
            idx = self.rng.integers(0, hi, size=n)
        b = make_batch(self.flat, self.imgs, idx, self.mod, replan_steps=self.replan,
                       action_horizon=self.horizon, discount=self.cfg.discount,
                       task=self.task, latency=self.latency)
        # critic 이 보는 액션/상태를 모델 공간으로 갈아끼운다. full_action(actor BC 대상) 은 raw.
        L, R, A = self.latency, self.replan, self.mod.action_dim
        b["action"] = self.actnorm[idx][:, :L + R].reshape(len(idx), (L + R) * A)
        b["next_action_prefix"] = self.actnorm[b["next_idx"]][:, :L].reshape(len(idx), L * A)
        b["state"] = self.statenorm[idx]
        b["next_state"] = self.statenorm[b["next_idx"]]
        dev = self.L.device
        out = {}
        for k, v in b.items():
            if k in ("full_action", "vla_obs", "vla_next_obs"):
                out[k] = v                                   # numpy 로 둔다 (VLA 가 받는다)
            elif isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(np.ascontiguousarray(v)).to(dev)
            else:
                out[k] = v
        return out

    # --- 라운드 -------------------------------------------------------------
    def run_round(self, round_dir: Path, ckpt_round: Path, stats: dict, ready: dict) -> None:
        self._build()
        buf = self.ingest(round_dir, int(ready.get("round", 0)))
        c, rc = self.cfg, self.rnd_cfg
        min_ep = int(rc.get("min_online_episodes", 0))
        n_updates = int(rc.get("updates_per_episode", 1)) * int(stats["episodes"])
        # 게이트는 **온라인 롤아웃 에피소드 누적** 으로 센다. 시드 teleop 은 세지 않는다 —
        # EXPO-FT 의 can_update 가 보는 training_log.ep_count 가 그렇고(insert_dataset 은
        # 올리지 않는다), 시드를 세면 데이터를 한 번도 모으기 전에 학습이 시작된다.
        if buf["online_episodes"] < min_ep:
            self.log(f"[학습] 온라인 에피소드 {buf['online_episodes']} < "
                     f"min_online_episodes {min_ep} — 이번 라운드는 버퍼에만 넣고 학습은 "
                     f"건너뛴다 (EXPO-FT can_update). 시드 {buf['seed_episodes']}개는 "
                     f"세지 않는다")
            n_updates = 0

        t0 = time.time()
        last: dict = {}
        for i in range(n_updates):
            last = self.L.update(self._batch(c.batch_size * c.utd_ratio),
                                 actor_batch=(self._batch(c.batch_size, success_only=True)
                                              if c.actor_success_only else None))
            self.updates += 1
            if self.wb is not None:
                self.wb.log({f"train/{k}": v for k, v in last.items()
                             if isinstance(v, (int, float))},
                            step=self.updates)
            if i == 0 or (i + 1) % 5 == 0 or i == n_updates - 1:
                self.log(f"  [{i+1}/{n_updates}] critic_loss={last.get('critic_loss', 0):.4f} "
                         f"q={last.get('q', 0):+.3f} q_max={last.get('q_max', 0):+.3f} "
                         f"actor_loss={last.get('actor_loss', 0):.4f} "
                         f"후보Q std={last.get('candidate_q_std', 0):.4f} "
                         f"edit선택={last.get('select_ratio_with_residual', 0):.2f} "
                         f"{(time.time()-t0)/(i+1):.1f}s/회")
        if n_updates:
            self.log(f"[학습] {n_updates}회 {time.time()-t0:.0f}s  steps={self.L.steps}")
            if last.get("q_max", 0) > 1.2:
                self.log(f"  [경고] q_max={last['q_max']:.2f} > 1.2 — sparse terminal reward 의 "
                         "리턴 상한이 1 이므로 발산 신호다")
        if self.wb is not None:
            n_ep = int(stats["episodes"])
            succ = ready.get("success")
            row = {"round/number": ready.get("round"), "round/episodes": n_ep,
                   "round/updates": n_updates, "round/seconds": round(time.time() - t0, 1),
                   "buffer/frames": buf["frames"], "buffer/episodes": buf["episodes"],
                   "buffer/seed_episodes": buf["seed_episodes"],
                   "buffer/online_episodes": buf["online_episodes"],
                   "buffer/success_episodes": buf["success_episodes"]}
            # 사람이 라벨한 성공 수 — 이 루프의 실제 목표다. --success 를 준 라운드만 남는다.
            if succ is not None and n_ep:
                row["round/success"] = int(succ)
                row["round/success_rate"] = int(succ) / n_ep
            self.wb.log(row, step=self.updates)
        self.export(ckpt_round, stats, ready, buf, n_updates, last)

    def export(self, ckpt_round: Path, stats: dict, ready: dict, buf: dict,
               n_updates: int, last: dict) -> None:
        import torch
        from rl import ddp
        # 파라미터는 rank 사이에서 같으므로 rank 0 것을 쓰면 된다 (rl/ddp.py 머리말).
        if not ddp.is_main():
            return
        ckpt_round.mkdir(parents=True, exist_ok=True)
        lora = {k: v.detach().cpu() for k, v in self.vla.model.state_dict().items()
                if "lora_" in k}
        theta = ckpt_round / "theta.pt"
        torch.save({"enc": self.L.encoder.state_dict(), "critic": self.L.critic.state_dict(),
                    "target": self.L.target_critic.state_dict(),
                    "residual": self.L.residual.state_dict(), "temp": self.L.temp.state_dict(),
                    "lora": lora}, theta)
        sha = hashlib.sha256(theta.read_bytes()).hexdigest()
        self.log(f"[산출물] theta.pt {theta.stat().st_size/1e6:.0f} MB "
                 f"(lora {len(lora)}텐서)  sha256 {sha[:16]}")
        write_atomic(ckpt_round / "meta.json", {
            "round": ready.get("round"),
            "collected_by": ready.get("collected_by"),
            "dataset": {k: stats[k] for k in ("sessions", "episodes", "frames", "files", "bytes")},
            "buffer": buf,
            "updates": n_updates,
            "steps": self.L.steps,
            "metrics": {k: last[k] for k in sorted(last) if isinstance(last[k], float)},
            "theta_sha256": sha,
            "artifacts": ["theta.pt"],
            "keys": ["enc", "critic", "target", "residual", "temp", "lora"],
            "seed": int(self.args.seed),
            "torch": torch.__version__,
            # expo.batch_size 는 rank 당 값이다. 실제로 한 critic 스텝이 본 표본 수는
            # 이 둘의 곱이므로 라운드마다 기록해 둔다 (GPU 수를 바꾸면 달라진다).
            "world_size": ddp.world_size(),
            "effective_batch_size": self.cfg.batch_size * ddp.world_size(),
            "expo_deviations": self.cfg.deviations(),
            "code": code_provenance(Path(self.args.repo)),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })


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
def process_round(n: int, round_dir: Path, ckpt_round: Path, log: Log, args,
                  trainer: "Trainer | None" = None) -> str:
    from rl import ddp
    log(f"[r{n:03d}] 감지: {round_dir}")

    # 검증과 센티넬 쓰기는 rank 0 만 한다. rank 마다 따로 validate 하면 파일이 아직
    # 늘어나는 중일 때 서로 다른 결론을 낼 수 있고, FAILED 를 4개가 동시에 쓰게 된다.
    verdict = None
    if ddp.is_main():
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
            verdict = {"ok": False}

        if verdict is None:
            stats, problems = validate(round_dir, ready)
            if problems:
                write_atomic(ckpt_round / "FAILED", {"round": n, "ready_sha": fingerprint,
                                                     "reason": "검증 실패", "problems": problems})
                for p in problems:
                    log(f"[r{n:03d}]   ✗ {p}")
                log(f"[r{n:03d}] FAILED — 학습하지 않음")
                verdict = {"ok": False}
            else:
                verdict = {"ok": True, "ready": ready, "stats": stats,
                           "fingerprint": fingerprint}

    verdict = ddp.broadcast_object(verdict)
    if not verdict["ok"]:
        return "failed"
    ready, stats, fingerprint = verdict["ready"], verdict["stats"], verdict["fingerprint"]

    log(f"[r{n:03d}] 검증 통과: 세션 {len(stats['sessions'])}개 / 에피소드 {stats['episodes']} / "
        f"프레임 {stats['frames']} / {stats['bytes']/1e6:.1f} MB")

    if trainer is not None:
        err = None
        try:
            trainer.run_round(round_dir, ckpt_round, stats, ready)
        except Exception as exc:                       # 학습 실패를 라운드 상태로 남긴다
            import traceback
            err = (str(exc), traceback.format_exc())
            if not ddp.is_main():
                # rank 0 이 아니면 로그가 꺼져 있다. torchrun 이 rank 를 붙여 stderr 로 낸다.
                print(f"[rank{ddp.rank()}] r{n:03d} 학습 실패\n{err[1]}",
                      file=sys.stderr, flush=True)
        # 실패는 rank 사이에서 합의한다 — 한쪽만 FAILED 로 빠져나가면 다음 라운드에서
        # 서로 다른 지점의 all-reduce 를 만나 멈춘다. (보통은 데이터·shape 문제라 모든
        # rank 가 같은 자리에서 같이 실패한다.)
        anyerr = any(x is not None for x in ddp.gather_object(err))
        if anyerr:
            if ddp.is_main():
                reason, tb = err if err else ("다른 rank 에서 실패", "")
                write_atomic(ckpt_round / "FAILED", {"round": n, "ready_sha": fingerprint,
                                                     "reason": f"학습 실패: {reason}",
                                                     "traceback": tb.splitlines()[-12:]})
                for line in tb.splitlines()[-12:]:
                    log(f"[r{n:03d}]   {line}")
                log(f"[r{n:03d}] FAILED — 학습 실패")
            return "failed"
        stub = False
    else:
        planned = args.updates_per_episode * stats["episodes"]
        log(f"[r{n:03d}] [STUB] --exp-config 가 없어 학습을 생략한다. 실제로는 "
            f"update() {planned}회")
        time.sleep(args.stub_seconds)
        if ddp.is_main():
            export_stub(ckpt_round, log, args, stats, ready)
        stub = True

    # DONE 은 actor 가 기다리는 신호다. 산출물(theta.pt)을 다 쓴 뒤에, rank 0 만 쓴다.
    ddp.barrier()
    if ddp.is_main():
        write_atomic(ckpt_round / "DONE", {"round": n, "stub": stub, "ready_sha": fingerprint,
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
    p.add_argument("--wandb-project", default="rd-rl-expo")
    p.add_argument("--no-wandb", action="store_true", help="WANDB_API_KEY 가 있어도 붙지 않는다")
    p.add_argument("--stub-seconds", type=float, default=5.0, help="stub 학습 소요 시간")
    p.add_argument("--stub-payload-mb", type=float, default=1.0)
    args = p.parse_args()

    # torchrun 으로 띄우면 rank 마다 GPU 하나씩. 단일 프로세스로 띄우면 world=1 이라
    # 아래 분기들이 전부 통과 상태가 된다 (같은 코드가 양쪽에서 돈다).
    from rl import ddp
    rank, world, device = ddp.init()

    runs_exp = args.runs_root / args.exp
    ckpt_exp = args.ckpt_root / "expo" / args.exp
    # 로그는 runs/<exp>/ 밖에 둔다 — 라운드 디렉토리를 지워도 남아야 한다.
    log = Log(args.runs_root / f"{args.exp}.learner.log", enabled=ddp.is_main())

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    log("=" * 70)
    log(f"learner 시작  exp={args.exp}")
    log(f"  python   {sys.version.split()[0]}  pid={os.getpid()}")
    log(f"  분산     world={world} device={device}"
        + (f"  실효 배치 = expo.batch_size × {world}" if world > 1 else "  (단일 GPU)"))
    log(f"  runs     {runs_exp}")
    log(f"  ckpt     {ckpt_exp}")
    log(f"  code     {code_provenance(Path(args.repo))}")
    if ddp.is_main():
        runs_exp.mkdir(parents=True, exist_ok=True)
        ckpt_exp.mkdir(parents=True, exist_ok=True)
    ddp.barrier()

    # 폴링 전에 θ₀ 를 내보낸다 — actor 가 이걸 받아야 round 0 을 돌 수 있다.
    # 파일을 쓰는 것은 rank 0 뿐이다 (같은 경로에 4개가 동시에 쓰면 안 된다).
    if ddp.is_main():
        export_init(ckpt_exp, log, args)
    ddp.barrier()

    # VLA(13.8GB) 와 학습기는 프로세스 수명 동안 한 번만 만든다 (라운드마다 재로드 금지).
    # 실제 로드는 첫 라운드가 올 때 (_build) — 데이터가 없으면 GPU 를 점유하지 않는다.
    trainer = Trainer(args, log, device=device) if args.exp_config else None
    if trainer is None:
        log("[경고] --exp-config 가 없어 학습 stub 으로 동작한다")

    last_beat = 0.0
    processed = 0
    while True:
        # 어느 라운드를 처리할지는 rank 0 이 정해서 알린다. rank 마다 각자 폴링하면
        # READY 를 보는 시점이 갈려 한쪽만 학습에 들어가고, 그 상태로 all-reduce 를
        # 만나면 통신이 맞물리지 않아 그대로 멈춘다. 종료 신호도 같이 보낸다.
        decision = None
        if ddp.is_main():
            found = find_next_round(runs_exp, ckpt_exp)
            decision = {"stop": _stop, "round": None if found is None else
                        (found[0], str(found[1]))}
        decision = ddp.broadcast_object(decision)
        if decision["stop"]:
            break
        found = decision["round"]

        if found is None:
            now = time.time()
            if now - last_beat >= args.heartbeat_seconds:
                log(f"[idle] 대기 중 (처리한 라운드 {processed}개)")
                last_beat = now
            if args.once and processed:
                break
            time.sleep(args.poll_seconds)
            continue

        n, round_dir = found[0], Path(found[1])
        process_round(n, round_dir, ckpt_exp / round_dir.name, log, args, trainer)
        processed += 1
        last_beat = 0.0
        if args.once:
            break

    log(f"learner 종료 (처리한 라운드 {processed}개)")
    ddp.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
