#!/usr/bin/env python3
"""Roll out an RLDX-1 checkpoint in DexJoCo and record LeRobot v2.1 sessions.

Two processes, the split RLDX-1's own sim eval uses (docs/evaluation.md): the model
server holds the GPU in the training venv, this client steps MuJoCo in the sim venv,
and they talk ZeroMQ + msgpack. Neither venv has to satisfy the other's pins
(mujoco vs torch/flash-attn).

    # terminal 1 — model server (RLDX-1 venv, GPU)
    third_party/RLDX-1/.venv/bin/python third_party/RLDX-1/rldx/eval/run_rldx_server.py \
        --model-path checkpoints/dexjoco/dexjoco_hammer_nail_..._mlxp \
        --embodiment-tag GENERAL_EMBODIMENT --use-sim-policy-wrapper \
        --host 127.0.0.1 --port 20200

    # terminal 2 — rollout (sim venv; MUJOCO_GL=egl needs a GPU)
    MUJOCO_GL=egl /workspace/junmo_cho/dexjoco/venv/bin/python sim/dexjoco/rollout_dexjoco.py \
        --task hammer_nail --episodes 1000 --port 20200 \
        --output /workspace/junmo_cho/dexjoco/rollout/hammer_nail_bc30k

Output is the same schema the BC training set uses (convert_raw_to_rldx.py): 25-dim
rot6d state/action, camera_front/camera_wrist at 256x256, next.success/next.done/
next.truncated. So `rl/data.py`, `rl/extract_cogfeat.py` and `rl/offline_iql.py`
consume it unchanged, and rollouts can be mixed with the demos.

The protocol client is ~40 lines of zmq + msgpack mirroring MsgSerializer in
rldx/policy/server_client.py; no RLDX import is needed on this side.
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_raw_to_rldx import (  # noqa: E402
    DEFAULT_CAMERA_KEYS,
    TASK_CAMERAS,
    TASK_PROMPTS,
    Config as ConvertConfig,
    video_path,
    write_episode,
    write_meta,
)

# Must match dexjoco_panda_allegro_config.py; --verify-layout cross-checks the
# server's own copy (it travels inside the checkpoint's processor config).
GROUPS = [("eef_position", 0, 3), ("eef_rotation", 3, 9), ("hand_joints", 9, 25)]
LANGUAGE_KEY = "annotation.human.task_description"


# ── minimal PolicyClient (zmq + msgpack, no rldx import) ───────────────────────
class PolicyClient:
    """Same wire format as rldx/policy/server_client.py::PolicyClient."""

    def __init__(self, host: str = "127.0.0.1", port: int = 20200, timeout_ms: int = 300000):
        import zmq

        self._zmq = zmq
        self.ctx = zmq.Context()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self._connect()

    def _connect(self) -> None:
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(self._zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj

    @staticmethod
    def _decode(obj: Any) -> Any:
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    def call(self, endpoint: str, data: dict | None = None) -> Any:
        import msgpack

        req: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(msgpack.packb(req, default=self._encode))
        resp = msgpack.unpackb(self.sock.recv(), object_hook=self._decode)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"policy server error: {resp['error']}")
        return resp

    def ping(self) -> Any:
        return self.call("ping")

    def wait_ready(self, budget_s: float = 1800.0, probe_s: float = 20.0) -> None:
        """서버가 응답할 때까지 기다린다.

        정책 서버는 VLA 백본을 올리느라 2~3분이 걸리는데, 잡 여러 개가 동시에 뜨면
        같은 체크포인트를 NFS 에서 같이 읽느라 훨씬 오래 걸린다. REQ 소켓은 한 번
        타임아웃하면 상태가 망가지므로 매 시도마다 새로 연다.
        """
        import time

        t0 = time.time()
        n = 0
        while True:
            self.sock.close()
            self.timeout_ms = int(probe_s * 1000)
            self._connect()
            try:
                r = self.ping()
                if n:
                    print(f"[i] 정책 서버 준비됨 ({time.time() - t0:.0f}초, 시도 {n + 1}회)",
                          flush=True)
                self.sock.close()
                self.timeout_ms = 300000
                self._connect()
                return r
            except Exception:
                n += 1
                el = time.time() - t0
                if el > budget_s:
                    raise RuntimeError(
                        f"정책 서버가 {budget_s:.0f}초 안에 응답하지 않았다 "
                        f"({self.host}:{self.port}). 서버 로그를 볼 것")
                if n % 3 == 1:
                    print(f"[i] 정책 서버 대기 중… {el:.0f}초 경과", flush=True)

    def reset(self) -> None:
        # Clears per-episode policy state (RTC prefix / memory). Older servers may
        # not implement it; a failure here must not abort the rollout.
        try:
            self.call("reset", {"options": None})
        except RuntimeError as exc:
            print(f"[warn] policy reset failed, continuing: {exc}", flush=True)

    def get_modality_config(self) -> Any:
        return self.call("get_modality_config")

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict, dict]:
        resp = self.call("get_action", {"observation": observation, "options": options})
        return tuple(resp)  # msgpack decodes the tuple as a list


# ── rot6d <-> quaternion ───────────────────────────────────────────────────────
def rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """(6,) first two ROWS of R -> (4,) wxyz.

    Gram-Schmidt, matching EndEffectorPose._rot6d_to_matrix (pose.py:426). The
    policy's 6 numbers are a regression output and are NOT exactly orthonormal, so
    they must be re-orthogonalised rather than reshaped.
    """
    from scipy.spatial.transform import Rotation as R

    r = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    n1 = np.linalg.norm(r[0])
    row1 = r[0] / (n1 if n1 > 1e-8 else 1.0)
    row2 = r[1] - np.dot(row1, r[1]) * row1
    n2 = np.linalg.norm(row2)
    row2 = row2 / (n2 if n2 > 1e-8 else 1.0)
    row3 = np.cross(row1, row2)
    return R.from_matrix(np.vstack([row1, row2, row3])).as_quat(scalar_first=True)


def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    return R.from_quat(q[[1, 2, 3, 0]]).as_matrix()[:2, :].reshape(6)


def env_state_to_25(state: np.ndarray) -> np.ndarray:
    """dexjoco obs['state'] (38-dim for hammer_nail) -> 25-dim training layout.

    Only [tcp_pose(7), allegro_qpos(16)] is proprioception; the tail (object poses,
    table height) is privileged and dropped, exactly as in the BC dataset.
    """
    s = np.asarray(state, dtype=np.float64)
    return np.concatenate([s[0:3], quat_wxyz_to_rot6d(s[3:7]), s[7:23]]).astype(np.float32)


def action_25_to_env(action25: np.ndarray) -> np.ndarray:
    """policy action -> the (23,) array SingleArmPolicyWrapper expects."""
    a = np.asarray(action25, dtype=np.float64)
    return np.concatenate([a[0:3], rot6d_to_quat_wxyz(a[3:9]), a[9:25]])


@dataclass
class Config:
    task: str = "hammer_nail"
    output: Path = Path("/workspace/junmo_cho/dexjoco/rollout/hammer_nail_bc")
    """Destination LeRobot v2.1 session directory. Keep this off /rlwrld2 (full)."""
    episodes: int = 50
    host: str = "127.0.0.1"
    port: int = 20200
    replan: int = 16
    """Actions executed per policy query (= execution horizon). Must be <= the
    checkpoint's action_horizon (32) and must equal `replan_steps` in
    configs/exp/*.yaml — rl/data.py builds n-step returns over exactly this window."""
    rtc_delay: int = 0
    """Real-Time Chunking prefix length d (0 = off, server must run --rtc-inference-mode none).

    With d > 0 the client sends the d actions it has already committed as
    `options["action_prefix"]`, and executes indices [d, d+replan) of the returned
    chunk — the convention ExpoServer uses (`out[0, latency:latency+replan]`).
    Requires d + replan <= action_horizon, and the checkpoint to have been trained
    with rtc_training_max_delay >= d (this one: 8). Without RTC every replan is a
    hard switch to a freshly sampled chunk, and because flow matching draws a new
    sample each call the commanded pose jumps mid-swing.
    """
    max_episode_steps: int = 360
    """Cut here rather than at the env's own 1000-step cap: a failed hammer_nail episode
    otherwise runs ~5x longer than a successful one and dominates both wall-clock and
    the frame mix.

    360 comes from the success-length tail measured over 86 successful rollouts:
    p50 192, p90 313, p95 330, p99 365, max 390. Cutting at 360 mislabels 2.3% of
    successes as failures; 320 would mislabel 5.8%, 300 would mislabel 11.6%. That
    error is worse than it looks — a truncated success puts reward 0 on a trajectory
    that was about to succeed — while the frame-balance gain is small (failure frames
    54% at cut 400 vs 51% at 360 vs 49% at 320). So err long.

    Cuts are written as terminals (next.done=1, next.truncated=0); see the comment at
    the truncation site.
    """
    seed: int = 0
    """Env construction seed. Later episodes differ because the env randomises from
    the global RNG, which advances across resets."""
    fixed_scene: int = -1
    """>=0 이면 **모든 에피소드가 같은 장면**이다 (이 값을 매 reset 직전 시드로 쓴다).

    critic 이 액션을 학습하려면 같은 상태에서 다른 액션의 결과가 데이터에 있어야 하는데,
    장면이 매번 다르면 상태만으로 결과가 거의 설명되어 (홀드아웃 AUC 0.978) 액션 항이
    학습될 유인이 없다 — 실측에서 액션을 난수로 갈아치워도 Q 가 격차의 3% 밖에 안 움직였다.
    장면을 고정하면 s0 가 상수라 Q(s0,A) 가 **A 만의 함수일 수밖에 없다**: 상태 지름길이
    네트워크에 아예 존재하지 않는다. 결과 변동 전부가 액션에서 온다.

    성공률이 0% 나 100% 인 장면은 정보가 없으므로 50% 근처인 시드를 골라 쓸 것."""
    seed_per_episode: bool = True
    """Re-seed the global RNG from (seed, episode index) before every env.reset().

    The env draws its scene (table height, hammer xy/yaw, nail xy) from the **global**
    numpy RNG and ignores reset(seed=). Without this flag the scene depends on how many
    resets have happened in this process, so:

      - a --resume restart replays the scenes from the beginning. The 1000-episode
        collection restarted once at ep500, which is why episodes k and 500+k are the
        same scene (verified: first-frame pixel diff 0.6-0.8 for twins vs 3-7 for
        distinct scenes). Half of that dataset is a duplicate.
      - two policy variants cannot be compared on the same scenes, so a success-rate
        difference is confounded by which scenes each arm happened to draw.

    Seeding per episode index fixes both: episode i is the same scene in every run,
    which makes arms exactly paired (McNemar on the same scenes instead of a
    two-proportion test) and makes resume reproduce what it skipped."""
    randomize: bool = False
    """rand_full visual randomisation. Off = rand_obj, matching the BC training data."""
    randomize_dynamics: bool = False
    image_size: int = 256
    fps: int = 50
    """비디오 fps 이자 데이터셋 info.json 의 fps (timestamp 도 여기서 나온다).

    **50 이 실제 값이다**: env.step 한 번이 control_dt 0.02s (physics 0.002 x 10 substeps)
    이고, 롤아웃은 step 마다 액션 1개 + 프레임 1개를 남긴다 (완전히 1:1). 그러므로 데이터의
    실질 제어 레이트는 50Hz 다.

    이전 수집분은 30 으로 선언되어 있다 — 원본 dexjoco 레코더가 self.hz=30 을 쓰는 관례를
    그대로 따랐기 때문이다. 그 결과 (1) mp4 가 실제보다 1.67배 느리게 재생되고
    (2) timestamp 가 스텝당 0.0333s 로 적혀 실제 0.02s 와 다르다. 프레임 수와 액션 수는
    맞으므로 학습·평가에는 영향이 없었지만 (RLDX 도 rl/data.py 도 프레임 인덱스를 쓴다),
    사람이 초로 읽을 때 어긋난다. 새 수집분부터 바로잡는다.

    주의: BC 학습 데이터(hammer_nail_rand_obj)는 여전히 30 으로 선언돼 있다. 두 세션을
    한 --data 로 같이 로드하면 info.json 의 fps 가 다르지만, rl/data.py 는 sessions[0] 의
    fps 를 비디오 stride 계산에만 쓰므로 학습에는 영향이 없다."""
    crf: int = 20
    fast_render: bool = True
    """Render only the two cameras the policy uses. hammer_nail and click_mouse render
    four (front, ego_left, ego_right, wrist) every step; the other two are discarded."""
    resume: bool = True
    """Continue an interrupted session: rebuild the episode list from the parquet files
    already on disk and keep going. Long collections outlive their slurm allocation
    (this one lost two nodes mid-run), and meta/ is only rewritten periodically, so the
    on-disk episodes — not meta/episodes.jsonl — are the source of truth."""
    log_every: int = 10
    verify_layout: bool = True
    """Cross-check the server's modality config against GROUPS / camera keys."""
    keep_failures: bool = True
    """Record failed and truncated episodes too — offline RL needs the negatives."""


def build_env(cfg: Config):
    from dexjoco.tasks import CONFIG_MAPPING

    env = CONFIG_MAPPING[cfg.task]().get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=cfg.randomize,
        seed=cfg.seed,
        randomize_dynamics=cfg.randomize_dynamics,
    )
    raw = env.unwrapped

    # The `hz` constructor kwarg is ignored — panda_hammer_nail_env.__init__ does a
    # literal `self.hz = 30` — and step() sleeps to that rate, which would pin the
    # rollout to real time. Patch the attribute after construction.
    raw.hz = 100000.0

    if cfg.fast_render:
        _install_fast_render(raw, cfg)
    return env


def _install_fast_render(raw, cfg: Config) -> None:
    """Replace discarded camera renders with a cached black frame.

    `_compute_observation` unpacks `render()` positionally, so the returned list must
    keep its length and order; only the GPU work for unused views is skipped. Envs
    that already render exactly what they need (water_plant returns [wrist, front])
    have no tuple `camera_id` and are left alone.
    """
    camera_id = getattr(raw, "camera_id", None)
    if not isinstance(camera_id, tuple) or len(camera_id) != 4:
        return
    # camera_id is (front, ego_left, ego_right, wrist); _compute_observation maps it to
    # (front|random_camera, ego_left, ego_right, wrist).
    slot_of = {"front": 0, "random_camera": 0, "ego_left": 1, "ego_right": 2, "wrist": 3}
    keep = {slot_of[name] for name in TASK_CAMERAS[cfg.task]}
    if cfg.randomize:
        keep.add(0)
    blank = None

    def fast_render():
        nonlocal blank
        frames = []
        for i, cam in enumerate(camera_id):
            if i in keep:
                frames.append(raw._viewer.render(render_mode="rgb_array", camera_id=cam))
            else:
                if blank is None:
                    blank = np.zeros_like(frames[0]) if frames else None
                frames.append(blank)
        # A discarded slot before the first kept one has no shape to copy yet.
        if any(f is None for f in frames):
            shape = next(f for f in frames if f is not None).shape
            frames = [np.zeros(shape, np.uint8) if f is None else f for f in frames]
        return frames

    raw.render = fast_render
    print(f"[i] fast_render: rendering slots {sorted(keep)} of 4", flush=True)


def obs_to_request(obs: dict, cam_keys: dict[str, str], prompt: str, size: int) -> tuple[dict, dict]:
    """env obs -> (server request, {modality key: resized uint8 frame})."""
    import cv2

    frames: dict[str, np.ndarray] = {}
    for out_key, raw_name in cam_keys.items():
        img = obs[raw_name]
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        frames[out_key] = np.ascontiguousarray(img, dtype=np.uint8)

    state25 = env_state_to_25(obs["state"])
    request: dict[str, Any] = {LANGUAGE_KEY: [prompt]}
    for key, frame in frames.items():
        request[f"video.{key}"] = frame[None, None]  # (B=1, T=1, H, W, C)
    for name, lo, hi in GROUPS:
        request[f"state.{name}"] = state25[lo:hi][None, None].astype(np.float32)  # (1,1,D)
    return request, frames


def chunk_from_response(action: dict[str, np.ndarray]) -> np.ndarray:
    """server action dict -> (horizon, 25) in our concat order."""
    parts = []
    for name, lo, hi in GROUPS:
        a = np.asarray(action[f"action.{name}"], dtype=np.float32)
        if a.ndim == 3:  # (B, T, D)
            a = a[0]
        assert a.shape[-1] == hi - lo, f"action.{name}: {a.shape[-1]} != {hi - lo}"
        parts.append(a)
    return np.concatenate(parts, axis=-1)


def verify_layout(client: PolicyClient, cam_keys: dict[str, str], replan: int) -> None:
    cfgs = client.get_modality_config()

    def keys_of(m):
        c = cfgs[m]
        c = c.get("as_json", c) if isinstance(c, dict) else c
        return list(c["modality_keys"]), len(c["delta_indices"])

    s_keys, _ = keys_of("state")
    a_keys, horizon = keys_of("action")
    v_keys, _ = keys_of("video")
    want = [g[0] for g in GROUPS]
    assert s_keys == want, f"server state keys {s_keys} != {want}"
    assert a_keys == want, f"server action keys {a_keys} != {want}"
    assert set(v_keys) == set(cam_keys), f"server video keys {v_keys} != {list(cam_keys)}"
    assert replan <= horizon, f"--replan {replan} > checkpoint action_horizon {horizon}"
    verify_layout.horizon = horizon
    print(f"[i] layout ok — groups {want}, cameras {v_keys}, action_horizon {horizon}", flush=True)


def scan_existing(cfg: Config, cam_keys: list[str], prompt: str) -> tuple[list[dict], int, int]:
    """Rebuild (episodes_meta, global_index, n_success) from episodes already on disk.

    Reads the flags back out of each parquet rather than trusting meta/episodes.jsonl,
    which is only rewritten every --log-every episodes and so lags a crash. An episode
    whose videos are missing (killed mid-write) is dropped so the run rewrites it.
    """
    import pandas as pd

    data_dir = cfg.output / "data/chunk-000"
    if not data_dir.is_dir():
        return [], 0, 0

    prior = {}
    meta_path = cfg.output / "meta/episodes.jsonl"
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                prior[row["episode_index"]] = row

    metas: list[dict] = []
    global_index = n_success = 0
    for path in sorted(data_dir.glob("episode_*.parquet")):
        ep = int(path.stem.split("_")[1])
        if any(not video_path(cfg, ep, k).exists() for k in cam_keys):
            print(f"[resume] episode {ep}: videos missing, will be rewritten", flush=True)
            break
        df = pd.read_parquet(path, columns=["next.success", "next.truncated"])
        length = len(df)
        success = bool(df["next.success"].values[-1])
        row = prior.get(ep) or {}
        # `next.truncated` is deliberately always 0 on disk (cuts are written as
        # terminals so rl/data.py's check 8 passes), so it cannot be read back to
        # tell a cut from a real termination. Infer it from the length instead: the
        # env only terminates below the cap on success (hammer_nail terminates at
        # `env_step >= 1000 or success`), so a non-success episode that reached the
        # cap was cut. This field is bookkeeping only — rl/data.py reads the parquet
        # columns, not meta.
        metas.append(
            {
                "episode_index": ep,
                "tasks": [prompt],
                "length": length,
                "success": success,
                "truncated": (not success) and length >= cfg.max_episode_steps,
                "steps": length,
                "final_nail_depth": row.get("final_nail_depth"),
            }
        )
        global_index += length
        n_success += int(success)
    if metas:
        print(
            f"[resume] {len(metas)} episodes on disk ({n_success} success), "
            f"continuing at index {metas[-1]['episode_index'] + 1}",
            flush=True,
        )
    return metas, global_index, n_success


def main(cfg: Config) -> None:
    prompt = TASK_PROMPTS[cfg.task]
    raw_cams = TASK_CAMERAS[cfg.task]
    if cfg.randomize and raw_cams[0] == "front":
        raw_cams = ["random_camera", raw_cams[1]]  # env renames it under rand_full
    cam_keys = dict(zip(DEFAULT_CAMERA_KEYS, raw_cams))

    conv = ConvertConfig(
        input=cfg.output, output=cfg.output, task=cfg.task,
        cameras=list(raw_cams), image_size=cfg.image_size, fps=cfg.fps, crf=cfg.crf,
    )
    cfg.output.mkdir(parents=True, exist_ok=True)

    client = PolicyClient(cfg.host, cfg.port)
    client.wait_ready()
    if cfg.verify_layout:
        verify_layout(client, cam_keys, cfg.replan + cfg.rtc_delay)
    print(
        f"[i] replan={cfg.replan} rtc_delay={cfg.rtc_delay} "
        f"max_episode_steps={cfg.max_episode_steps}",
        flush=True,
    )

    if cfg.resume:
        episodes_meta, global_index, n_success = scan_existing(cfg, list(cam_keys), prompt)
    else:
        episodes_meta, global_index, n_success = [], 0, 0
    done_before = len(episodes_meta)
    remaining = cfg.episodes - done_before
    if remaining <= 0:
        print(f"[i] already have {done_before} >= {cfg.episodes} episodes, nothing to do", flush=True)
        return

    env = build_env(cfg)
    t_start = time.time()

    for _ in range(remaining):
        ep_idx = len(episodes_meta)
        if cfg.fixed_scene >= 0 or cfg.seed_per_episode:
            # env.reset() ignores its seed= argument and draws from the global RNG
            # (panda_hammer_nail_env.py:396). Seed it here so scene == f(episode index),
            # or a constant when --fixed-scene is given.
            sd = (cfg.fixed_scene if cfg.fixed_scene >= 0
                  else (cfg.seed * 1_000_003 + ep_idx)) % (2 ** 31 - 1)
            random.seed(sd)
            np.random.seed(sd)
        obs, _ = env.reset()
        client.reset()

        states, actions, actions_raw, frames_buf = [], [], [], {k: [] for k in cam_keys}
        successes, dones, truncs, infos = [], [], [], []
        chunk: np.ndarray | None = None
        chunk_pos = 0
        prefix: np.ndarray | None = None  # the d actions already committed (RTC only)
        success = done = trunc = False
        first_call = True

        for step in range(cfg.max_episode_steps):
            request, frames = obs_to_request(obs, cam_keys, prompt, cfg.image_size)
            if chunk is None or chunk_pos >= cfg.replan:
                options: dict[str, Any] = {}
                if first_call:
                    # Clears the session's RTC chunk cache and memory tokens. Without
                    # it the previous episode's state leaks into this one.
                    options["reset_memory"] = [True]
                if cfg.rtc_delay > 0 and prefix is not None:
                    options["action_prefix"] = prefix.astype(np.float32)[None]  # (1,d,D)
                action_dict, _ = client.get_action(request, options or None)
                full = chunk_from_response(action_dict)
                # RTC: the first d entries are the frozen prefix we supplied, so the
                # actions still to execute start at d.
                start = cfg.rtc_delay if (cfg.rtc_delay > 0 and prefix is not None) else 0
                chunk = full[start:]
                chunk_pos = 0
                first_call = False
            a25 = chunk[chunk_pos]
            chunk_pos += 1

            # `action` is the action the environment actually executed: the policy's
            # 6 rotation numbers are a regression output that is not on SO(3)
            # (measured on a smoke rollout: |row1.row2| up to 0.28, row norms
            # 0.90-1.01), and action_25_to_env re-orthogonalises them. Two rot6d
            # values that project to the same rotation are the *same* MDP action, so
            # recording the pre-projection numbers would give the critic different
            # `a` for identical outcomes — and would put the rollouts off the
            # manifold the BC demos live on (their rot6d is exactly orthonormal).
            # The raw policy output is kept as a diagnostic column.
            env_action = action_25_to_env(a25)
            executed25 = np.concatenate(
                [env_action[0:3], quat_wxyz_to_rot6d(env_action[3:7]), env_action[7:23]]
            ).astype(np.float32)

            # record the (s, a) pair BEFORE stepping — same convention as the demos
            states.append(env_state_to_25(obs["state"]))
            actions.append(executed25)
            actions_raw.append(a25.astype(np.float32))
            if cfg.rtc_delay > 0:
                # Keep the last d executed actions as the next request's prefix.
                prefix = np.stack(actions_raw[-cfg.rtc_delay :])
            for k, f in frames.items():
                frames_buf[k].append(f)

            obs, _reward, terminated, truncated, info = env.step(env_action)
            success = bool(info.get("succeed", False))
            done = bool(terminated)
            infos.append({k: float(info[k]) for k in ("nail_depth",) if k in info})
            successes.append(success)
            dones.append(done or (step == cfg.max_episode_steps - 1))
            truncs.append(False)
            if done:
                break
        else:
            trunc = True

        length = len(actions)
        if trunc:
            # Ran out of step budget. `next.truncated` stays 0 and `next.done` goes to
            # 1, i.e. the cut is recorded as a terminal with value 0.
            #
            # Why not truncated=1, which is what actually happened: the whole rd-rl
            # critic stack assumes mask = 1 - done (rl/data.py:19) and rl/data.py's
            # check 8 fails outright if any truncated flag is set. Marking a cut as
            # terminal makes the Bellman target r + gamma^R * mask * V(s') drop the
            # bootstrap there — correct for an episode that was going nowhere, wrong
            # for one that would have succeeded later, which is why
            # --max-episode-steps is set above the success-length tail (360: 2.3% of
            # observed successes ran longer; 320 would have cut 5.8%).
            #
            # The env makes the same approximation at its own 1000-step cap — it
            # returns terminated=True, not truncated (panda_hammer_nail_env.py:571).
            dones[-1] = True

        if not (success or cfg.keep_failures):
            print("[ep] dropped (failure, --keep-failures off)", flush=True)
            continue

        ep_idx = len(episodes_meta)
        episodes_meta.append(
            write_episode(
                conv, ep_idx,
                np.stack(states), np.stack(actions),
                {k: np.stack(v) for k, v in frames_buf.items()},
                list(cam_keys), global_index,
                success=np.array(successes, dtype=bool),
                done=np.array(dones, dtype=bool),
                truncated=np.array(truncs, dtype=bool),
                extra_meta={
                    "success": success,
                    "truncated": trunc,
                    "steps": length,
                    "final_nail_depth": infos[-1].get("nail_depth") if infos else None,
                },
                extra_columns={"action_raw": np.stack(actions_raw)},
            )
        )
        global_index += length
        n_success += int(success)

        n_done = len(episodes_meta)
        write_meta(conv, episodes_meta, list(cam_keys), prompt=prompt)  # crash-safe
        if n_done % cfg.log_every == 0 or n_done == done_before + 1:
            elapsed = time.time() - t_start
            print(
                f"[{n_done}/{cfg.episodes}] success {n_success}/{n_done} "
                f"({100 * n_success / n_done:.1f}%)  last: {length} steps "
                f"{'SUCCESS' if success else ('TRUNC' if trunc else 'FAIL')}  "
                f"{elapsed / max(n_done - done_before, 1):.1f}s/ep  "
                f"eta {(cfg.episodes - n_done) * elapsed / max(n_done - done_before, 1) / 60:.0f}m",
                flush=True,
            )

    write_meta(conv, episodes_meta, list(cam_keys), prompt=prompt)
    env.close()

    n = len(episodes_meta)
    lengths = np.array([m["length"] for m in episodes_meta]) if n else np.array([0])
    ok = np.array([bool(m.get("success")) for m in episodes_meta]) if n else np.zeros(1, bool)
    # 성공 에피소드의 길이를 따로 남긴다. 성공만으로 학습한 critic 은 참값이 γ^(T-t) 라
    # "빨리 끝나는 것" 을 높게 보므로, test-time 선택/guidance 가 실제로 완주를 앞당기는지가
    # 성공률만큼 중요한 지표다 (실측: BC 251.7 -> selection 225.2 -> guidance 167.4 프레임).
    # 최소값은 특히 의미가 크다 — BC 가 200회 시도해서 못 낸 속도를 냈는지 보여준다.
    sl = lengths[ok] if ok.any() else np.array([0])
    summary = {
        "task": cfg.task,
        "episodes": n,
        "success": n_success,
        "success_rate": n_success / n if n else 0.0,
        "frames": global_index,
        "mean_length": float(lengths.mean()),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "success_mean_length": float(sl.mean()),
        "success_median_length": float(np.median(sl)),
        "success_min_length": int(sl.min()),
        "success_max_length": int(sl.max()),
        "replan": cfg.replan,
        "rtc_delay": cfg.rtc_delay,
        "max_episode_steps": cfg.max_episode_steps,
        "randomize": cfg.randomize,
        "seed": cfg.seed,
        "fixed_scene": cfg.fixed_scene,
        "seed_per_episode": cfg.seed_per_episode,
        # 어떤 critic / 어떤 서버 설정으로 얻은 결과인지 남긴다. 없으면 순정 BC 다.
        "serve": os.environ.get("RD_SERVE_INFO", ""),
        "wall_seconds": time.time() - t_start,
    }
    (cfg.output / "rollout_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nSUCCESS RATE {n_success}/{n} = {100 * n_success / max(n, 1):.1f}%", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
