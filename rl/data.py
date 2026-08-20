#!/usr/bin/env python3
"""LeRobot 세션 → RL transition 필드.

EXPO-FT ``expo_ft/data/replay_buffer.py`` 의 ``sample_jax`` 의미를 그대로 옮긴다:

    reward = Σ_{i<replan} r[t+i] · γ^i · (누적 mask)     n-step 리턴
    mask   = min(m[t..t+replan-1])                       terminal 이후 0
    valid  = replan-2 까지의 mask 곱                      critic loss 가중치
    next   = t + replan

에피소드 경계를 넘는 transition 을 **막지 않는다** — EXPO-FT 도 평평한 버퍼에서 아무
인덱스나 뽑고, 경계에서 done=1 이라 mask 가 0 이 되어 next_q 가 무의미해지는 방식으로
처리한다. 인덱스를 걸러내면 원본과 분포가 달라지므로 같은 방식을 쓴다.

우리 데이터(rrc 가 쓰는 LeRobot) → EXPO-FT 필드:

    reward_t   = 1.0 if next.success[t] else 0.0    성공 에피소드의 마지막 1프레임
    done_t     = next.done[t]
    mask_t     = 1 - done_t                          process_droid_dataset 과 동일
    is_success = 에피소드에 next.success 가 있으면 그 에피소드 전체 프레임에 True

실행: pixi `rldx` 환경 (pandas 가 거기 있다)
    cd third_party/RLDX-1 && PYTHONPATH="$PWD" pixi run -e rldx python ../../rl/data.py <세션 부모>
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RL_COLUMNS = ["next.success", "next.done", "next.truncated", "episode_index", "frame_index"]
DEFAULT_ACTION_KEY = "action"


@dataclass
class Modality:
    """modality.json 에서 유도한 embodiment 사양.

    컬럼 이름·슬라이스·카메라 목록·차원을 하드코딩하지 않는다. embodiment 마다 다르다:
      openarm_lefthand  state 0:28 (28), 카메라 2
      rby1m_rh56f1      state 4:38 (34) — 0:4(wheel) 를 **쓰지 않는다**, 카메라 3
      rby1m_wuji2       state 0:66 (66), 카메라 2
    """

    state: list[tuple[str, str, int, int]]   # (그룹명, original_key, start, end) — canonical 순
    action: list[tuple[str, str, int, int]]
    video: list[tuple[str, str]]             # (카메라명, original_key) — canonical 순
    task_key: str = "annotation.human.task_description"
    embodiment_tag: str = ""                 # 등록 config 가 정한다 (손으로 적지 않는다)
    layout_source: str = "modality.json(start 순)"

    @property
    def state_dim(self) -> int:
        return sum(e - s for _, _, s, e in self.state)

    @property
    def action_dim(self) -> int:
        return sum(e - s for _, _, s, e in self.action)

    @property
    def n_cams(self) -> int:
        return len(self.video)

    def offsets(self, which: str) -> list[tuple[str, int, int]]:
        """우리가 만든 concat 배열 안에서 각 그룹의 위치. (원본 start/end 가 아니다)"""
        out, cum = [], 0
        for name, _, s, e in getattr(self, which):
            out.append((name, cum, cum + (e - s)))
            cum += e - s
        return out

    def columns(self) -> list[str]:
        keys = {k for _, k, _, _ in self.state} | {k for _, k, _, _ in self.action}
        return sorted(keys) + RL_COLUMNS


def rldx_layout(rldx_root: Path, rel_config: str) -> tuple[str, dict[str, list[str]]]:
    """RLDX 등록 config 를 로드해 (embodiment_tag, {modality: modality_keys}) 를 얻는다.

    **concat 순서의 정본은 이 파일이다.** modality.json 의 start/end 는 raw 컬럼에서
    잘라낼 때만 쓴다 (두 순서가 같다는 보장이 없다 — openarm 은 3·4번째가 뒤바뀐다).
    embodiment_tag 도 config 파일이 정하므로 등록 결과에서 읽는다 (openarm_inspire →
    GENERAL_EMBODIMENT, rby1_f1 → NEW_EMBODIMENT). 태그별로 하나만 등록할 수 있다.
    """
    import sys as _sys
    _sys.path.insert(0, str(rldx_root))
    from rldx.configs.data.embodiment_configs import MODALITY_CONFIGS
    from rldx.experiment.utils import load_modality_config

    path = rldx_root / rel_config
    before = set(MODALITY_CONFIGS)
    load_modality_config(str(path))
    added = set(MODALITY_CONFIGS) - before
    if len(added) != 1:
        raise ValueError(f"{rel_config}: 등록된 embodiment 태그를 특정할 수 없다 (added={added}). "
                         "같은 프로세스에서 다른 config 를 이미 로드했을 수 있다")
    tag = added.pop()
    cfg = MODALITY_CONFIGS[tag]
    return tag, {mod: list(cfg[mod].modality_keys) for mod in ("video", "state", "action")
                 if mod in cfg}


def parse_modality(m: dict, order: dict[str, list[str]] | None = None,
                   tag: str = "", source: str = "modality.json(start 순)") -> Modality:
    def groups(mod: str, default_key: str) -> list[tuple[str, str, int, int]]:
        out = []
        for name, v in m.get(mod, {}).items():
            out.append((name, v.get("original_key", default_key), int(v["start"]), int(v["end"])))
        keys = (order or {}).get(mod)
        if keys is None:
            return sorted(out, key=lambda g: g[2])          # 정본이 없으면 start 순 (경고용 기본)
        have = {g[0] for g in out}
        if have != set(keys):
            raise ValueError(f"{mod}: modality.json 과 등록 config 의 그룹이 다르다\n"
                             f"  modality.json: {sorted(have)}\n  등록 config : {keys}")
        by = {g[0]: g for g in out}
        return [by[k] for k in keys]                        # ★ canonical 순

    vid = [(name, v.get("original_key", f"observation.images.{name}"))
           for name, v in m.get("video", {}).items()]
    vkeys = (order or {}).get("video")
    if vkeys is not None:
        have = {n for n, _ in vid}
        if have != set(vkeys):
            raise ValueError(f"video: modality.json {sorted(have)} != 등록 config {vkeys}")
        byv = {n: (n, k) for n, k in vid}
        vid = [byv[k] for k in vkeys]
    video = vid
    task = next((f"annotation.{k}" for k in m.get("annotation", {})),
                "annotation.human.task_description")
    return Modality(state=groups("state", "observation.state"),
                    action=groups("action", DEFAULT_ACTION_KEY),
                    video=video, task_key=task, embodiment_tag=tag, layout_source=source)


def load_modality(path: Path) -> Modality:
    """modality.json (또는 그것을 담은 디렉토리) 을 읽는다."""
    path = Path(path)
    if path.is_dir() and not (path / "meta").is_dir():
        path = path / "modality.json"
    return parse_modality(json.loads(path.read_text()))


def modality_from_sessions(sessions: list[Path], order: dict[str, list[str]] | None = None,
                           tag: str = "", source: str = "modality.json(start 순)") -> Modality:
    """embodiment 는 데이터에서 온다 — convert_data.py --modality 가 각 데이터셋의
    meta/modality.json 에 심어둔 것을 읽는다. 세션들이 서로 다르면 (다른 로봇을 한 버퍼에
    섞는 것) 즉시 실패한다."""
    metas, robots = [], {}
    for s in sessions:
        mp = s / "meta" / "modality.json"
        if not mp.is_file():
            raise ValueError(
                f"{s.name}: meta/modality.json 이 없다. "
                "convert_data.py --modality <파일> 로 변환해야 한다")
        metas.append((s.name, json.loads(mp.read_text())))
        robots[s.name] = json.loads((s / "meta" / "info.json").read_text()).get("robot_type")
    first_name, first = metas[0]
    for name, m in metas[1:]:
        if m != first:
            raise ValueError(f"세션들의 modality.json 이 다르다: {first_name} vs {name} "
                             "(embodiment 를 한 버퍼에 섞을 수 없다)")
    if len(set(robots.values())) > 1:
        raise ValueError(f"세션들의 robot_type 이 다르다: {robots}")
    return parse_modality(first, order, tag, source)


def checkpoint_layout(ckpt: Path, tag: str) -> dict[str, list[str]] | None:
    """체크포인트가 실제 학습에 쓴 modality_keys. 최종 안전장치.

    processor/processor_config.json 에 embodiment 별로 저장돼 있다. 등록 config 와 다르면
    (학습 후 config 파일이 바뀌었다면) 모델 공간 레이아웃이 어긋나므로 실패시킨다.
    """
    p = Path(ckpt) / "processor" / "processor_config.json"
    if not p.is_file():
        return None
    mc = json.loads(p.read_text()).get("processor_kwargs", {}).get("modality_configs", {})
    if tag not in mc:
        raise ValueError(f"체크포인트 processor_config 에 embodiment '{tag}' 가 없다 "
                         f"(있는 것 예: {sorted(mc)[:5]})")
    return {mod: list(v["modality_keys"]) for mod, v in mc[tag].items()
            if mod in ("video", "state", "action")}


def resolve_modality(root: Path, override: Path | None = None,
                     rldx_root: Path | None = None, rldx_config: str | None = None,
                     base_policy: Path | None = None) -> tuple[Modality, str]:
    """그룹 경계는 데이터(modality.json), **순서와 태그는 RLDX 등록 config** 에서.

    rldx_config 는 필수다. 없으면 start 순으로 조용히 잘못될 수 있고 (openarm 은 실제로
    3·4번째가 뒤바뀐다) 그 결과가 explore_groups 가 다른 관절을 가리키는 것이다.
    """
    if not rldx_config:
        raise SystemExit(
            "rldx_config 가 필요하다 (예: rldx/configs/data/openarm_inspire_config.py).\n"
            "  concat 순서와 embodiment_tag 의 정본이다. 없으면 modality.json 의 start 순을\n"
            "  쓰게 되는데 그게 모델과 다를 수 있다 (openarm: right_arm 과 left_hand 가 뒤바뀜).")
    tag, order = rldx_layout(Path(rldx_root), rldx_config)
    src = f"meta/modality.json + 순서/태그는 {rldx_config} ({tag})"

    if base_policy is not None:
        ck = checkpoint_layout(Path(base_policy), tag)
        if ck is None:
            print(f"  [경고] {base_policy} 에 processor/processor_config.json 이 없어 "
                  "체크포인트 교차검증을 건너뜀")
        else:
            for mod, keys in order.items():
                if mod in ck and ck[mod] != keys:
                    raise SystemExit(
                        f"{mod} 순서가 등록 config 와 체크포인트에서 다르다 — 모델 공간이 어긋난다\n"
                        f"  등록 config: {keys}\n  체크포인트 : {ck[mod]}")
            src += " + 체크포인트 교차검증 OK"

    if override is not None:
        return parse_modality(json.loads(_modality_path(override).read_text()), order,
                              tag, src), f"덮어쓰기 {override}"
    return modality_from_sessions(find_sessions(root), order, tag, src), src


def _modality_path(path: Path) -> Path:
    path = Path(path)
    return path / "modality.json" if path.is_dir() and not (path / "meta").is_dir() else path


@dataclass
class Flat:
    """세션들을 시간 순으로 이어붙인 평평한 배열 (EXPO-FT 리플레이 버퍼와 같은 모양)."""

    reward: np.ndarray      # (T,) float32
    mask: np.ndarray        # (T,) float32 — 1 - done
    done: np.ndarray        # (T,) bool
    truncated: np.ndarray   # (T,) bool
    is_success: np.ndarray  # (T,) bool — 에피소드 단위 성공 플래그를 프레임에 펼친 것
    action: np.ndarray      # (T, A) float32 — 실제 실행된 액션 (LeRobot 원본 공간)
    state: np.ndarray       # (T, S) float32 — observation.joint_position
    ep_end: np.ndarray      # (T,) int64 — 그 프레임이 속한 에피소드의 마지막 인덱스
    episode: np.ndarray     # (T,) int32 — 전역 에피소드 번호
    frame: np.ndarray       # (T,) int32 — 에피소드 내 프레임 번호
    session: np.ndarray     # (T,) int16 — 세션 번호
    sessions: list[str] = field(default_factory=list)
    ep_success: list[bool] = field(default_factory=list)
    ep_length: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.reward)


def is_lerobot(path: Path) -> bool:
    return all((path / d).is_dir() for d in ("data", "meta", "videos")) and (
        path / "meta" / "info.json"
    ).is_file()


def find_sessions(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if is_lerobot(root):
        return [root]
    return [p for p in sorted(root.iterdir()) if p.is_dir() and is_lerobot(p)]


def _gather(df: pd.DataFrame, groups: list[tuple[str, str, int, int]]) -> np.ndarray:
    """선언된 그룹들만 순서대로 이어붙인다 → (T, Σ(e-s)).

    모델이 보는 차원과 같아야 하므로 원본 컬럼 전체가 아니라 그룹 합집합을 쓴다
    (rby1m_rh56f1 은 0:4 를 쓰지 않는다)."""
    cols = {}
    parts = []
    for _, key, s, e in groups:
        if key not in cols:
            cols[key] = np.stack(df[key].to_numpy()).astype(np.float32)
        parts.append(cols[key][:, s:e])
    return np.concatenate(parts, axis=1)


def load_session(path: Path, mod: Modality) -> dict:
    """한 세션의 에피소드들을 시간 순으로 읽어 프레임 배열을 만든다."""
    files = sorted(path.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        raise ValueError(f"parquet 없음: {path}")
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=mod.columns())
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    return {
        "action": _gather(df, mod.action),
        "state": _gather(df, mod.state),
        "success": df["next.success"].to_numpy(dtype=bool).reshape(-1),
        "done": df["next.done"].to_numpy(dtype=bool).reshape(-1),
        "truncated": df["next.truncated"].to_numpy(dtype=bool).reshape(-1),
        "episode": df["episode_index"].to_numpy(dtype=np.int64).reshape(-1),
        "frame": df["frame_index"].to_numpy(dtype=np.int64).reshape(-1),
        "n_episodes": len(files),
    }


def build_flat(sessions: list[Path], mod: Modality) -> Flat:
    R, M, D, TR, S, EP, FR, SE, AC, ST = [], [], [], [], [], [], [], [], [], []
    names, ep_success, ep_length = [], [], []
    ep_offset = 0
    for si, path in enumerate(sessions):
        d = load_session(path, mod)
        names.append(path.name)
        succ, done = d["success"], d["done"]
        local_ep = d["episode"]
        # 에피소드 단위 성공 플래그를 프레임으로 펼친다 (actor 배치용)
        flag = np.zeros(len(succ), dtype=bool)
        for e in np.unique(local_ep):
            sel = local_ep == e
            ok = bool(succ[sel].any())
            flag[sel] = ok
            ep_success.append(ok)
            ep_length.append(int(sel.sum()))
        AC.append(d["action"])
        ST.append(d["state"])
        R.append(succ.astype(np.float32))
        M.append(1.0 - done.astype(np.float32))
        D.append(done)
        TR.append(d["truncated"])
        S.append(flag)
        EP.append(local_ep.astype(np.int32) + ep_offset)
        FR.append(d["frame"].astype(np.int32))
        SE.append(np.full(len(succ), si, dtype=np.int16))
        ep_offset += int(local_ep.max()) + 1
    done_all = np.concatenate(D)
    # ep_end[t] = t 가 속한 에피소드의 마지막 인덱스. done 이 에피소드 마지막에만 1 이므로
    # 뒤에서부터 누적하면 얻어진다 (검사 3 이 이 가정을 확인한다).
    ends = np.flatnonzero(done_all)
    ep_end = np.empty(len(done_all), dtype=np.int64)
    prev = 0
    for e in ends:
        ep_end[prev:e + 1] = e
        prev = e + 1
    if prev < len(done_all):            # done 으로 끝나지 않은 꼬리 (있으면 안 되지만 방어)
        ep_end[prev:] = len(done_all) - 1
    return Flat(
        action=np.concatenate(AC), state=np.concatenate(ST), ep_end=ep_end,
        reward=np.concatenate(R), mask=np.concatenate(M), done=done_all,
        truncated=np.concatenate(TR), is_success=np.concatenate(S),
        episode=np.concatenate(EP), frame=np.concatenate(FR), session=np.concatenate(SE),
        sessions=names, ep_success=ep_success, ep_length=ep_length,
    )


def nstep(flat: Flat, idx: np.ndarray, replan_steps: int = 8, discount: float = 0.99) -> dict:
    """EXPO-FT sample_jax 의 n-step 집계를 그대로 옮긴 것."""
    idx = np.asarray(idx, dtype=np.int64)
    reward = flat.reward[idx].copy()
    mask = flat.mask[idx].copy()
    done = flat.done[idx].copy()
    valid = np.ones(len(idx), dtype=np.float32)
    prev_mask = mask.copy()
    for i in range(1, replan_steps):
        j = idx + i
        reward += flat.reward[j] * (discount ** i) * prev_mask
        valid = prev_mask
        mask = np.minimum(mask, flat.mask[j])
        done = np.logical_or(done, flat.done[j])
        prev_mask = mask.copy()
    return {"reward": reward, "mask": mask, "valid": valid, "done": done,
            "next_idx": idx + replan_steps}


def action_chunk(flat: Flat, idx: np.ndarray, horizon: int = 16) -> np.ndarray:
    """(B, horizon, A) — 실제로 실행된 미래 액션 청크.

    EXPO-FT ``PiReplayBuffer.insert`` 의 back-fill 과 같은 의미:
      - chunk[k] = action[t+k]
      - 에피소드 끝을 넘는 위치는 **그 에피소드의 마지막 액션을 반복** (경계를 넘지 않는다)
    인덱스를 ep_end 로 클램프하면 그 반복 패딩이 그대로 재현된다.
    """
    idx = np.asarray(idx, dtype=np.int64)
    off = idx[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
    off = np.minimum(off, flat.ep_end[idx][:, None])
    return flat.action[off]


def make_batch(flat: Flat, imgs, idx: np.ndarray, mod: Modality, replan_steps: int = 8,
               action_horizon: int = 16, discount: float = 0.99, task: str = "",
               latency: int = 0) -> dict:
    """학습 배치 하나. EXPO-FT prepare_critic_batch 와 같은 분해.

    critic 계열은 원본 uint8 이미지를 채널로 이어붙여 쓰고 (인코더를 따로 학습하므로
    RLDX 프로세서를 통과시키지 않는다 — augmentation 이중 적용을 피한다),
    VLA 계열은 RLDX 프로세서에 넣을 nested dict 을 그대로 준다.

    latency (RTC 지연) 만큼 critic 액션 창을 뒤로 민다. 추론이 경계보다 latency 스텝 먼저
    돌고 앞 latency 개가 prefix 로 고정되므로, 실제로 실행되는 것은
    chunk[latency : latency+replan_steps] 이다. actor BC 대상은 청크 전체.
    """
    if action_horizon < latency + replan_steps:
        raise ValueError(f"action_horizon({action_horizon}) < latency({latency}) + "
                         f"replan_steps({replan_steps})")
    idx = np.asarray(idx, dtype=np.int64)
    n = nstep(flat, idx, replan_steps, discount)
    nxt = n["next_idx"]
    ch = action_chunk(flat, idx, action_horizon)          # (B, H, A)
    A = ch.shape[-1]

    def cat_cams(i):
        # (B, n_cams, H, W, 3) → (B, H, W, 3*n_cams)  EXPO-FT 는 axis=-1 로 이어붙인다
        x = np.asarray(imgs[i])
        return np.concatenate([x[:, c] for c in range(x.shape[1])], axis=-1)

    def vla_obs(i):
        return {
            "video": {name: np.asarray(imgs[i])[:, c][:, None]
                      for c, (name, _) in enumerate(mod.video)},
            "state": {name: flat.state[i][:, None, s:e] for name, s, e in mod.offsets("state")},
            "language": {mod.task_key: [[task]] * len(i)},
        }

    return {
        # --- critic / residual / 인코더 ---
        "obs": cat_cams(idx), "next_obs": cat_cams(nxt),
        "state": flat.state[idx], "next_state": flat.state[nxt],
        "action": ch[:, latency:latency + replan_steps].reshape(len(idx), replan_steps * A),
        # --- actor(VLA) ---
        "full_action": ch,                                                    # (B, 16, 28)
        "vla_obs": vla_obs(idx), "vla_next_obs": vla_obs(nxt),
        # --- RL 필드 ---
        "reward": n["reward"], "mask": n["mask"], "valid": n["valid"], "done": n["done"],
        "is_success": flat.is_success[idx],
        "idx": idx, "next_idx": nxt,
    }


# --------------------------------------------------------------------------- #
# 이미지: 세션 → uint8 memmap
#
# EXPO-FT 는 리플레이 버퍼에 디코딩된 uint8 이미지를 그대로 들고 있다. 우리도 같은 방식이고,
# 파일(memmap)로 두므로 OOM 이 구조적으로 불가능하다 (OS 페이지 캐시가 알아서 내보낸다).
# GPU 에는 배치만 올린다 (64 프레임 = 23.6 MB).
#
# 디코딩은 RLDX 가 학습 때 쓰는 것과 같은 torchcodec 백엔드를 쓴다 — 캐시된 이미지가
# RLDX 가 모델에 넣는 것과 같도록.
# --------------------------------------------------------------------------- #
def video_keys(session: Path, mod: Modality) -> list[str]:
    """modality 가 선언한 카메라의 original_key 목록 (선언 순서). 데이터셋에 더 많은
    카메라가 있어도 모델이 쓰는 것만 캐싱한다."""
    info = json.loads((session / "meta" / "info.json").read_text())
    keys = [k for _, k in mod.video]
    missing = [k for k in keys if k not in info["features"]]
    if missing:
        raise ValueError(f"{session.name}: modality 가 요구하는 카메라가 없다: {missing}")
    return keys


def image_shape(session: Path, mod: Modality) -> tuple[int, int]:
    info = json.loads((session / "meta" / "info.json").read_text())
    c, h, w = info["features"][video_keys(session, mod)[0]]["shape"]
    return int(h), int(w)


def video_path(session: Path, key: str, episode: int, chunk_size: int = 1000) -> Path:
    return session / "videos" / f"chunk-{episode // chunk_size:03d}" / key / f"episode_{episode:06d}.mp4"


def build_images(sessions: list[Path], flat: Flat, out: Path, mod: Modality) -> dict:
    """(T, n_cams, H, W, 3) uint8 memmap 을 flat 과 같은 순서로 채운다."""
    from rldx.utils.video_utils import get_frames_by_indices

    keys = video_keys(sessions[0], mod)
    H, W = image_shape(sessions[0], mod)
    T = len(flat)
    out.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out, dtype=np.uint8, mode="w+", shape=(T, len(keys), H, W, 3))

    pos = ep_global = 0
    t0 = time.time()
    for path in sessions:
        n_ep = len(sorted(path.glob("data/chunk-*/episode_*.parquet")))
        for e in range(n_ep):
            L = flat.ep_length[ep_global]
            for ci, k in enumerate(keys):
                fr = get_frames_by_indices(str(video_path(path, k, e)), np.arange(L),
                                           video_backend="torchcodec")
                if fr.shape[1:3] != (H, W):
                    raise ValueError(f"해상도 불일치 {fr.shape[1:3]} != {(H, W)}: {path.name} ep{e}")
                mm[pos:pos + L, ci] = fr
            pos += L
            ep_global += 1
        print(f"  {path.name}: {n_ep} 에피소드  누적 {pos}/{T} 프레임  {time.time()-t0:.0f}s")
    mm.flush()
    meta = {"path": str(out), "shape": [T, len(keys), H, W, 3], "dtype": "uint8",
            "keys": keys, "sessions": [p.name for p in sessions],
            "bytes": T * len(keys) * H * W * 3, "seconds": round(time.time() - t0, 1)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def open_images(out: Path) -> tuple[np.memmap, dict]:
    meta = json.loads(out.with_suffix(".json").read_text())
    mm = np.memmap(out, dtype=np.uint8, mode="r", shape=tuple(meta["shape"]))
    return mm, meta


def verify_images(root: Path, out: Path, mod: Modality, n_probe: int = 24) -> int:
    from rldx.utils.video_utils import get_frames_by_indices

    sessions = find_sessions(root)
    flat = build_flat(sessions, mod)
    mm, meta = open_images(out)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print(f"memmap {meta['shape']}  {meta['bytes']/1e9:.1f} GB  디코딩 {meta['seconds']}s")
    check("A shape[0] == 프레임 수", meta["shape"][0] == len(flat), f"{meta['shape'][0]} vs {len(flat)}")
    check("A 카메라 수/키", meta["shape"][1] == len(meta["keys"]), str(meta["keys"]))
    check("A 파일 크기 == 계산치", out.stat().st_size == meta["bytes"],
          f"{out.stat().st_size} vs {meta['bytes']}")

    # 전역 인덱스 → (세션, 에피소드, 에피소드 내 프레임) 을 flat 에서 되찾아 비교
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(flat), size=n_probe)
    strict_bad, loose_max = 0, 0.0
    for i in idx:
        si = int(flat.session[i]); e_local = int(flat.episode[i]); fr = int(flat.frame[i])
        # 전역 에피소드 번호를 세션 내 번호로 되돌린다
        base = min(int(x) for x in flat.episode[flat.session == si])
        e = e_local - base
        for ci, k in enumerate(meta["keys"]):
            vp = video_path(sessions[si], k, e)
            got = np.asarray(mm[i, ci])
            ref = get_frames_by_indices(str(vp), np.array([fr]), video_backend="torchcodec")[0]
            if not np.array_equal(got, ref):
                strict_bad += 1
            try:
                import cv2
                cap = cv2.VideoCapture(str(vp)); cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, bgr = cap.read(); cap.release()
                if ok:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    loose_max = max(loose_max, float(np.abs(rgb.astype(np.int16) - got.astype(np.int16)).mean()))
            except Exception:
                pass
    check("B 인덱스 정합 (torchcodec 재디코딩과 비트 일치)", strict_bad == 0,
          f"{n_probe*len(meta['keys'])}개 프레임 중 불일치 {strict_bad}")
    check("C 독립 디코더(cv2)와 평균 차이 < 8", loose_max < 8.0, f"최대 평균차 {loose_max:.2f}")

    nz = float((np.asarray(mm[idx[:8]]) > 0).mean())
    print(f"  [수치] 표본 픽셀 중 0 이 아닌 비율 {nz:.1%} (검은 프레임만 있으면 낮게 나온다)")
    print(f"\n{'전부 통과' if not fails else f'{len(fails)}개 실패: ' + ', '.join(fails)}")
    return 1 if fails else 0


def verify_batch(root: Path, out: Path, mod: Modality, batch_size: int = 64,
                 replan_steps: int = 8, action_horizon: int = 16) -> int:
    sessions = find_sessions(root)
    flat = build_flat(sessions, mod)
    imgs, meta = open_images(out)
    cams = meta["keys"]
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(flat) - replan_steps, size=batch_size)
    b = make_batch(flat, imgs, idx, mod, replan_steps, action_horizon, task="pick up")
    H, W = meta["shape"][2], meta["shape"][3]
    A = mod.action_dim
    B = batch_size

    check("a obs shape (B,H,W,3*cams)", b["obs"].shape == (B, H, W, 3 * len(cams)), str(b["obs"].shape))
    check("a next_obs shape 동일", b["next_obs"].shape == b["obs"].shape)
    check("a state shape (B, state_dim)", b["state"].shape == (B, mod.state_dim),
          f"{b['state'].shape} (modality state_dim={mod.state_dim})")
    check("a action shape (B, replan*A)", b["action"].shape == (B, replan_steps * A), str(b["action"].shape))
    check("a full_action shape (B,H_a,A)", b["full_action"].shape == (B, action_horizon, A),
          str(b["full_action"].shape))

    check("b action == full_action[:, :replan] 평탄화",
          bool(np.allclose(b["action"], b["full_action"][:, :replan_steps].reshape(B, -1))))

    # 이미지가 실제로 그 인덱스의 두 카메라를 채널로 이어붙인 것인지
    ref = np.concatenate([np.asarray(imgs[idx])[:, c] for c in range(len(cams))], axis=-1)
    check("c obs == 두 카메라 채널 concat", bool(np.array_equal(b["obs"], ref)))
    ref_n = np.concatenate([np.asarray(imgs[b["next_idx"]])[:, c] for c in range(len(cams))], axis=-1)
    check("d next_obs == idx+replan 의 이미지", bool(np.array_equal(b["next_obs"], ref_n)))
    check("d next_idx == idx + replan", bool((b["next_idx"] == idx + replan_steps).all()))

    # state 그룹 분할이 원본을 정확히 덮는지
    groups = np.concatenate([b["vla_obs"]["state"][n][:, 0] for n, _, _ in mod.offsets("state")],
                            axis=-1)
    check("e VLA state 그룹 concat == critic state", bool(np.allclose(groups, b["state"])),
          f"그룹 {[n for n,_,_ in mod.offsets('state')]}")

    # mask==1 이면 next 가 같은 에피소드 안이어야 한다
    m1 = b["mask"] == 1
    same_ep = flat.episode[b["next_idx"][m1]] == flat.episode[idx[m1]]
    check("f mask==1 이면 next 가 같은 에피소드", bool(same_ep.all()),
          f"mask==1 표본 {int(m1.sum())}개")

    # VLA 입력 모양
    v = b["vla_obs"]
    ok_v = (all(x.shape == (B, 1, H, W, 3) for x in v["video"].values())
            and all(v["state"][n].shape == (B, 1, e - s) for n, s, e in mod.offsets("state"))
            and len(v["language"][mod.task_key]) == B)
    check("g VLA 입력 shape (video (B,1,H,W,3) / state (B,1,d) / language B)", ok_v,
          f"video={ {k: tuple(x.shape) for k,x in v['video'].items()} }")

    print(f"\n  [수치] 배치 메모리 obs+next_obs {(b['obs'].nbytes + b['next_obs'].nbytes)/1e6:.1f} MB")
    print(f"  [수치] 보상 != 0 인 표본 {int((b['reward'] > 0).sum())}/{B}, "
          f"mask==0 {int((b['mask'] == 0).sum())}/{B}, is_success {int(b['is_success'].sum())}/{B}")
    print(f"\n{'전부 통과' if not fails else f'{len(fails)}개 실패: ' + ', '.join(fails)}")
    return 1 if fails else 0


# --------------------------------------------------------------------------- #
def verify(root: Path, mod: Modality, replan_steps: int = 8, discount: float = 0.99) -> int:
    sessions = find_sessions(root)
    print(f"세션 {len(sessions)}개  ({root})")
    print(f"modality: state {mod.state_dim}차원 {len(mod.state)}그룹 / action {mod.action_dim}차원 "
          f"/ 카메라 {mod.n_cams}개  tag={mod.embodiment_tag or '(없음)'}")
    print(f"레이아웃 출처: {mod.layout_source}")
    print(f"  action concat 순서: {[n for n, _, _ in mod.offsets('action')]}")
    print(f"  카메라 순서       : {[n for n, _ in mod.video]}")
    print(f"robot_type: "
          f"{json.loads((sessions[0] / 'meta' / 'info.json').read_text()).get('robot_type')}")
    flat = build_flat(sessions, mod)
    n_ep = len(flat.ep_success)
    n_succ = sum(flat.ep_success)
    fails = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print(f"프레임 {len(flat)}  에피소드 {n_ep}  성공 {n_succ} ({n_succ/n_ep:.1%})\n")

    # 1. info.json 의 총 프레임/에피소드 수와 일치
    tot_f = tot_e = 0
    for p in sessions:
        info = json.loads((p / "meta" / "info.json").read_text())
        tot_f += int(info["total_frames"]); tot_e += int(info["total_episodes"])
    check("1 프레임/에피소드 수가 info.json 과 일치",
          (tot_f, tot_e) == (len(flat), n_ep), f"info={tot_f}/{tot_e} 로드={len(flat)}/{n_ep}")

    # 2. 보상 프레임 수 == 성공 에피소드 수, 위치는 에피소드 마지막
    rpos = np.flatnonzero(flat.reward > 0)
    last_frame = np.zeros(len(flat), dtype=bool)
    ep_change = np.flatnonzero(np.diff(flat.episode) != 0)
    last_frame[ep_change] = True
    last_frame[-1] = True
    check("2 보상 프레임 수 == 성공 에피소드 수", len(rpos) == n_succ, f"{len(rpos)} vs {n_succ}")
    check("2 보상이 에피소드 마지막 프레임에만", bool(last_frame[rpos].all()),
          f"마지막이 아닌 위치 {int((~last_frame[rpos]).sum())}개")

    # 3. done 이 에피소드 마지막에만 (독립 소스 교차 확인)
    check("3 done 위치 == 에피소드 경계", bool((flat.done == last_frame).all()),
          f"done={int(flat.done.sum())} 경계={int(last_frame.sum())}")

    # 4. n-step 리턴이 γ^k 인지
    ok4 = True; detail4 = []
    succ_ends = rpos[:20]
    for e in succ_ends:
        for k in (0, 1, 3, 7, 8, 12):
            t = e - k
            if t < 0 or t + replan_steps >= len(flat):
                continue
            got = nstep(flat, np.array([t]), replan_steps, discount)["reward"][0]
            exp = discount ** k if k < replan_steps else 0.0
            # 창 안에 다른 에피소드 종료가 먼저 오면 mask 로 잘리므로 그 경우는 건너뛴다
            if flat.done[t:e].any():
                continue
            if abs(got - exp) > 1e-5:
                ok4 = False; detail4.append(f"t={t} k={k} got={got:.6f} exp={exp:.6f}")
    check("4 n-step 보상 == γ^k (k<8), 0 (k>=8)", ok4, "; ".join(detail4[:3]))

    # 5. 경계 포함 여부에 따른 mask
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(flat) - replan_steps, size=20000)
    out = nstep(flat, idx, replan_steps, discount)
    has_end = np.array([flat.done[t:t + replan_steps].any() for t in idx])
    check("5 창이 에피소드 끝을 포함하면 mask==0",
          bool((out["mask"][has_end] == 0).all()), f"{int(has_end.sum())}개 표본")
    check("5 창이 에피소드 안쪽이면 mask==1",
          bool((out["mask"][~has_end] == 1).all()), f"{int((~has_end).sum())}개 표본")

    # 6. valid==0 ⟹ mask==0
    v0 = out["valid"] == 0
    check("6 valid==0 이면 mask==0", bool((out["mask"][v0] == 0).all()) if v0.any() else True,
          f"valid==0 표본 {int(v0.sum())}개")

    # 7. is_success 프레임 수
    exp_sf = sum(l for l, ok in zip(flat.ep_length, flat.ep_success) if ok)
    check("7 is_success 프레임 수 == 성공 에피소드 총 프레임",
          int(flat.is_success.sum()) == exp_sf, f"{int(flat.is_success.sum())} vs {exp_sf}")

    # 8. truncated 는 실제로 안 쓰이는지 (mask = 1-done 가정의 근거)
    check("8 truncated 가 전부 0 (mask=1-done 가정)", not bool(flat.truncated.any()),
          f"truncated={int(flat.truncated.sum())}")

    # 9~12. 액션 청크 (EXPO-FT back-fill 의미)
    H = 16
    ch = action_chunk(flat, idx, H)
    check("9 청크 shape (B,H,action_dim)",
          ch.shape == (len(idx), H, mod.action_dim), str(ch.shape))
    check("10 chunk[:,0] == action[t]", bool(np.allclose(ch[:, 0], flat.action[idx])))

    # 창이 에피소드 안쪽인 표본: 청크가 실제 미래 액션과 정확히 같아야 한다
    room = flat.ep_end[idx] - idx + 1
    inside = room >= H
    exp_in = np.stack([flat.action[t:t + H] for t in idx[inside][:2000]])
    check("11 여유가 충분하면 chunk == action[t:t+H]",
          bool(np.allclose(ch[inside][:2000], exp_in)), f"{int(inside.sum())}개 표본")

    # 끝에 걸린 표본: 뒤가 마지막 액션의 반복이어야 하고, 반복 개수가 H-room 이어야 한다
    near = ~inside
    ok12 = True; bad = 0
    for t, c in list(zip(idx[near], ch[near]))[:2000]:
        r = int(flat.ep_end[t] - t + 1)
        last = flat.action[flat.ep_end[t]]
        if not np.allclose(c[:r], flat.action[t:t + r]) or not np.allclose(c[r:], last):
            ok12 = False; bad += 1
    check("12 끝에 걸리면 마지막 액션 반복 패딩", ok12,
          f"{int(near.sum())}개 표본 중 불일치 {bad}")

    # 청크가 에피소드 경계를 넘지 않는지 (인덱스 기준으로 직접 확인)
    off = np.minimum(idx[:, None] + np.arange(H), flat.ep_end[idx][:, None])
    check("13 청크가 에피소드 경계를 넘지 않음",
          bool((flat.episode[off] == flat.episode[idx][:, None]).all()))

    # 9. 신호 희박도 (합격/불합격 아님)
    nz = float((out["reward"] > 0).mean())
    print(f"\n  [수치] 보상 != 0 인 transition 비율 {nz:.2%}  "
          f"(샘플 가능한 시작 인덱스 {len(flat)-replan_steps})")
    print(f"  [수치] 실효 할인 γ^{replan_steps} = {discount**replan_steps:.4f}")
    print(f"  [수치] actor(success_only) 풀 {int(flat.is_success.sum())} 프레임 "
          f"({flat.is_success.mean():.1%})")

    print(f"\n{'전부 통과' if not fails else f'{len(fails)}개 실패: ' + ', '.join(fails)}")
    return 1 if fails else 0


USAGE = """
  verify        <세션부모>            [--modality <경로>]
  images        <세션부모> <출력.mm>  [--modality <경로>]
  check-images  <세션부모> <출력.mm>  [--modality <경로>]
  check-batch   <세션부모> <출력.mm>  [--modality <경로>]
  show-modality <세션부모 | modality 경로>

공통 옵션:
  --rldx-config <RLDX-1 기준 상대경로>   **필수.** concat 순서와 embodiment_tag 의 정본
                                          (예: rldx/configs/data/openarm_inspire_config.py)
  --base-policy <체크포인트 경로>          있으면 processor_config 와 순서를 교차검증
  --rldx-root   <경로>                    기본 third_party/RLDX-1
  --modality    <경로>                    modality.json 덮어쓰기

embodiment 는 기본적으로 **데이터셋에서** 읽는다 (<세션>/meta/modality.json, convert_data.py
--modality 가 심어둔 것). 세션들이 서로 다르면 실패한다. --modality 로 덮어쓸 수 있다.
"""

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__ + USAGE)
    argv = sys.argv[1:]
    override = rldx_config = None
    rldx_root = Path(__file__).resolve().parent.parent / "third_party" / "RLDX-1"
    base_policy = None
    for flag in ("--modality", "--rldx-config", "--rldx-root", "--base-policy"):
        if flag in argv:
            i = argv.index(flag)
            val = argv[i + 1]
            if flag == "--modality":
                override = Path(val)
            elif flag == "--rldx-config":
                rldx_config = val
            elif flag == "--base-policy":
                base_policy = Path(val)
            else:
                rldx_root = Path(val)
            argv = argv[:i] + argv[i + 2:]
    cmd, pos = argv[0], argv[1:]

    if cmd == "show-modality":
        src = Path(pos[0])
        mod = load_modality(src) if not (src / "meta").is_dir() and not find_sessions(src) \
            else modality_from_sessions(find_sessions(src))
        print(f"state  {mod.state_dim}차원")
        for n, s, e in mod.offsets("state"):
            print(f"   {n:24s} concat {s:3d}:{e:3d}")
        print(f"action {mod.action_dim}차원")
        for n, s, e in mod.offsets("action"):
            print(f"   {n:24s} concat {s:3d}:{e:3d}")
        print(f"video  {[k for _, k in mod.video]}")
        print(f"task   {mod.task_key}")
        print(f"columns {mod.columns()}")
        sys.exit(0)

    root = Path(pos[0])
    mod, src = resolve_modality(root, override, rldx_root, rldx_config, base_policy)
    print(f"[modality] {src}")
    if cmd == "verify":
        sys.exit(verify(root, mod))
    if cmd == "images":
        s = find_sessions(root)
        m = build_images(s, build_flat(s, mod), Path(pos[1]), mod)
        print(json.dumps(m, indent=2, ensure_ascii=False))
        sys.exit(0)
    if cmd == "check-images":
        sys.exit(verify_images(root, Path(pos[1]), mod))
    if cmd == "check-batch":
        sys.exit(verify_batch(root, Path(pos[1]), mod))
    sys.exit(f"모르는 명령: {cmd}")
