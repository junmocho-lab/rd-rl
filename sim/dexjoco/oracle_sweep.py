#!/usr/bin/env python3
"""오라클 액션을 앞 K 프레임 깔고 나머지는 정책에 넘겨 성공률을 잰다 — **여러 오라클 x 여러 K**.

배경 (전부 고정 장면 seed 0, 오라클 ep73 기준 실측):
  · 오라클 액션을 **전 구간** 재생하면 10/10 성공, 노이즈 0.005 에서 40%, 0.01 에서 10%
  · 오라클 궤적 위에서는 정책이 오라클과 거의 같은 청크를 낸다 (차원당 0.0022~0.0042).
    차이는 첫 청크에만 몰려 있었다 (0.030)
  · 그런데 첫 청크를 오라클로 깔아도 성공률이 안 올랐다 (K=0 85% / K=20 70% / K=40 70% /
    K=60 80%) — "첫 청크가 승부를 정한다" 가설은 기각됐다
따라서 승부처가 어디인지 K 를 훑어 찾는다. 오라클 하나로는 표본이 얇아 10개로 늘린다.

**롤아웃을 LeRobot 세션으로 저장한다** — 나중에 critic 학습·분석에 그대로 쓰기 위해서다.
메타에 oracle_ep / oracle_k 를 남겨 어떤 조건의 데이터인지 구분할 수 있게 한다.
앞 K 프레임은 오라클 액션이라 정책 분포 밖이지만, 그것이 오히려 **액션 다양성**을 준다
(지금까지 critic 이 액션을 구분하지 못한 원인으로 후보 산포 부족이 유력했다).

재개 가능: meta/episodes.jsonl 을 읽어 (oracle_ep, K) 별로 이미 채운 개수를 세고 건너뛴다.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_raw_to_rldx import (  # noqa: E402
    DEFAULT_CAMERA_KEYS, TASK_CAMERAS, TASK_PROMPTS, Config as ConvertConfig,
    write_episode, write_meta,
)
from rollout_dexjoco import (  # noqa: E402
    Config as RolloutConfig, PolicyClient, action_25_to_env, build_env,
    chunk_from_response, env_state_to_25, obs_to_request, quat_wxyz_to_rot6d,
)


@dataclass
class Config:
    session: Path = Path("/workspace/junmo_cho/dexjoco/rollout/fixed/s0_n100")
    """오라클을 고를 원본 롤아웃 (고정 장면 seed 0)."""
    out: Path = Path("/workspace/junmo_cho/dexjoco/rollout/oracle_sweep")
    task: str = "hammer_nail"
    host: str = "127.0.0.1"
    port: int = 20461
    n_oracle: int = 8
    """오라클로 쓸 성공 에피소드 수 (0 이면 전부).

    오라클 수보다 **오라클당 반복 수**가 우선이다 — 한 오라클의 K 곡선이 신뢰할 만해야
    "어디까지 깔면 성공하는가" 를 그 오라클에 대해 말할 수 있다. 오라클을 많이 하고
    반복이 1~2 회면 셀마다 성공률이 0% 아니면 100% 라 곡선이 안 나온다."""
    pick: str = "spread"
    """spread = 길이 분포에 걸쳐 고르게 (170 짜리와 250 짜리가 섞이도록),
    fastest = 가장 빨리 성공한 것부터. 길이가 다양해야 K 를 길이로 나눈 비율이 의미를 가진다."""
    k_step: int = 20
    """K 를 이 간격으로 훑는다 (0, 20, 40, ...). **replan(20)의 배수여야 한다** —
    정책은 청크 경계에서 인계받고 그 직전 d=5 개 액션을 RTC prefix 로 받는다. K 가 경계에
    안 맞으면 청크 중간에 넘겨받게 되어 실제 서빙 규약과 달라진다.

    각 오라클마다 K 는 0 부터 (그 오라클 길이 - k_step) 까지다. 길이가 다르므로 오라클마다
    K 개수가 다르고, 그것이 맞다 — "길이의 몇 %까지 깔아야 하는가" 는 길이로 나눠 보면 된다."""
    reps: int = 1
    """(오라클, K) 조합당 롤아웃 수. 오라클 수를 최대로 하는 것이 우선이므로 작게 두고,
    감독 스크립트가 1 -> 2 -> 3 으로 패스를 올리며 채운다."""
    scene_seed: int = 0
    replan: int = 20
    rtc_delay: int = 5
    max_steps: int = 360
    image_size: int = 256
    fps: int = 50
    crf: int = 20


def main(cfg: Config) -> None:
    eps = [json.loads(l) for l in (cfg.session / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    oracles = sorted([e for e in eps if e.get("success")], key=lambda e: e["length"])
    if cfg.n_oracle > 0 and cfg.n_oracle < len(oracles):
        if cfg.pick == "spread":
            idx = np.linspace(0, len(oracles) - 1, cfg.n_oracle).round().astype(int)
            oracles = [oracles[i] for i in sorted(set(idx.tolist()))]
        else:
            oracles = oracles[:cfg.n_oracle]
    L = [e["length"] for e in oracles]
    KMAP = {e["episode_index"]: list(range(0, max(e["length"] - cfg.k_step, 0) + 1, cfg.k_step))
            for e in oracles}
    n_cell = sum(len(v) for v in KMAP.values())
    print(f"[오라클] {len(oracles)}개  길이 {min(L)}~{max(L)} (중앙 {sorted(L)[len(L)//2]})")
    print(f"[K] 0 부터 (길이-{cfg.k_step}) 까지 {cfg.k_step} 간격  →  조합 {n_cell}개, "
          f"조합당 {cfg.reps}회 = {n_cell*cfg.reps} 롤아웃")
    KALL = sorted({k for v in KMAP.values() for k in v})

    ORC = {}
    for e in oracles:
        i = e["episode_index"]
        ORC[i] = np.vstack([np.asarray(v, np.float32) for v in
                            pd.read_parquet(cfg.session / f"data/chunk-000/episode_{i:06d}.parquet",
                                            columns=["action_raw"])["action_raw"]])

    raw_cams = TASK_CAMERAS[cfg.task]
    cam_keys = dict(zip(DEFAULT_CAMERA_KEYS, raw_cams))
    prompt = TASK_PROMPTS[cfg.task]
    conv = ConvertConfig(input=cfg.out, output=cfg.out, task=cfg.task,
                         cameras=list(raw_cams), image_size=cfg.image_size,
                         fps=cfg.fps, crf=cfg.crf)

    # ── 재개: 이미 저장된 것을 세어 건너뛴다 ────────────────────────────────────
    meta_path = cfg.out / "meta/episodes.jsonl"
    metas: list[dict] = []
    if meta_path.is_file():
        metas = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
        metas = sorted(metas, key=lambda m: m["episode_index"])
    done_cnt: dict[tuple[int, int], int] = {}
    for m in metas:
        key = (int(m.get("oracle_ep", -1)), int(m.get("oracle_k", -1)))
        done_cnt[key] = done_cnt.get(key, 0) + 1
    gidx = sum(m["length"] for m in metas)
    if metas:
        print(f"[재개] 이미 {len(metas)} 에피소드 / {gidx} 프레임 저장됨")

    rc = RolloutConfig(task=cfg.task, image_size=cfg.image_size,
                       replan=cfg.replan, rtc_delay=cfg.rtc_delay)
    env = build_env(rc)
    client = PolicyClient(cfg.host, cfg.port)
    t_start = time.time()
    n_run = 0

    # K 를 바깥 루프에 둔다: 중간에 멈춰도 "K 하나가 15개 오라클 전부에서 채워진" 상태가
    # 되어 K 곡선을 그릴 수 있다. 오라클을 바깥에 두면 앞쪽 오라클만 완성되고 뒤는 빈다.
    # K 를 바깥 루프에 둔다: 중간에 멈춰도 작은 K 들이 모든 오라클에서 채워져 곡선이 그려진다.
    for K in KALL:
        for e in oracles:
            oid = e["episode_index"]
            if K not in KMAP[oid]:
                continue                       # 이 오라클보다 긴 K 는 건너뛴다
            A_raw = ORC[oid]
            k_eff = K
            have = done_cnt.get((oid, K), 0)
            if have >= cfg.reps:
                continue
            n_ok = sum(1 for m in metas
                       if m.get("oracle_ep") == oid and m.get("oracle_k") == K and m.get("success"))
            for r in range(have, cfg.reps):
                random.seed(cfg.scene_seed); np.random.seed(cfg.scene_seed)
                obs, _ = env.reset()
                client.reset()
                states, acts, acts_raw = [], [], []
                succs, dones, truncs = [], [], []
                frames_buf = {k: [] for k in cam_keys}
                raw_hist, chunk, chunk_pos = [], None, 0
                first_call, success, depth = True, False, 0.0
                t = 0
                while t < cfg.max_steps and not success:
                    if t < k_eff:
                        a25 = A_raw[t]
                        chunk = None
                    else:
                        if chunk is None or chunk_pos >= cfg.replan:
                            req, fr = obs_to_request(obs, cam_keys, prompt, cfg.image_size)
                            opts = {}
                            if first_call:
                                opts["reset_memory"] = [True]
                            if cfg.rtc_delay > 0 and len(raw_hist) >= cfg.rtc_delay:
                                opts["action_prefix"] = np.stack(
                                    raw_hist[-cfg.rtc_delay:]).astype(np.float32)[None]
                            ad, _ = client.get_action(req, opts or None)
                            full = chunk_from_response(ad)
                            start = cfg.rtc_delay if (cfg.rtc_delay > 0 and
                                                      len(raw_hist) >= cfg.rtc_delay) else 0
                            chunk, chunk_pos, first_call = full[start:], 0, False
                        a25 = chunk[chunk_pos]; chunk_pos += 1
                    env_a = action_25_to_env(a25)
                    exec25 = np.concatenate([env_a[0:3], quat_wxyz_to_rot6d(env_a[3:7]),
                                             env_a[7:23]]).astype(np.float32)
                    # (s, a) 는 step 전에 기록한다 — 데모와 같은 규약
                    states.append(env_state_to_25(obs["state"]))
                    acts.append(exec25)
                    acts_raw.append(np.asarray(a25, np.float32))
                    raw_hist.append(np.asarray(a25, np.float32))
                    _, fr = obs_to_request(obs, cam_keys, prompt, cfg.image_size)
                    for k in cam_keys:
                        frames_buf[k].append(fr[k])
                    obs, _r, term, _tr, info = env.step(env_a)
                    depth = float(info.get("nail_depth", depth))
                    success = bool(info.get("succeed", False))
                    t += 1
                    succs.append(success); dones.append(bool(term) or success)
                    truncs.append(False)
                    if success or term:
                        break
                trunc = (not success) and t >= cfg.max_steps
                if trunc:
                    dones[-1] = True                    # 잘린 것도 종단으로 (rl/data.py 규약)
                ep_idx = len(metas)
                m = write_episode(
                    conv, ep_idx, np.stack(states), np.stack(acts),
                    {k: np.stack(v) for k, v in frames_buf.items()}, list(cam_keys),
                    gidx, success=np.array(succs, bool), done=np.array(dones, bool),
                    truncated=np.zeros(len(succs), bool),
                    extra_meta={"oracle_ep": oid, "oracle_k": K,
                                "oracle_frac": round(K / len(A_raw), 4),
                                "oracle_len": int(len(A_raw)), "rep": r,
                                "final_nail_depth": depth, "truncated": bool(trunc)},
                    extra_columns={"action_raw": np.stack(acts_raw)})
                metas.append(m); gidx += m["length"]; n_ok += success; n_run += 1
                write_meta(conv, metas, list(cam_keys), prompt=prompt)   # 크래시 안전
            el = time.time() - t_start
            print(f"  ep{oid:<3}(len {len(A_raw)}) K={K:<4}({100*K/len(A_raw):3.0f}%) "
                  f"{n_ok:>2}/{cfg.reps}  "
                  f"({100*n_ok/cfg.reps:5.1f}%)   누적 {n_run}롤아웃 "
                  f"{el/max(n_run,1):.1f}s/ep  경과 {el/60:.0f}분", flush=True)

    print(f"\n[완료] {len(metas)} 에피소드 → {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
