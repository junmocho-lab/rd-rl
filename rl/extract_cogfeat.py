#!/usr/bin/env python3
"""RLDX-1 의 cognition token 을 mean-pool 해서 critic 입력용 feature 로 캐시한다.

왜: critic 비전 인코더(34.1M, from-scratch)가 60 에피소드에 과적합해 **에피소드를 암기**했다
(실측: 픽셀 1-NN 은 성패를 못 맞추는데 critic 은 초기 프레임에서 AUC 0.94~1.00). BC 로
사전학습된 표현을 frozen 으로 쓰면 학습 파라미터가 38.5M → 6.5M 으로 줄고, 롤아웃에서도
VLA 백본이 관측당 1회 이미 돌기 때문에 추가 비용이 0 이다.

cognition token 이란: 학습된 임베딩 `cog_emb` (n_cog_tokens=64) 를 VLM 입력 뒤에 붙이고
(`rldx/model/modules/backbone/adapter.py:454`) 그 위치의 백본 출력을 쓰는 것. action expert 가
attend 하는 대상이라 **정책이 액션을 낼 때 실제로 보는 표현**이다.

출력: <work>/cogfeat.npy   (T, 4096) float32   fuji 55,564 프레임 → 910MB

usage:
  PYTHONPATH="$L_PYTHONPATH" $L_PY -u -m rl.extract_cogfeat \\
      --exp fuji --data rl-dataset/fuji-rl-dataset --checkpoints checkpoints \\
      --model-path rldx-img-curated/rldx_img_curated-0810-0818-r05
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from rl.data import build_flat, build_images, find_sessions, open_images, resolve_modality

REPO = Path(__file__).resolve().parent.parent
RLDX = REPO / "third_party/RLDX-1"

p = argparse.ArgumentParser()
p.add_argument("--exp", default="fuji")
p.add_argument("--data", type=Path, required=True)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--model-path", default="", help="비우면 exp yaml 의 base_policy")
p.add_argument("--out", default="cogfeat.npy")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--dtype", default="float32", choices=("float32", "float16"))
p.add_argument("--resume", action="store_true", help="이어서 (부분 파일이 있으면)")
p.add_argument("--shard", default="",
               help="멀티 GPU 병렬: 'i/N' 이면 프레임 [i·T/N, (i+1)·T/N) 만 뽑아 "
                    "<out>.partI_of_N.npy 로 쓴다. GPU 마다 CUDA_VISIBLE_DEVICES 를 달리해 "
                    "N개 프로세스를 띄우고, 전부 끝나면 --merge N 으로 합친다.\n"
                    "**images.mm 이 먼저 완성돼 있어야 한다** (N개가 동시에 디코딩하면 "
                    "memmap 이 깨진다) — 한 번은 --shard 없이 돌리거나 rl.data images 로 만들 것.\n"
                    "**OMP_NUM_THREADS=<코어수/N> 를 꼭 걸 것** — 안 걸면 프로세스마다 torch 가 "
                    "전체 코어만큼 스레드를 만들어 서로 싸운다 (실측 H200x4/128코어: 캡 없이 "
                    "9 fps, OMP=16 으로 213 fps — 23배). torchrun 는 자동으로 1 로 잡아주지만 "
                    "이 스크립트는 수동 병렬이라 직접 걸어야 한다")
p.add_argument("--merge", type=int, default=0,
               help="N 개의 .partI_of_N.npy 를 <out> 으로 합치고 part 파일을 지운다 (GPU 불필요)")
a = p.parse_args()

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / (a.model_path or exp["base_policy"])
out_path = work / a.out
meta_path = out_path.with_suffix(".json")

mod, src = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
work.mkdir(parents=True, exist_ok=True)
T = len(flat)

# ── 샤드/머지: 멀티 GPU 병렬 추출 ────────────────────────────────────────────
SHARD = None
if a.shard:
    si, sn = (int(x) for x in a.shard.split("/"))
    if not (0 <= si < sn):
        raise SystemExit(f"--shard {a.shard}: i/N 에서 0 <= i < N 이어야 한다")
    SHARD = (si, sn)
    out_path = work / f"{Path(a.out).stem}.part{si}_of_{sn}.npy"
    meta_path = out_path.with_suffix(".json")
LO, HI = (0, T) if SHARD is None else (SHARD[0] * T // SHARD[1],
                                       (SHARD[0] + 1) * T // SHARD[1])

if a.merge:
    # GPU 불필요 — part 들을 최종 npy 로 이어붙인다.
    sn = a.merge
    offs = [k * T // sn for k in range(sn + 1)]
    final = pm = None
    for k in range(sn):
        pp = work / f"{Path(a.out).stem}.part{k}_of_{sn}.npy"
        mp = pp.with_suffix(".json")
        if not (pp.is_file() and mp.is_file()):
            raise SystemExit(f"part 없음: {pp}")
        pm = json.loads(mp.read_text())
        if pm.get("T") != T or pm.get("done") != offs[k + 1]:
            raise SystemExit(f"{pp.name}: 미완성/불일치 (done={pm.get('done')} 필요 "
                             f"{offs[k+1]}, T={pm.get('T')} vs {T}) — 그 샤드를 다시 돌릴 것")
        part = np.lib.format.open_memmap(pp, mode="r")
        if final is None:
            final = np.lib.format.open_memmap(out_path, mode="w+", dtype=part.dtype,
                                              shape=(T, part.shape[1]))
        for c0 in range(0, offs[k + 1] - offs[k], 8192):
            c1 = min(c0 + 8192, offs[k + 1] - offs[k])
            final[offs[k] + c0:offs[k] + c1] = part[c0:c1]
    final.flush()
    meta_path.write_text(json.dumps({"done": T, "T": T, "n_cog": pm.get("n_cog"),
                                     "dim": int(final.shape[1]), "dtype": a.dtype,
                                     "model": str(base), "merged_from": sn,
                                     "sessions": [s.name for s in sessions]},
                                    indent=2) + "\n")
    for k in range(sn):
        (work / f"{Path(a.out).stem}.part{k}_of_{sn}.npy").unlink()
        (work / f"{Path(a.out).stem}.part{k}_of_{sn}.json").unlink()
    print(f"[merge] {out_path}  (T={T}, dim={final.shape[1]})  part {sn}개 삭제")
    raise SystemExit(0)

if SHARD is not None:
    # N개 프로세스가 동시에 비디오를 디코딩하면 images.mm 이 깨진다 — 완성본을 요구한다.
    imj = (work / "images.mm").with_suffix(".json")
    if not imj.is_file() or json.loads(imj.read_text())["shape"][0] != T:
        raise SystemExit("images.mm 이 없거나 이 --data 와 안 맞는다 — 먼저 --shard 없이 한 번 "
                         "돌려 images.mm 을 완성하거나 (추출 시작되면 끊어도 됨), "
                         "rl.data images 로 만들 것")
else:
    build_images(sessions, flat, work / "images.mm", mod)  # 없으면 만들고, 있으면 이어받기만
imgs, meta = open_images(work / "images.mm")
assert meta["shape"][0] == T, "images.mm 이 이 --data 로 만들어진 것이 아니다"
tasks = json.loads((sessions[0] / "meta/tasks.jsonl").read_text().splitlines()[0])
task = tasks["task"]
print(f"[데이터] 프레임 {T} / 세션 {len(sessions)} / 카메라 {mod.n_cams}\n[task] {task}")

from rl.vla_rldx import RLDXVLA                     # noqa: E402  (무거운 import 를 뒤로)
vla = RLDXVLA(base, mod, RLDX, exp["rldx_data_config"], device="cuda")
n_cog = int(getattr(vla.model, "_n_cog_tokens", getattr(vla.model.backbone, "n_cog_tokens", 64)))
print(f"[모델] {base.name}  n_cog_tokens={n_cog}")

# get_action_with_features 를 감싸 backbone_features 를 가로챈다 (expanded() 와 같은 자리).
grab = {}
_orig = vla.model.action_model.get_action_with_features

def hooked(backbone_features, state_features, embodiment_id, backbone_output, action_input=None):
    grab["f"] = backbone_features.detach()
    raise StopIteration                              # 디노이저는 돌릴 필요가 없다

vla.model.action_model.get_action_with_features = hooked

def vla_obs(idx):
    """make_batch 의 vla_obs 와 같은 규약 (video (B,1,H,W,3) / state (B,1,d) / language)."""
    x = np.asarray(imgs[idx])
    return {"video": {name: x[:, c][:, None] for c, (name, _) in enumerate(mod.video)},
            "state": {name: flat.state[idx][:, None, s:e] for name, s, e in mod.offsets("state")},
            "language": {mod.task_key: [[task]] * len(idx)}}

dt = np.float32 if a.dtype == "float32" else np.float16
ROWS = HI - LO                                       # 이 프로세스가 쓸 행 수 (샤드면 T/N)
start = LO
if a.resume and out_path.is_file() and meta_path.is_file():
    _pm = json.loads(meta_path.read_text())
    start = int(_pm.get("done", LO))
    feats = np.lib.format.open_memmap(out_path, mode="r+")
    if SHARD is not None:
        # 샤드는 T 가 변하면 경계가 같이 밀리므로 grow 가 성립하지 않는다 — 새로 뽑는다.
        if _pm.get("T") != T or feats.shape[0] != ROWS:
            raise SystemExit(f"{out_path.name}: T 가 달라졌다 (part T={_pm.get('T')} vs {T}) "
                             f"— part 파일들을 지우고 다시 돌릴 것")
    elif feats.shape[0] != T:
        # 세션이나 에피소드가 늘면 T 가 커진다. npy 는 헤더에 shape 이 박혀 있어 제자리에서
        # 늘릴 수 없으므로 새 파일로 옮겨 담는다 — 기존 추출분은 그대로 살린다.
        # 이 처리가 없으면 작은 파일에 쓰다 IndexError 로 죽고, 몇 시간짜리 추출을
        # 처음부터 다시 하게 된다 (실제로 겪었다: 288019 짜리에 309490 번째를 쓰려다 죽음).
        keep = min(start, feats.shape[0], T)
        tmp = out_path.with_suffix(".npy.grow")
        new_f = np.lib.format.open_memmap(tmp, mode="w+", dtype=feats.dtype,
                                          shape=(T, feats.shape[1]))
        for c0 in range(0, keep, 8192):
            c1 = min(c0 + 8192, keep)
            new_f[c0:c1] = feats[c0:c1]
        new_f.flush()
        del new_f, feats
        os.replace(tmp, out_path)
        feats = np.lib.format.open_memmap(out_path, mode="r+")
        start = keep
        print(f"[확장] cogfeat.npy 를 ({T}, {feats.shape[1]}) 로 늘렸다 "
              f"— 기존 {keep} 프레임 보존")
    print(f"[이어받기] {start - LO}/{ROWS} 프레임"
          + (f" (shard {a.shard}: [{LO},{HI}))" if SHARD else ""))
else:
    feats = None

if SHARD:
    print(f"[shard {a.shard}] 프레임 [{LO}, {HI}) = {ROWS}개 → {out_path.name}")
t0 = time.time()
for c in range(start, HI, a.batch):
    idx = np.arange(c, min(c + a.batch, HI))
    with torch.no_grad():
        try:
            vla.runtime._forward(vla._collate(vla_obs(idx)))
        except StopIteration:
            pass
    f = grab["f"]                                    # (B, seq, d) 또는 (B, n_cog, d)
    z = f[:, -n_cog:, :].float().mean(1).cpu().numpy().astype(dt)
    if feats is None:
        feats = np.lib.format.open_memmap(out_path, mode="w+", dtype=dt,
                                          shape=(ROWS, z.shape[1]))
        print(f"[출력] {out_path}  ({ROWS}, {z.shape[1]}) {a.dtype} = "
              f"{ROWS * z.shape[1] * np.dtype(dt).itemsize / 1e6:.0f} MB")
        print(f"[hook] backbone_features {tuple(f.shape)} → cog {n_cog}개 mean-pool → {z.shape[1]}")
    feats[idx - LO] = z
    if (c // a.batch) % 20 == 0:
        el = time.time() - t0
        done = c + len(idx) - start
        print(f"  {c + len(idx)}/{HI}  {el:.0f}s  "
              f"({done / max(el, 1e-9):.1f} 프레임/s, 남은 {(HI - c - len(idx)) / max(done / max(el,1e-9), 1e-9) / 60:.0f}분)",
              flush=True)
        meta_path.write_text(json.dumps({"done": int(c + len(idx)), "T": T, "lo": LO, "hi": HI,
                                         "n_cog": n_cog,
                                         "dim": int(z.shape[1]), "dtype": a.dtype,
                                         "model": str(base), "sessions": [s.name for s in sessions]},
                                        indent=2) + "\n")
feats.flush()
meta_path.write_text(json.dumps({"done": HI, "T": T, "lo": LO, "hi": HI, "n_cog": n_cog,
                                 "dim": int(feats.shape[1]),
                                 "dtype": a.dtype, "model": str(base),
                                 "sessions": [s.name for s in sessions]}, indent=2) + "\n")
print(f"[완료] {out_path}  {time.time()-t0:.0f}s")
