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
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from rl.data import build_flat, find_sessions, open_images, resolve_modality

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
a = p.parse_args()

exp = yaml.safe_load((REPO / "configs/exp" / f"{a.exp}.yaml").read_text())
work = a.checkpoints / f"{a.exp}-critic"
base = a.checkpoints / (a.model_path or exp["base_policy"])
out_path = work / a.out
meta_path = out_path.with_suffix(".json")

mod, src = resolve_modality(a.data, None, RLDX, exp["rldx_data_config"], base)
sessions = find_sessions(a.data)
flat = build_flat(sessions, mod)
imgs, meta = open_images(work / "images.mm")
T = len(flat)
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
start = 0
if a.resume and out_path.is_file() and meta_path.is_file():
    start = int(json.loads(meta_path.read_text()).get("done", 0))
    feats = np.lib.format.open_memmap(out_path, mode="r+")
    print(f"[이어받기] {start}/{T} 프레임")
else:
    feats = None

t0 = time.time()
for c in range(start, T, a.batch):
    idx = np.arange(c, min(c + a.batch, T))
    with torch.no_grad():
        try:
            vla.runtime._forward(vla._collate(vla_obs(idx)))
        except StopIteration:
            pass
    f = grab["f"]                                    # (B, seq, d) 또는 (B, n_cog, d)
    z = f[:, -n_cog:, :].float().mean(1).cpu().numpy().astype(dt)
    if feats is None:
        feats = np.lib.format.open_memmap(out_path, mode="w+", dtype=dt, shape=(T, z.shape[1]))
        print(f"[출력] {out_path}  ({T}, {z.shape[1]}) {a.dtype} = "
              f"{T * z.shape[1] * np.dtype(dt).itemsize / 1e6:.0f} MB")
        print(f"[hook] backbone_features {tuple(f.shape)} → cog {n_cog}개 mean-pool → {z.shape[1]}")
    feats[idx] = z
    if (c // a.batch) % 20 == 0:
        el = time.time() - t0
        done = c + len(idx) - start
        print(f"  {c + len(idx)}/{T}  {el:.0f}s  "
              f"({done / max(el, 1e-9):.1f} 프레임/s, 남은 {(T - c - len(idx)) / max(done / max(el,1e-9), 1e-9) / 60:.0f}분)",
              flush=True)
        meta_path.write_text(json.dumps({"done": int(c + len(idx)), "T": T, "n_cog": n_cog,
                                         "dim": int(z.shape[1]), "dtype": a.dtype,
                                         "model": str(base), "sessions": [s.name for s in sessions]},
                                        indent=2) + "\n")
feats.flush()
meta_path.write_text(json.dumps({"done": T, "T": T, "n_cog": n_cog, "dim": int(feats.shape[1]),
                                 "dtype": a.dtype, "model": str(base),
                                 "sessions": [s.name for s in sessions]}, indent=2) + "\n")
print(f"[완료] {out_path}  {time.time()-t0:.0f}s")
