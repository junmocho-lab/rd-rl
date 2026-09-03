#!/usr/bin/env python3
"""LoRA distill 체크포인트(디렉토리)를 serve 의 --artifacts 형식(.pt)으로 바꾼다.

왜: merge 가 필요 없다. `rl/vla_rldx.py:540` 의 ExpoServer._load() 가 이미
`sd["lora"]` 를 읽어 base 정책 위에 LoRA 를 주입한다 (EXPO-FT 가 쓰던 경로다):

    if sd.get("lora"):
        self.vla.setup_training(lora=True)          # LoRA 주입
        self.vla.model.load_state_dict(sd["lora"], strict=False)

문제는 **형식**뿐이었다:
    EXPO-FT      theta.pt          — torch.load 로 읽는 dict, sd["lora"] 에 텐서
    launch_train checkpoint-N/     — HF 디렉토리, model.safetensors 에 텐서
둘의 **키 이름은 같다** (실측: action_model.model.*.lora_{A,B}.default.weight 64개
+ backbone.cog_emb). 그래서 담는 그릇만 바꾸면 된다.

merge 와 비교:
    merge         13GB/arm  (base 사본을 만든다).  8 arm = 104GB,
                  스텝 ablation(8스텝 x 8arm)이면 832GB 로 감당 불가
    이 변환        11MB/체크포인트.  base 는 한 벌만 두고 어댑터만 갈아끼운다

  PY=third_party/RLDX-1/.venv/bin/python
  $PY utils/adapter_to_artifacts.py --arm success_m0.01 --all-steps
  $PY utils/adapter_to_artifacts.py --all --all-steps                # 전 arm 전 스텝

출력: **체크포인트 디렉토리 안에** adapter.pt 로 쓴다.
      checkpoints/fuji_distill/<arm>/<exp>/checkpoint-<N>/adapter.pt
      model.safetensors 옆에 두면 체크포인트를 옮기거나 지울 때 같이 따라가고,
      두 디렉토리를 대조할 일이 없다.
서빙:  ARM=success_m0.01 bash sbatch/fuji/rollout_w_critic/serve_distill.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parent.parent
# 기본은 fuji 지만 **하드코딩하면 안 된다** — dexjoco 판 lora_train.sbatch 도 이 스크립트를
# 부르는데 그쪽 출력은 checkpoints/dexjoco_distill/ 이라 조용히 "arm 이 없다" 로 끝난다.
DISTILL = REPO / "checkpoints/fuji_distill"


def find_ckpt(arm: str, step: int | None) -> Path:
    """<arm>/<experiment-name>/checkpoint-N 을 찾는다. step 이 없으면 최신."""
    root = DISTILL / arm
    cks = sorted(root.glob("*/checkpoint-*"),
                 key=lambda p: int(p.name.split("-")[1]))
    if not cks:
        raise SystemExit(f"체크포인트가 없다: {root}/*/checkpoint-*")
    if step is None:
        return cks[-1]
    want = [c for c in cks if int(c.name.split("-")[1]) == step]
    if not want:
        have = [int(c.name.split('-')[1]) for c in cks]
        raise SystemExit(f"step {step} 이 없다. 있는 것: {have}")
    return want[0]


def convert(arm: str, step: int | None, out_dir: Path | None = None) -> Path:
    ck = find_ckpt(arm, step)
    n = int(ck.name.split("-")[1])
    sd = load_file(ck / "model.safetensors")
    # 키는 그대로 둔다 — setup_training(lora=True) 이 주입하는 이름과 같아야
    # load_state_dict(strict=False) 가 붙는다. 이름을 건드리면 조용히 아무것도
    # 안 붙고 base BC 가 그대로 서빙된다 (ExpoServer._load 가 unexpected_keys 는
    # 잡지만 missing 은 안 잡는다).
    n_lora = sum("lora" in k for k in sd)
    # 체크포인트 디렉토리 안에 둔다 (model.safetensors 옆).
    out = (ck / "adapter.pt") if out_dir is None else (out_dir / f"{arm}@{n}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": sd,
                "meta": {"arm": arm, "step": n, "src": str(ck.relative_to(REPO)),
                         "n_tensors": len(sd), "n_lora": n_lora}}, out)
    mb = out.stat().st_size / 1e6
    print(f"  {arm}@{n:<6}  텐서 {len(sd)} (lora {n_lora})  ->  "
          f"{out.relative_to(REPO)}  ({mb:.1f}MB)")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", help="success_m0.01 등. --all 이면 생략")
    p.add_argument("--step", type=int, help="비우면 최신 체크포인트")
    p.add_argument("--all", action="store_true", help="fuji_distill 아래 전 arm")
    p.add_argument("--all-steps", action="store_true",
                   help="그 arm 의 **모든** 체크포인트를 변환한다 (스텝 ablation 용). "
                        "하나가 10MB 라 8스텝 x 8arm 이어도 640MB 다")
    p.add_argument("--root", type=Path, default=None,
                   help="distill 출력 루트 (기본 checkpoints/fuji_distill). dexjoco 면 "
                        "checkpoints/dexjoco_distill 을 준다")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="비우면 각 체크포인트 디렉토리 안에 adapter.pt 로 쓴다 (권장)")
    a = p.parse_args()
    global DISTILL
    if a.root is not None:
        DISTILL = a.root if a.root.is_absolute() else REPO / a.root

    if not a.all and not a.arm:
        return print("--arm 또는 --all 을 줄 것") or 2
    arms = ([d.name for d in sorted(DISTILL.iterdir())
             if d.is_dir() and d.name not in ("artifacts", "smoke")
             and not d.name.endswith("_merged") and list(d.glob("*/checkpoint-*"))]
            if a.all else [a.arm])
    if not arms:
        return print(f"변환할 arm 이 없다: {DISTILL}") or 2
    print(f"[변환] {len(arms)} arm -> " + (str(a.out_dir) if a.out_dir else "각 checkpoint-N/adapter.pt"))
    for arm in arms:
        try:
            if a.all_steps:
                steps = sorted(int(c.name.split("-")[1])
                               for c in (DISTILL / arm).glob("*/checkpoint-*"))
                for st in steps:
                    convert(arm, st, a.out_dir)
            else:
                convert(arm, a.step, a.out_dir)
        except SystemExit as e:
            print(f"  {arm}: 건너뜀 — {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
