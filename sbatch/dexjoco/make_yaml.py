#!/usr/bin/env python3
"""Generate one MLXP (k8s Job) manifest per converted DexJoCo dataset.

Six near-identical yamls hand-copied drift apart — the epoch arithmetic in the
comments depends on each dataset's frame count, and every run shares the same
batch / horizon / resource block. So the template lives here and the per-task
numbers are read straight out of each dataset's `meta/info.json`.

    python sbatch/dexjoco/make_yaml.py                 # all datasets found
    python sbatch/dexjoco/make_yaml.py hammer_nail     # just one

Verified on the H200 node: per-device batch 64 fits and trains (predicted
98,631 MiB = 69% of 143,771 by the memory model measured on A100 —
mem(MiB) = 46,458 + 815.2 x batch, from batches 8/24/32/40).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCAL_DATASETS = Path("/rlwrld2/home/junmo_cho/ws/dexjoco_dataset")
OUT_DIR = REPO / "sbatch/dexjoco"

# --- the knobs every run shares -------------------------------------------------
ACTION_HORIZON = 32          # must equal len(action delta_indices) in the modality config
PER_DEVICE_BATCH = 64        # = --global-batch-size at 1 GPU; measured to fit on H200
MAX_STEPS = 30000
SAVE_STEPS = 10000
LEARNING_RATE = "1e-4"
NUM_GPUS = 1
DEADLINE_SECONDS = 172800    # 48h — see the comment block in the template

# Remote paths on the MLXP side (DDN mount).
REMOTE_REPO = "/data/junmo_cho/workspace/rd-rl/third_party/RLDX-1"
REMOTE_DATA_ROOT = "/data/junmo_cho/workspace/datasets/dexjoco"
REMOTE_CKPT_ROOT = "/data/rlwrld-unified-checkpoints/junmo_cho/checkpoints"
REMOTE_LOG_DIR = "/data/junmo_cho/workspace/logs"
REMOTE_HF_HOME = "/data/junmo_cho/hf_cache"

TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: junmo-cho-dexjoco-{dash}-bc
  namespace: p-rlwrld
  labels:
    project-code: rd
    project-short-name: rd
    mlxp/job-class: normal
    user: junmo_cho
spec:
  ttlSecondsAfterFinished: 300
  backoffLimit: 0
  # 처리량: A100-80GB / batch 32 / 40 step 실측 train_samples_per_second 12.13
  # (첫 스텝 워밍업 제외 약 14). H200 을 1.5~2x 로 보면 21~29 samples/s.
  #   {max_steps} step x batch {batch} = {total_samples:,} sample -> 18 ~ 25 h
  # 24h 는 아슬아슬해서 48h 로 둔다. 플랫폼이 거부하면 86400 으로 내리고
  # --max-steps 를 15000 으로 줄일 것 (epoch 은 아래 절반).
  activeDeadlineSeconds: {deadline}
  template:
    metadata:
      labels:
        project-code: rd
        project-short-name: rd
        mlxp/job-class: normal
        user: junmo_cho
      annotations:
        mlx.navercorp.com/zone: private-h200-rlwrld-0
        sidecar.istio.io/inject: "false"
    spec:
      restartPolicy: Never
      imagePullSecrets:
        - name: mlxp-registry
      nodeSelector:
        mlx.navercorp.com/zone: private-h200-rlwrld-0
      volumes:
        - name: ddn
          persistentVolumeClaim:
            claimName: ddn-rlwrld-shared
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
      containers:
        - name: main
          image: mlxp.kr.ncr.ntruss.com/rlwrld-gpu-base:latest
          env:
            - name: MODEL_OUTPUT_DIR
              value: {ckpt_root}/dexjoco-{dash}-bc-ptimg
            - name: HF_TOKEN
              value: {hf_token}
            - name: HUGGINGFACE_HUB_TOKEN
              value: {hf_token}
            - name: HF_HOME
              value: {hf_home}
            - name: WANDB_API_KEY
              value: {wandb_key}
          command: ["/bin/bash", "-c"]
          args:
            - |
              #!/bin/bash
              set -euo pipefail

              RUN_ID="$(date +%Y%m%d-%H%M%S)"
              LOG_DIR="{log_dir}"
              LOG_FILE="${{LOG_DIR}}/${{RUN_ID}}-dexjoco-{dash}.log"

              mkdir -p "$MODEL_OUTPUT_DIR"
              mkdir -p "$LOG_DIR"

              exec > >(tee "$LOG_FILE") 2>&1

              command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg)
              cd {remote_repo}
              source .venv/bin/activate
              export NO_ALBUMENTATIONS_UPDATE=1

              DATA={data_root}/{task}_rand_obj
              export NUM_GPUS={num_gpus}      # 아래 resources 의 nvidia.com/gpu 와 반드시 일치시킬 것

              # ---- 프리플라이트 --------------------------------------------------
              # modality config 는 rd-rl 의 서브모듈 체크아웃에 들어 있다. 없으면
              # launch_train 이 FileNotFoundError 로 죽으니 여기서 먼저, 20초 안에 잡는다.
              test -f rldx/configs/data/dexjoco_panda_allegro_config.py || {{
                  echo "[FATAL] rldx/configs/data/dexjoco_panda_allegro_config.py 없음."
                  echo "        /data/junmo_cho/workspace/rd-rl 에서 서브모듈까지 pull 할 것:"
                  echo "          git -C /data/junmo_cho/workspace/rd-rl pull"
                  echo "          git -C /data/junmo_cho/workspace/rd-rl submodule update --init --recursive"
                  exit 1
              }}
              python - <<'PY'
              import json, os
              D = "{data_root}/{task}_rand_obj"
              info = json.load(open(f"{{D}}/meta/info.json"))
              mod  = json.load(open(f"{{D}}/meta/modality.json"))
              assert info["codebase_version"] == "v2.1", info["codebase_version"]
              assert set(mod["video"]) == {{"camera_front", "camera_wrist"}}, mod["video"]
              assert info["features"]["action"]["shape"] == [25], info["features"]["action"]["shape"]
              assert info["features"]["observation.state"]["shape"] == [25]
              assert os.path.exists(f"{{D}}/meta/stats.json"), "meta/stats.json 없음"
              assert info["total_episodes"] == {episodes}, (info["total_episodes"], {episodes})
              assert info["total_frames"] == {frames}, (info["total_frames"], {frames})
              print(f"[ok] {{info['total_episodes']}} ep / {{info['total_frames']}} frames / fps {{info['fps']}}"
                    f" / cameras {{info.get('dexjoco_cameras')}}")
              PY

              # --- 배치 / step / epoch ----------------------------------------------
              # 1 GPU 에서는 per_device_train_batch_size = global_batch_size // num_gpus
              # (experiment.py:187) 이므로 --global-batch-size 가 곧 per-device 배치다.
              # A100-80GB 에서 배치별 peak GPU memory 를 재서 뽑은 선형 모델:
              #     batch  8 -> 53,443 MiB     batch 24 -> 65,277 MiB
              #     batch 32 -> 72,181 MiB     batch 40 -> 79,711 MiB     batch 64/128 -> OOM
              #     mem(MiB) = 46,458 + 815.2 x batch   (잔차 +-750 MiB)
              # -> H200 143,771 MiB 예측: batch 64 = 98,631 MiB (69%), batch 128 = 105% (OOM).
              #    batch {batch} 는 H200 노드에서 실제로 돌려서 확인했다.
              #
              # {task}: {episodes} ep / {frames:,} frame -> effective step {effective:,}
              #   (= frame - ep x (action_horizon-1). RLDX-1 은 에피소드 뒤쪽
              #    action_horizon-1 프레임을 버린다. 학습 시작 로그의
              #    "StandardSingleStepDataset: N steps" 와 일치해야 한다)
              # epoch = {max_steps} x {batch} / {effective:,} = {epochs:.1f}
              #   참고: DexJoCo 논문의 single-arm baseline (pi0.5 / GR00T N1.5) 은 같은
              #   100-에피소드 데이터셋에 LoRA 30,000 step (openpi 기본 배치 32) = 약 45
              #   epoch 를 썼다. 위 값이 그보다 크면 과적합 구간을 지날 수 있으니
              #   --save-steps {save_steps} 로 남는 중간 체크포인트들의 롤아웃 성공률을
              #   반드시 비교해서 최고점을 base policy 로 쓸 것.
              #
              # --- action-horizon {horizon} ------------------------------------------------
              # dexjoco_panda_allegro_config.py 의 action delta_indices=range({horizon}) 과
              # 반드시 일치해야 한다. 불일치면 assembly.py:305 의
              # _validate_action_horizon_matches_modality 가 ValueError 로 즉시 죽는다.
              # 데이터가 30 fps 이므로 {horizon} step = {horizon_sec:.2f} s.
              #
              # --- episode-sampling-rate -------------------------------------------
              # 기본값 0.1 (dataset_mode=sharded) 을 그대로 둔다 — openarm/rby1 실런과
              # 같은 설정. 100 에피소드 단일 태스크라 1.0 으로 올리는 것도 합리적이니
              # 첫 런의 loss 곡선을 보고 판단할 것.
              torchrun --nproc_per_node=$NUM_GPUS --standalone \\
                  rldx/experiment/launch_train.py \\
                  --base-model-path RLWRLD/RLDX-1-PT-IMG \\
                  --backbone-path RLWRLD/RLDX-1-VLM \\
                  --dataset-paths $DATA \\
                  --embodiment-tag GENERAL_EMBODIMENT \\
                  --modality-config-path rldx/configs/data/dexjoco_panda_allegro_config.py \\
                  --output-dir "$MODEL_OUTPUT_DIR" \\
                  --video-length 1 \\
                  --n-cog-tokens 64 \\
                  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \\
                  --global-batch-size {batch} \\
                  --learning-rate {lr} \\
                  --max-steps {max_steps} \\
                  --save-steps {save_steps} \\
                  --save-only-model \\
                  --dataloader-num-workers 8 \\
                  --experiment-name dexjoco_{task}_randobj_{episodes}ep_ptimg_h{horizon}_rtc8tr_drop03_bs{batch}_{steps_k}k_{num_gpus}gpu_mlxp \\
                  --use-wandb --wandb-project RLDX-1 \\
                  --action-horizon {horizon} \\
                  --rtc-training-max-delay 8 \\
                  --state-dropout-prob 0.3 \\
                  --num-gpus $NUM_GPUS \\
                  --random-crop-fraction 0.95
          volumeMounts:
            - name: ddn
              mountPath: /data
            - name: shm
              mountPath: /dev/shm
          # {num_gpus} GPU — 레퍼런스(8 GPU 96c/1408Gi)의 GPU당 12c/176Gi 비례 유지.
          resources:
            requests: {{ cpu: "12", memory: "176Gi", nvidia.com/gpu: "{num_gpus}" }}
            limits: {{ cpu: "12", memory: "176Gi", nvidia.com/gpu: "{num_gpus}" }}
"""


def read_secrets() -> tuple[str, str]:
    """HF / WANDB 키는 파일에서 읽는다 — 이 스크립트에 하드코딩하지 않는다."""
    hf = wandb = ""
    secrets = Path.home() / ".rldx_secrets.sh"
    if secrets.exists():
        for line in secrets.read_text().splitlines():
            if line.startswith("export HF_TOKEN="):
                hf = line.split("=", 1)[1].strip().strip('"')
            if line.startswith("export WANDB_API_KEY="):
                wandb = line.split("=", 1)[1].strip().strip('"')
    # 기존 yaml 에 이미 박혀 있던 값을 잃지 않도록, 없으면 그쪽에서 가져온다.
    for existing in sorted(OUT_DIR.glob("bc_*.yaml")):
        text = existing.read_text()
        for key, cur in (("HF_TOKEN", hf), ("WANDB_API_KEY", wandb)):
            if cur:
                continue
            marker = f"- name: {key}\n              value: "
            if marker in text:
                val = text.split(marker, 1)[1].splitlines()[0].strip()
                if key == "HF_TOKEN":
                    hf = val
                else:
                    wandb = val
    if not hf:
        raise SystemExit("HF_TOKEN 을 ~/.rldx_secrets.sh 나 기존 yaml 에서 찾지 못했다")
    return hf, wandb


def emit(task: str, hf_token: str, wandb_key: str) -> Path | None:
    info_path = LOCAL_DATASETS / f"{task}_rand_obj/meta/info.json"
    if not info_path.exists():
        print(f"[skip] {task}: {info_path} 없음 (변환 미완료)")
        return None
    info = json.loads(info_path.read_text())
    episodes = info["total_episodes"]
    frames = info["total_frames"]
    effective = frames - episodes * (ACTION_HORIZON - 1)
    total_samples = MAX_STEPS * PER_DEVICE_BATCH
    body = TEMPLATE.format(
        task=task,
        dash=task.replace("_", "-"),
        episodes=episodes,
        frames=frames,
        effective=effective,
        epochs=total_samples / effective,
        total_samples=total_samples,
        batch=PER_DEVICE_BATCH,
        lr=LEARNING_RATE,
        max_steps=MAX_STEPS,
        steps_k=MAX_STEPS // 1000,
        save_steps=SAVE_STEPS,
        horizon=ACTION_HORIZON,
        horizon_sec=ACTION_HORIZON / info["fps"],
        num_gpus=NUM_GPUS,
        deadline=DEADLINE_SECONDS,
        remote_repo=REMOTE_REPO,
        data_root=REMOTE_DATA_ROOT,
        ckpt_root=REMOTE_CKPT_ROOT,
        log_dir=REMOTE_LOG_DIR,
        hf_home=REMOTE_HF_HOME,
        hf_token=hf_token,
        wandb_key=wandb_key,
    )
    out = OUT_DIR / f"bc_{task}.yaml"
    out.write_text(body)
    print(
        f"[ok] {out.relative_to(REPO)}  {episodes} ep / {frames:,} frame / "
        f"eff {effective:,} / epoch {total_samples / effective:.1f}"
    )
    return out


def main(tasks: list[str]) -> None:
    hf_token, wandb_key = read_secrets()
    if not tasks:
        tasks = sorted(p.name.removesuffix("_rand_obj") for p in LOCAL_DATASETS.iterdir() if p.is_dir())
    for task in tasks:
        emit(task, hf_token, wandb_key)


if __name__ == "__main__":
    main(sys.argv[1:])
