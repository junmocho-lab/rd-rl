# DexJoCo — the sim rehearsal of the real-robot RL loop

Purpose: run the whole BC -> rollout -> critic -> test-time-guidance chain in
simulation, where success is cheap to measure, instead of on the robot. Every
stage after BC reuses the existing `rl/` code unchanged; the only sim-specific
pieces are the dataset converter and (later) the rollout client.

Upstream benchmark: `third_party/dexjoco` (arXiv 2605.16257), 11 MuJoCo dexterous
tasks on a Franka Panda + Allegro hand.

## Task choice: `hammer_nail`

DexJoCo paper Table 2, "rand-obj" regime. GR00T N1.5 is the closest
architectural relative of RLDX-1, so its column is the BC estimate to plan
against:

| task | GR00T N1.5 | pi0.5 | mean episode length |
| --- | --- | --- | --- |
| **hammer_nail** | **67.3%** | 84.7% | **215.7 frames / 7.2 s** |
| water_plant | 72.7% | 88.7% | 277.4 / 9.2 s |
| click_mouse | 85.3% | 64.7% | 326.8 / 10.9 s |
| pick_bucket | 72.0% | 84.0% | 433.0 / 14.4 s |
| fold_glasses | 27.3% | 72.0% | 536.3 / 17.9 s |
| pinch_tongs | 12.7% | 24.0% | 400.6 / 13.4 s |

`hammer_nail` is the shortest horizon of all 11 tasks and leaves ~30 pt of
headroom over BC — the band where a critic can actually show an effect.
click_mouse and pick_bucket are nearly saturated for this model family;
pinch_tongs and fold_glasses are too weak to improve on; every bimanual task is
both longer and near zero. Its success condition (nail depth >= 0.04 m, advanced
0.008 m per impact that lands with enough downward velocity) also has a monotone
underlying progress variable, which makes a learned value function easy to sanity
check.

Backup: `water_plant` — slightly longer but it *terminates early on failure*
(trigger pulled outside the plant region), so negatives are crisp and cheap.

## Environments in one paragraph

`dexjoco.tasks.CONFIG_MAPPING[name]().get_environment(policy_mode=True,
render_mode="rgb_array", randomize=..., seed=..., randomize_dynamics=...)`
returns a gym env whose action for single-arm tasks is
`(23,) = [xyz(3), quat_wxyz(4), allegro_targets(16)]` — an **absolute** mocap
target consumed by an opspace controller, not a delta. Observations are
`{"state": (38,), "<cam>": HxWx3 uint8, ...}`; only `state[:23] =
[tcp_pose(7), allegro_qpos(16)]` is proprioception, the rest is privileged
(object poses, table height). Reward is 1.0 on success, `info["succeed"]` carries
the flag, and the env terminates on success or at its step cap (1000 for
hammer_nail). Two gotchas for rollouts: `step()` sleeps to throttle to `hz=30`,
and `_compute_observation` renders **all four** cameras (front, ego_left,
ego_right, wrist) every step even though a single-arm policy needs two.

Headless rendering: `MUJOCO_GL=egl` needs a GPU (it fails on the CPU-only login
node with `EGLError`), `MUJOCO_GL=osmesa` works everywhere but is software and
slow, `glfw` core-dumps without a display. So: `osmesa` for CPU-side smoke tests,
`egl` on the GPU node that runs rollouts.

## Data pipeline

```
DexJoCo/DexJoCo-Datasets-Raw          download_raw.py
  dexjoco_raw_datasets/<task>/<demo>/ ------------------> /workspace/junmo_cho/dexjoco/raw/
    replay.zarr/data/{action,action_rotvec,state,timestamp}
    videos/{front,wrist,ego_left,ego_right}.mp4
                                      convert_raw_to_rldx.py
                                      ------------------> /workspace/junmo_cho/dexjoco/lerobot/<task>_rand_obj/
                                                            data/chunk-000/episode_%06d.parquet
                                                            videos/chunk-000/observation.images.camera_{front,wrist}/episode_%06d.mp4
                                                            meta/{info,modality}.json meta/{episodes,tasks}.jsonl meta/stats.json
```

Two things the upstream tooling gets wrong for our purposes, both handled by
`convert_raw_to_rldx.py`:

**1. `DexJoCo-Datasets-LeRobot` is LeRobot v3.0.** One aggregated
`data/chunk-000/file-000.parquet` for all 100 episodes, videos concatenated
across episodes, `meta/episodes/*.parquet` + `meta/tasks.parquet`. RLDX-1's
`LeRobotEpisodeLoader` wants v2.1: per-episode parquet and mp4 addressed by
`info.json`'s `data_path` / `video_path` patterns, plus `meta/episodes.jsonl`,
`meta/tasks.jsonl` and a GR00T-style `meta/modality.json`. The raw repo is one
directory per episode, so converting from raw is both easier and lossless.

**2. `action_rotvec` — the field dexjoco's own converter trains on — is
discontinuous.** The DexJoCo end-effector works near a 180-degree rotation
(state quat w ~ 0), which puts the rotation vector exactly on the +-pi antipode:

| | frames | steps jumping > 3 rad | episodes affected |
| --- | --- | --- | --- |
| hammer_nail | 21571 | 216 (1.01%), up to a full 2*pi flip | 66 / 100 |
| water_plant | 27745 | 212 (0.77%) | 66 / 100 |

The underlying quaternion is smooth across those same frames (max step 0.023),
so the discontinuity is purely representational — but a 32-step *absolute*
action chunk that straddles a flip is an impossible regression target, and it
happens two to three times per episode. Quaternions are not the fix either: the
q / -q double cover means per-episode sign canonicalisation is needed, and
canonicalisation is only consistent *within* an episode, so two episodes can put
the same visual state on opposite branches. So the converter stores **rot6d**
(the first two rows of the rotation matrix, flattened — the convention of
`EndEffectorPose._matrix_to_rot6d` in RLDX-1), which is continuous *and* unique.

Resulting layout, symmetric between state and action (25 dims each):

```
eef_position [ 0: 3]   xyz, world frame
eef_rotation [ 3: 9]   rot6d
hand_joints  [ 9:25]   Allegro ffj0-3 / mfj0-3 / rfj0-3 / thj0-3
```

state is the *measured* flange pose + finger qpos; action is the *commanded*
target. Well under `RLDXConfig.max_state_dim / max_action_dim` (64), so none of
the padding hazards of the 66-DOF rby1m_wujihand2 case apply.

Smaller fixes the converter also applies:

* Leading static frames are trimmed with the same rule as dexjoco's converter
  (`find_first_non_static_frame`), then **one more frame is dropped**: the first
  recorded action of an episode is the teleop's jump from the home pose (up to
  1.1 m in one 1/30 s step) and in 8 of the 100 hammer_nail demos it is exactly
  zero, which the env reads as "hold" rather than as a target and which has no
  valid quaternion.
* Videos are re-encoded to 256x256. That is exactly the processor's default
  `image_max_area = 65536` and a multiple of `image_resize_m = 32`, so
  `AspectAreaResizeAndCrop` becomes a no-op and no information is thrown away
  twice.
* Only `front` + `wrist` are converted. The raw demos also ship an
  ego_left/ego_right stereo pair, but the *policy-mode* env for several tasks
  (e.g. water_plant) renders only wrist + front, so a checkpoint trained on the
  ego views could not be served.
* `next.success` / `next.done` (True on the last frame; every raw demo is a
  successful take) and `next.truncated` (always False) are written, so the demo
  set is directly consumable by `rl/data.py` as expert data for the offline
  critic — not just by BC.

## Commands

```bash
SIM=/workspace/junmo_cho/dexjoco/venv/bin/python        # zarr / mujoco / imageio
RLDX=third_party/RLDX-1/.venv/bin/python                # torch / rldx

# 1. download (CPU, ~250 MB per task)
$RLDX sim/dexjoco/download_raw.py hammer_nail

# 2. convert (CPU, re-encodes 2 x 100 videos)
$SIM sim/dexjoco/convert_raw_to_rldx.py \
    --input  /workspace/junmo_cho/dexjoco/raw/dexjoco_raw_datasets/hammer_nail \
    --output /workspace/junmo_cho/dexjoco/lerobot/hammer_nail_rand_obj \
    --task hammer_nail

# 3. normalisation stats (training would do this itself; doing it here fails fast)
PYTHONPATH=third_party/RLDX-1 $RLDX -c "from rldx.data.stats import generate_stats; \
    generate_stats('/workspace/junmo_cho/dexjoco/lerobot/hammer_nail_rand_obj')"

# 4. BC finetune (8 GPUs; needs HF_TOKEN for the private RLDX-1-PT-IMG)
sbatch sbatch/dexjoco/bc_hammer_nail.sbatch

# format contract check — no download, no GPU, no sim venv needed
PYTHONPATH=third_party/RLDX-1:. NO_ALBUMENTATIONS_UPDATE=1 $RLDX sim/dexjoco/test_format.py
```

`test_format.py` writes a synthetic dataset through the same `write_episode` /
`write_meta` / `_features` / `_modality` helpers the converter uses, then loads it
with `LeRobotEpisodeLoader` and `generate_stats`, so any schema drift fails there
rather than 20 minutes into a training job.

## Where things live (and why not on /rlwrld2)

`/rlwrld2` is at 100% with 68 GB free — `checkpoints/` alone is 117 GB and a
single RLDX-1 checkpoint is ~13 GB. So:

| what | where |
| --- | --- |
| raw + converted datasets | `/workspace/junmo_cho/dexjoco/` |
| sim venv | `/workspace/junmo_cho/dexjoco/venv` |
| uv cache for that venv | `/workspace/junmo_cho/uv_cache` |
| HF cache for training | `/workspace/junmo_cho/hf_cache` |
| BC checkpoints | `/rlwrld-unified-checkpoints/junmo_cho/dexjoco/` |

## Remaining stages (not built yet)

The real-robot loop is `rl/vla_rldx.py serve` (RLDX-1 + critic-guided chunk
selection, speaking RLDX-1's ZeroMQ + msgpack protocol from
`rldx/policy/server_client.py`) with `rrc` on the robot side. The sim version
replaces only the client:

1. **`rollout_dexjoco.py`** (sim venv) — steps the dexjoco env, sends
   `video.camera_front` / `video.camera_wrist` (uint8 `(B,T,H,W,C)`) and
   `state.eef_position` / `state.eef_rotation` / `state.hand_joints` (float32
   `(B,T,D)`) plus the task string to the policy server, converts the returned
   rot6d back to a quaternion for `SingleArmPolicyWrapper`, and writes each
   episode as a LeRobot v2.1 session dir with real `next.success` / `next.done`.
   The protocol client needs only `zmq + msgpack + numpy`, no RLDX imports.
   Note `ExpoServer` defaults to `img_size=(320,192)`; sim needs `256 256`.
2. **BC success rate** = fraction of rollout episodes with `next.success`.
3. **`rl/extract_cogfeat.py` -> `rl/offline_iql.py`** on that rollout parent
   directory, unchanged, driven by a new `configs/exp/dexjoco_hammer_nail.yaml`
   (`rldx_data_config: rldx/configs/data/dexjoco_panda_allegro_config.py`,
   `modality: modality/dexjoco_panda_allegro/modality.json`, `base_policy`,
   `action_horizon: 32`, `replan_steps`, `inference_latency`, `explore_groups`;
   the code asserts `action_horizon >= inference_latency + replan_steps`).
4. **Guidance eval** — same rollout client against a server started with the
   critic artifacts, compared against the un-guided BC number from step 2.
