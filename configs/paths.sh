# --- actor (robot desktop) ---
A_RL=/home/openarm14/ws/junmo_cho/rd-rl
A_CKPT=/home/openarm14/ws/junmo_cho/checkpoints
A_DS=/home/openarm14/ws/junmo_cho/rd-rl/rl-dataset
A_RUNS=/home/openarm14/ws/junmo_cho/rd-rl/runs

# --- learner (training server) ---
L_RL=/data/junmo_cho/workspace/rd-rl
L_CKPT=/data/rlwrld-unified-checkpoints/junmo_cho/checkpoints
L_DS=/data/junmo_cho/workspace/rd-rl/rl-dataset
L_RUNS=/data/junmo_cho/workspace/rd-rl/runs

# --- k8s ---
L_NS=p-rlwrld
L_POD=junmo-cho-data-pod

# --- base policy (BC/SFT) ---
# 0814-openarm-rh56f1-rldx-ptimg/openarm_0814_rh56f1_teleop_all200ep_egostereo_ptimg_framewt_drop03_rtc12tr_bs128_30k_4gpu_mlxp
RDRL_BASE_POLICY=0814-openarm-rh56f1-rldx-ptimg/openarm_0814_rh56f1_teleop_all200ep_egostereo_ptimg_framewt_drop03_rtc12tr_bs128_30k_4gpu_mlxp
