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
# 학습서버에서 쓸 인터프리터. DDN 의 rd-rl/.venv 는 파드 안에서 uv sync 로 만들어져
# bin/python 이 /root/.local/... (컨테이너 로컬)을 가리키므로 fresh 파드에서 못 쓴다.
# (RLDX-1/.venv 는 home 이 /usr/bin 이라 Job 에서 동작한다.)
# 실제 학습 단계에서는 RLDX-1 이 requires-python "==3.10.*" 이므로
# /usr/bin/python3 (3.10.12) 기준 venv 를 만들고 이 값만 그리로 바꾼다.
L_PY=/opt/conda/bin/python3

# --- k8s ---
L_NS=p-rlwrld
L_POD=junmo-cho-data-pod

# --- base policy (BC/SFT) ---
# 0814-openarm-rh56f1-rldx-ptimg/openarm_0814_rh56f1_teleop_all200ep_egostereo_ptimg_framewt_drop03_rtc12tr_bs128_30k_4gpu_mlxp
RDRL_BASE_POLICY=0814-openarm-rh56f1-rldx-ptimg/openarm_0814_rh56f1_teleop_all200ep_egostereo_ptimg_framewt_drop03_rtc12tr_bs128_30k_4gpu_mlxp
