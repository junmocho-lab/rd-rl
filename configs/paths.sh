# --- actor (robot desktop) ---
A_RL=/home/rd/ws/junmo_cho/rd-rl
A_CKPT=/home/rd/ws/junmo_cho/checkpoints
A_DS=/home/rd/ws/junmo_cho/rd-rl/rl-dataset
A_RUNS=/home/rd/ws/junmo_cho/rd-rl/runs

# --- learner (training server) ---
L_RL=/data/junmo_cho/workspace/rd-rl
# 산출물($L_CKPT/expo/<run id>/)은 junmo_cho 쪽에 쌓는다. base 모델 원본은
# younghoon_shin 쪽에 있고, 여기 rldx-img-curated/ 아래 심볼릭 링크로 연결돼 있다.
L_CKPT=/data/rlwrld-unified-checkpoints/junmo_cho/checkpoints
L_DS=/data/junmo_cho/workspace/rd-rl/rl-dataset
L_RUNS=/data/junmo_cho/workspace/rd-rl/runs
# 학습서버에서 쓸 인터프리터. DDN 의 rd-rl/.venv 는 파드 안에서 uv sync 로 만들어져
# bin/python 이 /root/.local/... (컨테이너 로컬)을 가리키므로 fresh 파드에서 못 쓴다.
#
# 그래서 RLDX-1 학습 잡이 쓰는 venv 를 그대로 쓴다 — bin/python 이 /usr/bin/python3
# 심볼릭 링크라 fresh 파드에서도 살아있고, RLDX-1 이 요구하는 파이썬·패키지가 다 있다:
#   python 3.10.12 / torch 2.7.0+cu126 / peft 0.17.1 / torchcodec 0.4.0 / transformers 4.57.0
# (참고: actor 로컬 pixi rldx 는 torch 2.8.0+cu128 이다. 버전이 달라 같은 seed 로도
#  θ₀ 가 비트 단위로 같지 않을 수 있다 — 그래서 θ₀ 는 재구성하지 않고 파일로 옮긴다.)
# 이미지의 /opt/conda/bin/python3 는 3.13.12 이고 torch 가 없다. 쓰지 말 것.
L_PY=/data/junmo_cho/workspace/RLDX-1/.venv/bin/python

# rldx 를 **우리가 pin 한 서브모듈**에서 import 하게 만든다. 이게 없으면 위 venv 에
# 설치된 다른 체크아웃(/data/junmo_cho/workspace/RLDX-1)이 잡혀서, 라운드 기록에는
# 우리 SHA 가 남는데 실제로는 다른 코드가 도는 상황이 된다.
L_PYTHONPATH=$L_RL/third_party/RLDX-1:$L_RL

# --- k8s ---
L_NS=p-rlwrld
L_POD=younghoon-shin-rldx-img-0827-0903-4gpu-46ndb-gpvp5

# base policy 는 여기 없다 — 실험마다 다르므로 configs/exp/<이름>.yaml 에 둔다
# (paths.sh 는 이 머신의 사실만 담는다).
