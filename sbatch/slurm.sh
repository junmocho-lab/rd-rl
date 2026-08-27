srun --gpus=1 --nodes=1 --wckey=project-short-name:rd --pty bash

squeue --me
srun --jobid=121246 --overlap --pty bash
tail -f out/critic_121xxx.out  # 로그 실시간 보기

MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/junmo_cho/ sbatch ./sbatch/offline_critic_openarm.sbatch

scancel 14