tmux new -s training
srun -p gpu_a100 -t 1:00:00 --gpus=1 --pty /bin/bash
tmux attach -t training
