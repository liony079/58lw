#!/bin/sh

tmux new-session -d -s tensorboard
tmux send-keys 'conda activate pytorch3.6' 'C-m'
tmux send-keys 'tensorboard --bind_all --reload_multifile True --logdir=/data/amax/output_sEMG-reconstructure/1 --port=6007' 'C-m'
tmux new-session -d -s watch
tmux send-keys 'watch -n 1 nvidia-smi' 'C-m'

#tmux new-session -d
#tmux send-keys 'conda activate pytorch3.6' 'C-m'
#tmux send-keys 'python main.py' 'C-m'

sub=(2 3 4 5 6 7 8) # 2 3 4 5 6 7 8
for i in "${!sub[@]}"
do
    tmux new-session -d -s sub${sub[$i]}
    tmux send-keys 'conda activate pytorch3.6' 'C-m'
    tmux send-keys 'python main.py --subject '${sub[$i]}' --device '$(($i% 4 ))'' 'C-m' #$(( $a % 5 ))
done