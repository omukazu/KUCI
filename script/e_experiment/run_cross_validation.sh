#!/usr/local/bin/zsh

while getopts c:n:p:g:s: OPT
do
  case $OPT in
    c) CONFIG="$OPTARG" ;;
    n) NPROC_PER_NODE="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    g) GPU="$OPTARG" ;;
    s) SEED="$OPTARG" ;;
    *) echo "invalid option";;
  esac
done

HOST=$(hostname)
nice -n 10 torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" --rdzv_endpoint="$HOST":"$PORT" train.py "$CONFIG" --fold 1 --seed "$SEED" --gpu "$GPU"
nice -n 10 torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" --rdzv_endpoint="$HOST":"$PORT" train.py "$CONFIG" --fold 2 --seed "$SEED" --gpu "$GPU"
nice -n 10 torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" --rdzv_endpoint="$HOST":"$PORT" train.py "$CONFIG" --fold 3 --seed "$SEED" --gpu "$GPU"
nice -n 10 torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" --rdzv_endpoint="$HOST":"$PORT" train.py "$CONFIG" --fold 4 --seed "$SEED" --gpu "$GPU"
nice -n 10 torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" --rdzv_endpoint="$HOST":"$PORT" train.py "$CONFIG" --fold 5 --seed "$SEED" --gpu "$GPU"