#/bin/bash
CUDA_VISIBLE_DEVICE=5 ./train.py \
-cu 5 \
-a ../raw_data/itny/itny_new_10 \
-tb 128 \
-pb 32  \
-r 10 \
-k 5 \
-n True \
-f gru \
-p False