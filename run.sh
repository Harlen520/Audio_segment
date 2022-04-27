#!/bin/bash

unset CUDA_VISIBLE_DEVICES

python -u -m paddle.distributed.launch --gpus "0,1" run.py \
       --train_set train.txt \
       --dev_set dev.txt \
       --device gpu \
       --eval_step 100 \
       --save_dir ./checkpoints \
       --train_batch_size 32 \
       --learning_rate 2E-5 \
       --rdrop_coef 0.0
