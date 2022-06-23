#!/bin/bash

unset CUDA_VISIBLE_DEVICES
#--init_from_ckpt /home/th/paddle/audio_segment/checkpoint/auc-binary/model_9400/model_state.pdparams  \
python run_vauc.py \
       --init_from_ckpt /home/th/paddle/audio_segment/checkpoint/auc-binary/model_9400/model_state.pdparams  \
       --learning_rate 1e-4 \
       --loss vauc-binary  \
       --train_batch_size 64 \
       --gpuids 1 \
       --step 1
