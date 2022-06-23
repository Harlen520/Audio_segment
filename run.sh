#!/bin/bash

python run.py \
       --init_from_ckpt /home/th/paddle/audio_segment/checkpoint/1/vauc-binary/vauc_model_1000/model_state.pdparams \
       --step 1 \
       --learning_rate 1e-4
