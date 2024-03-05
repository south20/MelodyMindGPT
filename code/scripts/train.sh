#!/bin/bash

deepspeed --include localhost:6 --master_addr 127.0.0.1 --master_port 28459 train.py \
    --model southgpt \
    --stage 1\
    --save_path  ../ckpt/delta_ckpt/southgpt/7b_tiva_v0\
    --log_path ../ckpt/delta_ckpt/southgpt/7b_tiva_v0/log

