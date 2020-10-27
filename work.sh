#!/usr/bin/env bash
TASK_NAME="rest15"
ABSA_HOME="./bert-linear-rest15-finetune"
CUDA_VISIBLE_DEVICES=0 python work.py --absa_home ${ABSA_HOME} \
                      --ckpt ${ABSA_HOME}/checkpoint-1500 \
                      --model_type bert \
                      --data_dir ./data/${TASK_NAME} \
                      --task_name ${TASK_NAME} \
                      --model_name_or_path bert-base-uncased \
                      --cache_dir ./model_cache \
                      --max_seq_length 128 \
                      --tagging_schema BIEOS