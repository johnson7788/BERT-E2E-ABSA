#!/usr/bin/env bash
python main.py --model_type bert --absa_type gru --tfm_mode finetune --fix_tfm 0 --model_name_or_path bert-base-chinese --data_dir ./data/cosmetics --task_name cosmetics --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 8 --learning_rate 2e-5 --max_steps 1500 --warmup_steps 0 --do_train --do_eval --do_lower_case --seed 42 --tagging_schema BIEOS --overfit 0 --overwrite_output_dir --eval_all_checkpoints --MASTER_ADDR localhost --MASTER_PORT 28