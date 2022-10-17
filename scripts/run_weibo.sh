#!/bin/bash

# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export CUDA_VISIBLE_DEVICES=2
dataset='weibo'

for k in '5'
do
    for group in '0' '1' '2' '3' '4'
    do

        python run_ner.py \
            --do_train \
            --do_eval \
            --types_path datasets/$dataset/types.json \
            --train_path datasets/$dataset/few-shot/$((k))_shot_train_$((group)).json \
            --valid_path datasets/$dataset/few-shot/$((k))_shot_dev_$((group)).json \
            --test_path datasets/$dataset/test.json \
            --plm_path bert-base-chinese \
            --tokenizer_path bert-base-chinese \
            --lowercase \
            --sampling_processes 4 \
            --save_path outputs/$dataset/$((k))_shot_$((group)) \
            --se_train_batch_size 1 \
            --entail_train_batch_size 4 \
            --se_eval_batch_size 8 \
            --entail_eval_batch_size 4 \
            --epochs 35 \
            --lr 3e-5 \
            --lr_warmup 0.1 \
            --weight_decay 0.01 \
            --max_grad_norm 1 \
            --seed 47 \
            --seed_threshold 0.7

    done
done