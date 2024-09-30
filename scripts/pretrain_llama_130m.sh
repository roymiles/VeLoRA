#!/bin/sh
echo "start"
torchrun --standalone --nproc_per_node 4 pretrain_llama.py \
    --model_config configs/llama_130m.json \
    --lr 0.003 \
    --velora_r 1 \
    --velora_layers 'vd' \
    --num_groups 64,64 \
    --init_type batch_average_once \
    --peft_type velora+full \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype float16 \
    --eval_every 1000 \
    --optimizer velora \
    --velora_scale 1.0