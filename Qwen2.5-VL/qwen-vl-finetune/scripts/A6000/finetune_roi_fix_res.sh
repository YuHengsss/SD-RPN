#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=/home/yuheng/code/Qwen2.5-VL/qwen-vl-finetune/scripts/zero2.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-4
batch_size=16
grad_accum_steps=2
twig_K=21
twig_T=3
# Training entry point
entry_file=qwen-vl-finetune/qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=my_roi_dataset

# Output configuration
run_name="qwen2_5vl-3b-roi-K${twig_K}T${twig_T}-152k-v1bf16Mheads-twiginit"
output_dir=./output/${run_name}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 451584 \
    --min_pixels 451584 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard \
    --enable_twig True \
    --twig_K ${twig_K} \
    --twig_T ${twig_T} \
    --roi_branch True \
    --twig_freeze 0 \
    --roi_source "qk" \
    --roi_loss "bce" \
    --roi_super_type "v1" \
    --roi_multi_head True \
    --twig_init True \
    --roi_data_path /home/yuheng/datasets/qwen2_5vl_pseudo_3b_576res_release.pkl \
    --fix_res True
    --multi_scale_training False \
    --roi_samples -1
    "

# Launch training
torchrun --nproc_per_node=4 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}


