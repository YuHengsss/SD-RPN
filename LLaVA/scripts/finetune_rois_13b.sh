#!/bin/bash

source /home/yuheng/miniconda3/etc/profile.d/conda.sh
conda activate llava_roi

export HUGGINGFACE_HUB_CACHE=/home/yuheng/site-packages

#export NCCL_DEBUG=INFO  # To see detailed NCCL logs
#export NCCL_IB_DISABLE=1  # Try this if you still have issues
#export NCCL_P2P_DISABLE=1  # Try this if you still have issues
DATASET_PATH=/home/yuheng/datasets

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2_clip.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path $DATASET_PATH/llava_v1_5_13b_pseudo_roi_release.pkl \
    --image_folder $DATASET_PATH\
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --fp16 False \
    --output_dir ./checkpoints/llava-v1.5-13b-roi-K15T3-152k-v1bf16Mheads-twiginit \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --enable_twig True \
    --twig_K 15 \
    --twig_T 3 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --roi_branch True \
    --twig_freeze 0 \
    --roi_source 'qk' \
    --roi_loss 'bce' \
    --roi_super_type 'v1' \
    --roi_multi_head True \
    --twig_init True
