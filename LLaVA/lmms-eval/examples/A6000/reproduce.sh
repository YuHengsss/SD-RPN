#!/bin/bash


##Construct common path components
checkpoint_path="liuhaotian/llava-v1.5-7b"
log_suffix_base="llava15_7b"
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=False,roi_baseline=False"

###textvqa_val
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks textvqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_textvqa_val --output_path ./logs/
#
## infovqa_val
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks infovqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_infovqa_val --output_path ./logs/
#
## chartqa_val
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks chartqa --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_chartqa --output_path ./logs/
#
## ocrbench
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks ocrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_ocrbench --output_path ./logs/
#
## docvqa
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks docvqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_docvqa_val --output_path ./logs/

## vstar_bench
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks vstar_bench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_vstar_bench --output_path ./logs/

## pope
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks pope --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_pope --output_path ./logs/
#
##hrbench
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks hrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_hrbench --output_path ./logs/



ROI_T=3
ROI_K=15

# Construct common path components
checkpoint_path="./checkpoints/llava-v1.5-7b-roi-K${ROI_K}T${ROI_T}-152k-v1bf16Mheads-twiginit-filled"
log_suffix_base="llava15_7b_K${ROI_K}T${ROI_T}v1bf16MheadsTwiginit-maskUpScale-A6000"
roi2stage_max_ratio=-1.0

##textvqa_val
#roi_conf_thresh=0.15
#model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},roi2stage_max_ratio=${roi2stage_max_ratio},roi2stage_max_ratio=${roi2stage_max_ratio}"
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks textvqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_textvqa_val --output_path ./logs/
#
## infovqa_val
#roi_conf_thresh=0.15
#model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},roi2stage_max_ratio=${roi2stage_max_ratio}"
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks infovqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_infovqa_val --output_path ./logs/
#
## chartqa_val
#roi_conf_thresh=0.15
#model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},roi2stage_max_ratio=${roi2stage_max_ratio}"
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks chartqa --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_chartqa --output_path ./logs/
#
## ocrbench
#roi_conf_thresh=0.05
#model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},roi2stage_max_ratio=${roi2stage_max_ratio}"
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks ocrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_ocrbench --output_path ./logs/
#
## docvqa
#roi_conf_thresh=0.1
#model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},roi2stage_max_ratio=${roi2stage_max_ratio}"
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks docvqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_docvqa_val --output_path ./logs/

# vstar_bench
#roi_conf_thresh=0.1
#model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},roi2stage_max_ratio=${roi2stage_max_ratio}"
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks vstar_bench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_vstar_bench --output_path ./logs/

## pope
#roi_conf_thresh=0.15
#model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},roi2stage_max_ratio=${roi2stage_max_ratio}"
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks pope --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_pope --output_path ./logs/
#
##hrbench
#roi_conf_thresh=0.1
#model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},roi2stage_max_ratio=${roi2stage_max_ratio}"
#CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args "${model_args}" --tasks hrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_hrbench --output_path ./logs/

