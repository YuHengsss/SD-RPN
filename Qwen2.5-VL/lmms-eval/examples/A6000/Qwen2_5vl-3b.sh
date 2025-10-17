#!/bin/bash

export HUGGINGFACE_HUB_CACHE=/home/yuheng/site-packages

#
## Construct common path components
checkpoint_path="Qwen/Qwen2.5-VL-3B-Instruct"
log_suffix_base="Qwen2.5-VL-3B"
min_pixels=451584 #576*28*28
max_pixels=451584 #576*28*28
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=False,roi_baseline=False,min_pixels=${min_pixels},max_pixels=${max_pixels}"
#
###textvqa_val
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks textvqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_textvqa_val --output_path ./logs/

# infovqa_val
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks infovqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_infovqa_val --output_path ./logs/

# chartqa_val
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks chartqa --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_chartqa --output_path ./logs/

# ocrbench
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks ocrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_ocrbench --output_path ./logs/

# docvqa
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks docvqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_docvqa_val --output_path ./logs/

###vstar_bench
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks vstar_bench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_vstar_bench --output_path ./logs/

# pope
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks pope --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_pope --output_path ./logs/

##hrbench
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4  -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks hrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_hrbench --output_path ./logs/


ROI_T=3
ROI_K=21
#
## Construct common path components
checkpoint_path="./output/qwen2_5vl-3b-roi-K21T3-152k-v1bf16Mheads-twiginit-filled"
log_suffix_base="qwen2_5vl_3b_K${ROI_K}T${ROI_T}-min_pixels${min_pixels}-v1bf16Mheads-twiginit"

##textvqa_val
roi_conf_thresh=0.15
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks textvqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_textvqa_val --output_path ./logs/

# infovqa_val
roi_conf_thresh=0.1
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks infovqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_infovqa_val --output_path ./logs/

# chartqa_val
roi_conf_thresh=0.1
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks chartqa --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_chartqa --output_path ./logs/

###vstar_bench
roi_conf_thresh=0.175
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks vstar_bench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_vstar_bench --output_path ./logs/

# ocrbench
roi_conf_thresh=0.03
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks ocrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_ocrbench --output_path ./logs/

# docvqa
roi_conf_thresh=0.1
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks docvqa_val --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_docvqa_val --output_path ./logs/


# pope
roi_conf_thresh=0.1
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks pope --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_pope --output_path ./logs/

##hrbench
roi_conf_thresh=0.1
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks hrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_hrbench --output_path ./logs/


#####highres setting for vstar_bench and hrbench
##vstar_bench
min_pixels=802816 #576*28*28
max_pixels=3211264 #4096*28*28
checkpoint_path="Qwen/Qwen2.5-VL-3B-Instruct"
log_suffix_base="Qwen2.5-VL-3B"
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=False,roi_baseline=False,min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks vstar_bench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_vstar_bench --output_path ./logs/

roi_conf_thresh=0.075
checkpoint_path="./output/qwen2_5vl-3b-roi-K${ROI_K}T${ROI_T}-152k-v1bf16Mheads-twiginit-filled"
log_suffix_base="qwen2_5vl_3b_K${ROI_K}T${ROI_T}-min_pixels${min_pixels}-v1bf16Mheads-twiginit"
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks vstar_bench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_vstar_bench --output_path ./logs/




##hrbench
min_pixels=802816 #576*28*28
max_pixels=2408448 #4096*28*28
checkpoint_path="Qwen/Qwen2.5-VL-3B-Instruct"
log_suffix_base="Qwen2.5-VL-3B"
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=False,roi_baseline=False,min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks hrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_vstar_bench --output_path ./logs/

roi_conf_thresh=0.125
checkpoint_path="./output/qwen2_5vl-3b-roi-K${ROI_K}T${ROI_T}-152k-v1bf16Mheads-twiginit-filled"
log_suffix_base="qwen2_5vl_3b_K${ROI_K}T${ROI_T}-min_pixels${min_pixels}-v1bf16Mheads-twiginit"
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks hrbench4k --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_vstar_bench --output_path ./logs/


##hrbench
min_pixels=802816 #576*28*28
max_pixels=3211264 #4096*28*28
checkpoint_path="Qwen/Qwen2.5-VL-3B-Instruct"
log_suffix_base="Qwen2.5-VL-3B"
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=False,roi_baseline=False,min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks hrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_vstar_bench --output_path ./logs/

roi_conf_thresh=0.05
checkpoint_path="./output/qwen2_5vl-3b-roi-K${ROI_K}T${ROI_T}-152k-v1bf16Mheads-twiginit-filled"
log_suffix_base="qwen2_5vl_3b_K${ROI_K}T${ROI_T}-min_pixels${min_pixels}-v1bf16Mheads-twiginit"
model_args="pretrained=${checkpoint_path},device_map=auto,two_stage_roi=True,roi_conf_thresh=${roi_conf_thresh},min_pixels=${min_pixels},max_pixels=${max_pixels}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval --model qwen2_5_vl --model_args "${model_args}" --tasks hrbench --batch_size 1 --log_samples --log_samples_suffix ${log_suffix_base}_vstar_bench --output_path ./logs/

