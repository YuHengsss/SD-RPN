# install lmms_eval without building dependencies
cd lmms_eval;
pip install --no-deps -U -e .

# install LLaVA without building dependencies
cd LLaVA
pip install --no-deps -U -e .

# install all the requirements that require for reproduce llava results
pip install -r llava_repr_requirements.txt

# Run and exactly reproduce llava_v1.5 results!
# mme as an example
accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b,use_flash_attention_2=False,device_map=auto"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/

#mme
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=./checkpoints/llava-v1.5-7b-roi-K12T3-v1bf16Mheads-filled,device_map=auto,two_stage_roi=True"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix K12T3v1bf16Mheads --output_path ./logs/

#docvqa
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=./checkpoints/llava-v1.5-7b-roi-K12T3-v1bf16Mheads-filled,device_map=auto,two_stage_roi=True"   --tasks docvqa_val  --batch_size 1 --log_samples --log_samples_suffix K12T3v1bf16Mheads --output_path ./logs/
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=liuhaotian/llava-v1.5-7b,device_map=auto"   --tasks docvqa_val  --batch_size 1 --log_samples --log_samples_suffix llava_baseline --output_path ./logs/



## LLaVA-1.5-7b-roi-K12T3-v1bf16Mheads

#docvqa
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=./checkpoints/llava-v1.5-7b-roi-K12T3-v1bf16Mheads-filled,device_map=auto,two_stage_roi=True"   --tasks docvqa_val  --batch_size 1 --log_samples --log_samples_suffix llava15_7b_K12T3v1bf16Mheads_docvqa --output_path ./logs/

##ChartQA
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=./checkpoints/llava-v1.5-7b-roi-K12T3-v1bf16Mheads-filled,device_map=auto,two_stage_roi=True"   --tasks chartqa  --batch_size 1 --log_samples --log_samples_suffix llava15_7b_K12T3v1bf16Mheads_chartqa --output_path ./logs/

###MME-RealWorld
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=./checkpoints/llava-v1.5-7b-roi-K12T3-v1bf16Mheads-filled,device_map=auto,two_stage_roi=True"   --tasks mmerealworld  --batch_size 1 --log_samples --log_samples_suffix llava15_7b_K12T3v1bf16Mheads_mmeRealWorld --output_path ./logs/

#MMStar
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=./checkpoints/llava-v1.5-7b-roi-K12T3-v1bf16Mheads-filled,device_map=auto,two_stage_roi=True"   --tasks mmstar  --batch_size 1 --log_samples --log_samples_suffix llava15_7b_K12T3v1bf16Mheads_mmstar --output_path ./logs/

#ocrbench
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=./checkpoints/llava-v1.5-7b-roi-K12T3-v1bf16Mheads-filled,device_map=auto,two_stage_roi=True"   --tasks ocrbench  --batch_size 1 --log_samples --log_samples_suffix llava15_7b_K12T3v1bf16Mheads_ocrbench --output_path ./logs/

#infovqa_val
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=./checkpoints/llava-v1.5-7b-roi-K12T3-v1bf16Mheads-filled,device_map=auto,two_stage_roi=True"   --tasks infovqa_val  --batch_size 1 --log_samples --log_samples_suffix llava15_7b_K12T3v1bf16Mheads_infovqa_val --output_path ./logs/

#gqa
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=./checkpoints/llava-v1.5-7b-roi-K12T3-v1bf16Mheads-filled,device_map=auto,two_stage_roi=True"   --tasks gqa  --batch_size 1 --log_samples --log_samples_suffix llava15_7b_K12T3v1bf16Mheads_gqa --output_path ./logs/

#ai2d
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args "pretrained=./checkpoints/llava-v1.5-7b-roi-K12T3-v1bf16Mheads-filled,device_map=auto,two_stage_roi=True"   --tasks ai2d  --batch_size 1 --log_samples --log_samples_suffix llava15_7b_K12T3v1bf16Mheads_ai2d --output_path ./logs/

