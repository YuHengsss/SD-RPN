export HF_ENDPOINT=https://hf-mirror.com # for China users
python make_pseudo_label_textual.py \
    --model-path 'Qwen/Qwen2.5-VL-7B-Instruct' \
    --question-file '../datasets/llava_v1_5_mix665k_selected_qa.jsonl' \
    --answers-file '../datasets/qwen2_5vl_pseudo_7b_ocr_vqa_fixres.pkl' \
    --cur-dataset 'ocr_vqa' \
    --gpu-ids '0,1,2,3'