export HF_ENDPOINT=https://hf-mirror.com # for China users
python make_pseudo_label_textual.py \
    --model-path '../site-packages/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/' \
    --question-file '/home/yuheng/datasets/llava_v1_5_mix665k_selected_qa.jsonl' \
    --answers-file '/home/yuheng/datasets/qwen2_5vl_pseudo_7b_gqa_fixres.pkl' \
    --cur-dataset 'gqa' \
    --gpu-ids '2,3'