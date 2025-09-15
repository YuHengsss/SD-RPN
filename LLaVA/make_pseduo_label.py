# debug_llava.py
import os
import sys
import subprocess
from llava.eval.model_vqa_label import eval_model

# Set environment variables
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/yuheng/site-packages'#'/data/autodl-tmp/site-packages'

#textvqa
sys.argv = [
    'llava.eval.model_vqa_analysis',
    '--model-path', 'liuhaotian/llava-v1.5-13b',
    '--source-file', '/home/yuheng/datasets/llava_v1_5_mix665k.json',
    '--question-file','/home/yuheng/datasets/llava_v1_5_mix665k_selected_qa.jsonl',
    '--answers-file', '/home/yuheng/datasets/llava_v1_5_pseudo_13b_gqa.pkl',
    '--data-path', '/home/yuheng/datasets',
    '--temperature', '0',
    '--conv-mode', 'vicuna_v1',
    '--cur-dataset', 'ocr_vqa', #gqa, textvqa, ocr_vqa, VG_100K, coco
]


# Import the argument parser
import argparse

# Run the CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cur-dataset", type=str, default="textvqa")  # gqa, textvqa, ocr_vqa, VG_100K
    args = parser.parse_args()

    eval_model(args)
