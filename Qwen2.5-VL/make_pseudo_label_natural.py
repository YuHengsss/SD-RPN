import torch
import math
from PIL import Image
import numpy as np
import json
import os
import pickle
from tqdm import tqdm
import argparse
import multiprocessing as mp  # Import the multiprocessing library

from qwen_src.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_src.ana_utils import register_hooks, get_attn_weights_map
from qwen_src.mm_utils import expand2square


def worker_process(chunk_questions, gpu_id, args):
    """
    This function is executed by each parallel process on a single GPU.
    It loads its own model instance and processes its assigned chunk of data.
    """
    # The unpacking line is no longer needed, as the arguments are passed directly.
    device = f'cuda:{gpu_id}'

    print(f"[GPU {gpu_id}] Loading model to {device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left',
                                              use_fast=True)
    processor.image_processor.max_pixels = 576 * 28 * 28

    activations, _ = register_hooks(model)
    results = []

    # Use position for tqdm to avoid overlapping progress bars
    pbar = tqdm(chunk_questions, position=gpu_id, desc=f"GPU {gpu_id}")

    for line in pbar:
        # --- The core logic from your original function goes here ---
        activations.clear()
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        raw_qs = qs
        qs = qs.replace('Answer the question using a single word or phrase.',
                        'Output the grounding bounding box of Region of Interest for the question. IMPORTANT: The output MUST be raw text, one box per line. DO NOT use JSON. Follow this exact format: x_min y_min x_max y_max {detail_label}.')
        try:
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        except FileNotFoundError:
            print(
                f"[GPU {gpu_id}] Warning: Image not found at {os.path.join(args.image_folder, image_file)}. Skipping.")
            continue

        multi_scale_training = False
        scale_range = [0.33, 3.0]
        default_max_pixels = 576 * 28 * 28
        default_min_pixels = 576 * 28 * 28

        if multi_scale_training:
            scale = np.random.uniform(scale_range[0], scale_range[1])
            processor.image_processor.max_pixels = int(default_max_pixels * scale)
            processor.image_processor.min_pixels = int(default_min_pixels * scale)
        else:
            processor.image_processor.max_pixels = default_max_pixels
            processor.image_processor.min_pixels = default_min_pixels

        image = expand2square(image, (127, 127, 127))
        cur_prompt = f"{qs}"
        messages = [{"role": "user", "content": [{"type": "text", "text": "<image>\n" + cur_prompt},
                                                 {"image": os.path.join(args.image_folder, image_file)}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text.replace('<|vision_start|><|image_pad|><|vision_end|>', '').replace('<image>',
                                                                                       '<image><|vision_start|><|image_pad|><|vision_end|>')
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=128, output_attentions=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if activations.get('visual_mask', None) is None:
            vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
            pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
            pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
            activations['visual_mask'] = torch.zeros((1, inputs['input_ids'].shape[1]), dtype=torch.bool, device=device)
            activations['visual_mask'][0, pos:pos_end] = True
        output_shape = inputs['image_grid_thw'].cpu().numpy().squeeze(0)[1:] / 2
        output_shape = np.array(output_shape, dtype=np.int32)

        if "Qwen2.5-VL-3B" in args.model_path:
            required_heads = {21: [9, 11], 22: [0, 7]} # layer: [heads]  #            required_heads = {17: [10, 11], 20: [1,3,6], 22: [10, 14]} # layer: [heads]
            #required_heads = {27: [1, 10]}  # layer: [heads]
            sink_heads = {8: [1, 13]}
        elif "Qwen2.5-VL-7B" in args.model_path:
            required_heads = {16: [1, 7, 17], 19: [17, 20]}
            sink_heads = {9: [4, 22]}
        else:
            raise NotImplementedError()

        required_layers = list(required_heads.keys()) + list(sink_heads.keys())
        layers = max(required_layers) + 1 if required_layers else 0
        attn_weights_map = get_attn_weights_map(activations, layers=layers, required_layers=required_layers)

        grounding_attn_o2i = np.zeros(output_shape)
        total_grounding_heads = sum([len(heads) for heads in required_heads.values()])
        if total_grounding_heads > 0:
            for layer, heads in required_heads.items():
                attn_weights = attn_weights_map[f'layer{layer}']['output2images'].reshape(-1, output_shape[0], output_shape[1])
                for head in heads: grounding_attn_o2i += attn_weights[head]
            grounding_attn_o2i /= total_grounding_heads

        sink_attn = np.zeros(output_shape)
        total_sink_heads = sum([len(heads) for heads in sink_heads.values()])
        if total_sink_heads > 0:
            for layer, heads in sink_heads.items():
                attn_weights = attn_weights_map[f'layer{layer}']['output2images'].reshape(-1, output_shape[0], output_shape[1])
                for head in heads: sink_attn += attn_weights[head]
            sink_attn = sink_attn / total_sink_heads

        tmp_result = {"question_id": idx, "prompt": raw_qs, "text": output_text[0], "image": image_file,
                      "sink_attn": sink_attn, "grounding_attn_o2i": grounding_attn_o2i}
        results.append(tmp_result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-labels in parallel with a Qwen-VL model.")
    parser.add_argument('--model-path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='Path to the pretrained model.')
    parser.add_argument('--question-file', type=str, required=True,
                        help='Path to the input question file (.jsonl).')
    parser.add_argument('--image-folder', type=str, default='/',
                        help='Root directory for images.')
    parser.add_argument('--answers-file', type=str, required=True,
                        help='Path to save the final combined output pickle file.')
    parser.add_argument('--cur-dataset', type=str, required=True,
                        help='Substring to filter dataset (e.g., ocr_vqa).')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3").')

    args = parser.parse_args()

    # 1. Prepare Data and Split into Chunks
    gpu_ids = [int(gid) for gid in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    print("Loading and filtering questions...")
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    filtered_questions = [q for q in questions if args.cur_dataset in q["image"]]
    print(f"Found {len(filtered_questions)} questions for dataset '{args.cur_dataset}'.")

    chunk_size = math.ceil(len(filtered_questions) / num_gpus)
    chunks = [filtered_questions[i:i + chunk_size] for i in range(0, len(filtered_questions), chunk_size)]

    # Ensure we have a chunk for each GPU, even if some are empty
    while len(chunks) < num_gpus:
        chunks.append([])

    # 2. Set up arguments for each worker process
    worker_args = [(chunks[i], gpu_ids[i], args) for i in range(num_gpus)]

    # 3. Run processes in parallel
    print(f"Starting parallel processing on {num_gpus} GPUs: {gpu_ids}")
    # Use 'spawn' start method for CUDA compatibility
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        # starmap applies the arguments from each tuple in worker_args to the worker_process function
        results_from_pool = pool.starmap(worker_process, worker_args)

    # 4. Gather and Save Results
    print("All chunks processed. Gathering results...")
    final_results = []
    for sublist in results_from_pool:
        final_results.extend(sublist)

    with open(os.path.expanduser(args.answers_file), 'wb') as f:
        pickle.dump(final_results, f)

    print(f"Successfully saved {len(final_results)} combined pseudo labels to {args.answers_file}")