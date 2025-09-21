import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.train.train import preprocess, LazySupervisedDataset
from PIL import Image
import math
import numpy as np
from llava.ana_utils import register_hooks_in_llava, get_attn_weights_map
import pickle

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_head_wise_attn(attn_dict, key='output2image', ave_sample=False):
    import numpy as np
    rec = []
    for layer, attn_weights in attn_dict.items():
        if key in attn_weights:
            stacked = np.vstack([weights for weights in attn_weights[key]]) #samples, heads
            if ave_sample:
                stacked = np.mean(stacked, axis=0)
            rec.append(stacked)
    return np.array(rec)

def make_jsonl_file(json_path, save_path, save_name = 'llava_v1_5_selected_qa.jsonl', data_path='/home/yuheng/datasets', qa_pair='first'):
    """
    Save the json content to a jsonl file.
    """
    json_content = json.load(open(json_path, "r"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_file = os.path.join(save_path, save_name)
    if os.path.exists(save_file):
        print(f"{save_file} already exists, skip")
    else:
        all_results = []
        for idx, ele in enumerate(json_content):
            if 'image' in ele.keys():# and 'coco' not in ele['image']:
                question_id = str(idx)
                image = os.path.join(data_path, ele['image'])
                conversations = ele['conversations']
                if qa_pair == 'first':
                    text = conversations[0]['value'].replace('<image>\n', '').replace('\n<image>', '')
                else:
                    raise NotImplementedError(f"qa_pair={qa_pair} not implemented")
                if '<image>' in text:
                    pass
                result = {
                    "question_id": question_id,
                    "image": image,
                    "text": text,
                    "category": "default",
                }
                all_results.append(result)
            else: continue
        with open(save_file, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")
        print(f"Saved {len(all_results)} results to {save_file}")

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    make_jsonl_file(
        json_path=args.source_file, save_path=os.path.dirname(os.path.expanduser(args.question_file)),
        save_name=os.path.basename(os.path.expanduser(args.question_file)), data_path=args.data_path
    )
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    activations, hooks = register_hooks_in_llava(model)
    cur_dataset = args.cur_dataset
    image_folder = args.image_folder if args.image_folder else ""
    results = []
    new_questions = [line for line in questions if cur_dataset in line["image"]]
    for line in tqdm(new_questions):
        activations.clear()
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        full_image_path = os.path.join(image_folder, image_file)
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        #image = Image.open(image_file).convert('RGB')
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=False
            )
        if activations.get('visual_mask', None) is None:
            activations['visual_mask'] = model.model.visual_masks

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if "7b" in model_name:
            attn_weights_map = get_attn_weights_map(activations, layers=15, required_layers=[2, 14])
            grounding_attn_o2i = attn_weights_map['layer14']['output2images']
            grounding_attn_o2i = (grounding_attn_o2i[24] + grounding_attn_o2i[13] + grounding_attn_o2i[26])/3
            grounding_attn_q2i = attn_weights_map['layer14']['question2images'][0]
            grounding_attn_q2i = (grounding_attn_q2i[24] + grounding_attn_q2i[13] + grounding_attn_q2i[26])/3
            sink_attn = attn_weights_map['layer2']['output2images'].mean(0) #check process_hidden_states_and_find_indices in ./llava/llm_src/visual_processing if you want exactly the same.
            #as we find sink tokens got very high attention in not vision-centric heads or layers.
            grounding_attn = grounding_attn_o2i
            grounding_attn *= (sink_attn<1e-3).astype(float) #remove sink tokens
        elif "13b" in model_name:
            required_heads = {16: [2, 21, 30], 15: [2], 13: [21, 23, 26]} # layer: [heads]
            required_layers = list(required_heads.keys())
            sink_heads = {16: [17, 22]} # layer: [heads] #check process_hidden_states_and_find_indices in ./llava/llm_src/visual_processing if you want exactly the same.
            #as we find sink tokens got very high attention in not vision-centric heads or layers.
            layers = max(required_layers) + 1
            attn_weights_map = get_attn_weights_map(activations, layers=layers, required_layers=required_layers)
            visual_token_num = attn_weights_map[f'layer{required_layers[0]}']['output2images'].shape[-1]

            grounding_attn_o2i = np.zeros(visual_token_num)
            total_grounding_heads = sum([len(heads) for heads in required_heads.values()])
            for layer, heads in required_heads.items():
                attn_weights = attn_weights_map[f'layer{layer}']['output2images']
                for head in heads:
                    grounding_attn_o2i += attn_weights[head]
            grounding_attn_o2i /= total_grounding_heads

            sink_attn_o2i = np.zeros(visual_token_num)
            total_sink_heads = sum([len(heads) for heads in sink_heads.values()])
            for layer, heads in sink_heads.items():
                attn_weights = attn_weights_map[f'layer{layer}']['output2images']
                for head in heads:
                    sink_attn_o2i += attn_weights[head]
            sink_attn_o2i /= total_sink_heads

            #FIXME: rename layer2_attn to sink_attn
            sink_attn = sink_attn_o2i

        else: raise ValueError(f"Unsupported model {model_name} for attention analysis")


        tmp_result = {"question_id": idx,
                       "prompt": cur_prompt,
                       "text": outputs,
                       "image": os.path.relpath(full_image_path, image_folder),
                       "sink_attn": sink_attn,
                       "grounding_attn_o2i": grounding_attn_o2i,
                       "grounding_attn_q2i": grounding_attn_q2i}
        results.append(tmp_result)

    # Save to pickle file
    with open(answers_file, 'wb') as f:
        pickle.dump(results, f)
    pass

if __name__ == "__main__":
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
    args = parser.parse_args()

    eval_model(args)
