import os
import json

def make_jsonl_file(json_path, save_path, save_name = 'llava_v1_5_selected_qa.jsonl', data_path='/home/yuheng/datasets', qa_pair='first'):
    """
    Save the json content to a jsonl file.
    """
    json_content = json.load(open(json_path, "r"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_file = os.path.join(save_path, save_name)
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

make_jsonl_file(json_path='../datasets/llava_v1_5_mix665k.json',
                 save_path='/root/autodl-tmp/datasets',
                 save_name='llava_v1_5_mix665k_selected_qa.jsonl',
                 data_path='/root/autodl-tmp/datasets',
                 qa_pair='first')