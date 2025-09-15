import os
import json

dataset_dir = '/home/yuheng/datasets/' #replace with your dataset path
dst_json_path = os.path.join(dataset_dir,'llava_v1_5_mix665k.json') #"/home/yuheng/datasets/llava_v1_5_mix665k.json"
tmp_json_content = json.load(open(dst_json_path, "r"))
json_content_nococo =  [ele for ele in tmp_json_content if 'image' in ele.keys() and 'coco' not in ele['image']]
coco = [ele for ele in tmp_json_content if 'image' in ele.keys() and 'coco' in ele['image']]
vg_100k = [ele for ele in json_content_nococo if 'image' in ele.keys() and 'VG_100K' in ele['image']]
gqa = [ele for ele in json_content_nococo if 'image' in ele.keys() and 'gqa' in ele['image']]
textvqa = [ele for ele in json_content_nococo if 'image' in ele.keys() and 'textvqa' in ele['image']]
ocr_vqa = [ele for ele in json_content_nococo if 'image' in ele.keys() and 'ocr_vqa' in ele['image']]

test_json_152k = os.path.join(dataset_dir,'llava_v1_5_mix152k_test.json')
test_json_152k_content = gqa + ocr_vqa
with open(test_json_152k, "w") as f:
    json.dump(test_json_152k_content, f, indent=2)

