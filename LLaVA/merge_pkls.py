import os

def merge_pkls(pkl_list, dst_path, dst_pkl_name='llava_v1_5_pseudo_roi.pkl'):
    """
    Get the pkl files from the list and save them to a single pkl file.
    """
    import pickle
    all_results = []
    for pkl_path in pkl_list:
        tmp_pkl_content = pickle.load(open(pkl_path, "rb"))
        all_results.extend(tmp_pkl_content)
    os.makedirs(dst_path, exist_ok=True)
    dst_pkl_path = os.path.join(dst_path, dst_pkl_name)
    with open(dst_pkl_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Saved {len(all_results)} results to {dst_pkl_path}")
    return dst_pkl_path

if __name__ == "__main__":
    merge_pkls(
        pkl_list=[
            '/home/yuheng/datasets/llava_v1_5_pseudo_13b_gqa.pkl',
            '/home/yuheng/datasets/llava_v1_5_pseudo_13b_ocr_vqa.pkl',
        ],
        dst_path='/home/yuheng/datasets/',
        dst_pkl_name='llava_v1_5_pseudo_13b.pkl'
    )
