# Getting Started with SD-RPN upon Qwen2.5-VL

## Installation
Please check the installation instructions in `qwen-vl-finetue/README.md`.

## Training

To train SD-RPN upon Qwen2.5-VL, please follow the instructions below:
1. Download the pkl [annotation](https://huggingface.co/YuhengSSS/roi_pseudo) and **move** it in your `dataset` folder.
```
# export HF_ENDPOINT=https://hf-mirror.com # for China users
huggingface-cli download YuhengSSS/roi_pseudo --local-dir ./
```
2. Download the `GQA` and `OCR-VQA` in your `dataset` folder. 


3. **Start training!** Note that before training, you need to configure the `miniconda3 path`, `HUGGINGFACE_HUB_CACHE` and `DATASET_PATH` in the scripts under `qwen-vl-finetue/scripts/A6000/finetune_roi_fix_res.sh` and `qwen-vl-finetue/scripts/A6000/finetune_roi_fix_res_7b.sh` to your own path. It takes less than 4 hours to train SD-RPN+7B on 4 A6000 GPUs.

```
# for 3B
bash qwen-vl-finetue/scripts/A6000/finetune_roi_fix_res.sh

# for 7B
bash qwen-vl-finetue/scripts/A6000/finetune_roi_fix_res_7b.sh
```
4. Merge the checkpoint with the original weights to obtain the final model.
```
python merge_pruned_model.py #change FINETUNED_MODEL_PATH, ORIGINAL_MODEL_PATH and DSTINATION_MODEL_PATH to your own path
```


## Inference
We utilize lmms-eval to evaluate the model. Please follow the instructions below:

0. Download the pretrained model and move it to your `checkpoints` folder if you want to evaluate our pretrained model.
```
# export HF_ENDPOINT=https://hf-mirror.com # for China users
#7B
huggingface-cli download YuhengSSS/qwen2_5vl-7b-roi-K16T3-152k-v1bf16Mheads-twiginit-filled --repo-type model --local-dir ./
```

1. Install lmms-eval, check the script in `lmms-eval/README.md`.


2. Run the evaluation script in `lmms-eval`. Change the `checkpoint_path` to your own path.
```
bash lmms-eval/examples/A6000/Qwen2.5vl-7b.sh
```

## Make pseudo labels
1. Generate pseudo labels for OCR-VQA and GQA datasets
```
python make_jsonl.py #change the path to your own path
bash make_pseudo_label_natural.sh #change the path to your own path
bash make_pseudo_label_textual.sh #change the path to your own path
```

2. Merge the pseudo labels
```
python merge_pkls.py #change the path to your own path
```


