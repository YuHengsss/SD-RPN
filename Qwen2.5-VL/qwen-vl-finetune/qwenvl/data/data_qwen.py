import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
from torchcodec.decoders import VideoDecoder
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

from qwen_src.mm_utils import create_pseudo_labels,expand2square

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    #tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = ["<image>"]
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                 data_args, load_data=True, fix_res=False, multi_scale_training=False):
        super(LazySupervisedDataset, self).__init__()

        if load_data:
            dataset = data_args.dataset_use.split(",")
            dataset_list = data_list(dataset)
            rank0_print(f"Loading datasets: {dataset_list}")
            
            list_data_dict = []

            for data in dataset_list:
                file_format = data["annotation_path"].split(".")[-1]
                if file_format == "jsonl":
                    annotations = read_jsonl(data["annotation_path"])
                else:
                    annotations = json.load(open(data["annotation_path"], "r"))
                sampling_rate = data.get("sampling_rate", 1.0)
                if sampling_rate < 1.0:
                    annotations = random.sample(
                        annotations, int(len(annotations) * sampling_rate)
                    )
                    print(f"sampling {len(annotations)} examples from dataset {data}")
                else:
                    rank0_print(f"dataset name: {data}")
                for ann in annotations:
                    ann["data_path"] = data["data_path"]
                list_data_dict += annotations
            
            rank0_print(f"Total training samples: {len(list_data_dict)}")
            random.shuffle(list_data_dict)  # Randomly shuffle the data for training
            self.list_data_dict = list_data_dict
        else:
            self.list_data_dict = []


        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        self.fix_res = fix_res
        self.multi_scale_training = multi_scale_training

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file, return_size=False, preset_pixels=None):
        processor = copy.deepcopy(self.data_args.image_processor)
        if self.multi_scale_training:
            processor.max_pixels = int(preset_pixels)
            processor.min_pixels = int(preset_pixels)
        image = Image.open(image_file).convert("RGB")
        image_size = image.size
        if self.fix_res: image = expand2square(image, (127, 127, 127))
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        if return_size:
            return image_tensor, grid_thw, image_size
        return image_tensor, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    if self.fix_res:
                        grounding_attn_o2i = self.list_data_dict[i].get('grounding_attn_o2i', None)
                        processor_pixels = grounding_attn_o2i.shape[0] * grounding_attn_o2i.shape[1] * 28 * 28
                        image, grid_thw, image_size = self.process_image_unified(image_file, return_size=self.fix_res, preset_pixels=processor_pixels)
                    else:
                        image, grid_thw = self.process_image_unified(image_file, return_size=self.fix_res)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                if self.fix_res:
                    grounding_attn_o2i = self.list_data_dict[i].get('grounding_attn_o2i', None)
                    processor_pixels = grounding_attn_o2i.shape[0] * grounding_attn_o2i.shape[1] * 28 * 28
                    image, grid_thw, image_size = self.process_image_unified(image_file, return_size=self.fix_res, preset_pixels=processor_pixels)
                else:
                    image, grid_thw = self.process_image_unified(image_file, return_size=self.fix_res)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(
                    video_file
                )
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
            if self.fix_res: data_dict["original_image_size"] = image_size  # (width, height) of the first
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        if 'roi_target_map' in instances[0]:
            roi_target_maps = [instance['roi_target_map'] for instance in instances]
            if all(x is not None and x.shape == roi_target_maps[0].shape for x in roi_target_maps):
                batch['roi_target_map'] = torch.stack(roi_target_maps)
            else:
                batch['roi_target_map'] = roi_target_maps
            batch['src_images'] = [instance['src_images'] for instance in instances]
                
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        if 'roi_target_map' in instances[0]:
            roi_target_maps = [instance['roi_target_map'] for instance in instances] #FIXME, the roi_target_map may not be the same shape
            if all(x is not None and x.shape == roi_target_maps[0].shape for x in roi_target_maps):
                batch['roi_target_map'] = torch.stack(roi_target_maps)
            else:
                batch['roi_target_map'] = roi_target_maps

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    roi_super_type='lazy',
    is_2tage_tuning=False,
    pseudo_gaussian_smooth=False,
    ab_sink=False,
    ab_fg_bbox=False,
    fix_res=False,
    multi_scale_training=False,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if roi_super_type == 'lazy':
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    elif roi_super_type == 'v1':
        train_dataset = ROITrainingDataset(
            roi_data_path=data_args.roi_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            pseudo_gaussian_smooth=pseudo_gaussian_smooth,
            ab_sink=ab_sink,
            ab_fg_bbox=ab_fg_bbox,
            roi_samples=getattr(data_args, 'roi_samples', -1),
            fix_res=fix_res,
            multi_scale_training=multi_scale_training,
        )
    else:
        raise NotImplementedError(f"Unsupported roi_super_type: {roi_super_type}. Supported types are 'lazy' and 'v1'.")

    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

class ROITrainingDataset(LazySupervisedDataset):
    """Dataset for RoI prediction, inheriting from Qwen's LazySupervisedDataset."""

    def __init__(self, roi_data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 roi_sink_threshold: float = 1e-2,
                 roi_binary_coeff: float = 0.2,
                 pseudo_gaussian_smooth=False,
                 ab_sink=False, ab_fg_bbox=False,
                 roi_samples=-1, fix_res=False,
                 multi_scale_training=False,
                 ):

        # Initialize parent with load_data=False to skip loading of base dataset
        super().__init__(tokenizer=tokenizer, data_args=data_args, load_data=False, fix_res=fix_res, multi_scale_training=multi_scale_training)

        rank0_print(f"Loading RoI data from: {roi_data_path}")
        try:
            custom_list_data_dict = pickle.load(open(roi_data_path, "rb"))
        except Exception as e:
            raise ValueError(f"Could not load or parse RoI data from {roi_data_path}: {e}")

        rank0_print("Formatting RoI inputs...")
        self.list_data_dict = []  # This will be populated with Qwen-compatible dicts #test sample gqa-0353
        self.roi_sink_threshold = roi_sink_threshold
        self.roi_binary_coeff = roi_binary_coeff
        self.pseudo_gaussian_smooth = pseudo_gaussian_smooth
        self.ab_sink = ab_sink
        self.ab_fg_bbox = ab_fg_bbox
        self.image_folder = os.path.dirname(roi_data_path)
        self.fix_res = fix_res
        for roi_sample in custom_list_data_dict:
            image_path = roi_sample['image']
            #if 'ocr_vqa' not in image_path: continue
            prompt = roi_sample['prompt']
            converted_sample = {
                "id": roi_sample['question_id'],
                "image": image_path,
                "data_path": self.image_folder,
                "conversations": [
                    {"from": "human", "value": "<image>\n" + prompt},
                    {"from": "gpt", "value": roi_sample['text']}
                ],
                # Store paths to RoI related .npy files
                "sink_attn": roi_sample['sink_attn'],
                "grounding_attn_o2i": roi_sample['grounding_attn_o2i'],
                #"grounding_attn_q2i": roi_sample.get('grounding_attn_q2i')  # Optional
            }
            self.list_data_dict.append(converted_sample)

        if roi_samples != -1:
            # Shuffle list_data_dict with fixed seed and select 'roi_samples' pairs
            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(self.list_data_dict)
            self.list_data_dict = self.list_data_dict[:roi_samples]
            rank0_print(f"Selected {roi_samples} samples from the dataset")
        else:
            rank0_print(f"Formatted {len(self.list_data_dict)} RoI samples.")

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        # Get the basic data_dict (input_ids, labels, image tensor) from parent
        # Note: As video is not supported for ROI, we assume single image per sample.
        data_dict = super()._get_item(i)

        # Retrieve RoI data for the current sample
        current_converted_sample = self.list_data_dict[i]
        sink_attn = current_converted_sample['sink_attn']
        grounding_attn_o2i = current_converted_sample['grounding_attn_o2i']

        # Process RoI labels
        original_image_size = data_dict.pop('original_image_size', None) if self.fix_res else None
        pseudo_set = create_pseudo_labels(
            sink_attn=sink_attn,
            grounding_attn_o2i=grounding_attn_o2i,
            sink_thresh=self.roi_sink_threshold,
            binary_coff=self.roi_binary_coeff,
            K=100,  # TODO: pass K as an argument
            pseudo_gaussian_smooth=self.pseudo_gaussian_smooth,
            ab_sink=self.ab_sink,
            ab_fg_bbox=self.ab_fg_bbox,
            mask_known_bg=self.fix_res,
            original_image_size=original_image_size
        )
        roi_target_map = torch.tensor(pseudo_set['labels'])
        data_dict['roi_target_map'] = roi_target_map
        if self.ab_sink and self.ab_fg_bbox:
            data_dict['roi_target_map'] = roi_target_map.type_as(data_dict['pixel_values'])
        else:
            data_dict['roi_target_map'] = roi_target_map.type_as(data_dict['labels'])
            data_dict['src_images'] = current_converted_sample['image']
        return data_dict


if __name__ == "__main__":
    pass
