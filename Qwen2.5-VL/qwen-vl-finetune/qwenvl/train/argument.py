import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

     #for ROI prediction
    enable_twig: bool = field(default=False) #enable twig layers for roi prediction
    twig_K: Optional[int] = field(default=0) #how many layers to keep before roi prediction
    twig_T: Optional[int] = field(default=0) # how many layers are used to used as roi predictor
    roi_branch: bool = field(default=False) # just keep the same as enable_twig
    twig_freeze: Optional[int] = field(default=None)  # If None, do not freeze any layers. If int, freeze the last `twig_freeze` layers.
    roi_source: Optional[str] = field(default='qk')  # 'hidden_states' or 'qk'
    roi_loss: Optional[str] = field(default='bce') #bce or focal loss
    twig_init: Optional[bool] = field(default=False) #just set it to true for simplicity, init twig_t weights from original llava's model
    roi_super_type: Optional[str] = field(default='lazy')  #'v1' or 'lazy', v1 for llava's self prediction as supervision while lazy use supervision tokens from llava's annotation
    roi_multi_head: Optional[bool] = field(default=False)  # If True, use multi-head ROI loss, otherwise use single head

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)

    roi_samples: int = -1
    roi_data_path: Optional[str] = field(default=None)
    ab_sink: Optional[bool] = field(default=False)  #
    ab_fg_bbox: Optional[bool] = field(default=False)  # only used for ablation study
    fix_res: Optional[bool] = field(default=False)  # fix the image resolution to 576 tokens by resizing and expanding square
    multi_scale_training: Optional[bool] = field(default=False)  # use multi-scale training
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
