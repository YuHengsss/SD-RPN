import torch
from tqdm import tqdm
import os
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval import utils
from accelerate import Accelerator, DistributedType
from transformers import AutoModelForCausalLM, AutoProcessor
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from typing import List, Optional, Union, Tuple
import logging
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from datetime import timedelta

eval_logger = logging.getLogger("lmms-eval")

# Suppress verbose deepseek_vl logging
logging.getLogger("deepseek_vl").setLevel(logging.WARNING)

@register_model("deepseek_vl")
class DeepSeekVL(lmms):
    def __init__(
        self,
        pretrained: str = "deepseek-ai/deepseek-vl-7b-chat",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        device_map: Optional[str] = "auto",
        two_stage_roi=False,  # whether to use two-stage roi model
        roi_baseline=False,  # whether to use roi baseline model
        roi_conf_thresh=0.1,  # confidence threshold for roi
        roi2stage_max_ratio = -1,
        is_debug = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.batch_size_per_gpu = int(batch_size)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Load model and processor
        self._model = MultiModalityCausalLM.from_pretrained(
            self.pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self._config = self._model.config

        self.processor = VLChatProcessor.from_pretrained(
            self.pretrained,
            trust_remote_code=True
        )
        self._tokenizer = self.processor.tokenizer
        # Deepseek doesn't have a single max_length attribute easily accessible. This is a reasonable guess.
        self._max_length = self._model.config.language_config.max_position_embeddings

        if self.accelerator.num_processes > 1:
            self._model = self.accelerator.prepare(self._model)
        else:
            self._model.to(self._device)

        self.model.eval()
        self.roi_baseline = roi_baseline
        self.roi_conf_thresh = roi_conf_thresh
        self.roi2stage_max_ratio = roi2stage_max_ratio
        self.model.language_model.model.upscale_method = 'mask'
        self.model.language_model.model.upscale_method = 'mask'

        if two_stage_roi:
            self.model.language_model.model.enable_twig = True
            self.model.language_model.model.roi_enable2stage = True
            self.model.language_model.model.two_stage_vanilla = True
            self.model.language_model.model.is_debug = is_debug
        else:
            self.model.language_model.model.enable_twig = False
            self.model.language_model.model.roi_enable2stage = False
            self.model.language_model.model.two_stage_vanilla = False
            self.model.language_model.model.is_debug = is_debug
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for request in requests:
            # The Instance object holds the arguments for each request
            args = request.args
            context, continuation = args[0], args[1]

            # Retrieve image data from the Instance
            image_list = request.doc_to_visual(request.doc)
            if not isinstance(image_list, list):
                image_list = [image_list]
            pil_images = [img.convert("RGB") for img in image_list if img]

            # Create conversation payload
            image_placeholders = " ".join(["<image_placeholder>"] * len(pil_images))

            # Prepare context for tokenization (prompt without completion)
            conv_context = [
                {"role": "User", "content": f"{image_placeholders}\n{context}", "images": pil_images},
                {"role": "Assistant", "content": ""},
            ]
            inputs_context = self.processor(conversations=conv_context, images=pil_images, force_batchify=True).to(self.model.device)

            # Prepare full conversation for labels (prompt + completion)
            conv_full = [
                {"role": "User", "content": f"{image_placeholders}\n{context}", "images": pil_images},
                {"role": "Assistant", "content": continuation},
            ]
            inputs_full = self.processor(conversations=conv_full, images=pil_images, force_batchify=True).to(self.model.device)

            # Create labels and mask out the context part
            labels = inputs_full.input_ids.clone()
            context_length = inputs_context.input_ids.shape[1]
            labels[:, :context_length] = -100 # Mask out context tokens

            with torch.no_grad():
                inputs_embeds = self.model.prepare_inputs_embeds(
                    input_ids=inputs_full.input_ids,
                    pixel_values=inputs_full.pixel_values,
                    images_seq_mask=inputs_full.images_seq_mask,
                    images_emb_mask=inputs_full.images_emb_mask
                )

                outputs = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs_full.attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

            # Check if the predicted tokens match the completion
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = inputs_full.input_ids[:, context_length:]
            is_greedy = (greedy_tokens[:, context_length-1:-1] == cont_toks).all().item()

            res.append((float(loss.item()), is_greedy))
            pbar.update(1)

        pbar.close()
        return res


    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        #
        # The following code is adapted from the LLaVA implementation in lmms-eval
        #
        def _collate(req: Instance):
            # The negative sign on len(toks) sorts descending to batch longest sequences together
            toks = self.tokenizer.encode(req[0])
            return -len(toks), req[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size_per_gpu, batch_fn=None)

        num_iters = len(requests) // self.batch_size_per_gpu if len(requests) % self.batch_size_per_gpu == 0 else len(requests) // self.batch_size_per_gpu + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        self.model.language_model.model.roi_conf_thresh = self.roi_conf_thresh
        self.model.language_model.model.roi2stage_max_ratio = self.roi2stage_max_ratio

        for chunk_instances in chunks:
            all_images = []
            context, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk_instances)
            assert len(context) == 1, "This model currently only supports batch size of 1 for generation."
            task = task[0]
            split = split[0]
            context = context[0]  # only consider the batch size 1 for now
            gen_kwargs = all_gen_kwargs[0]
            image_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]

            pil_images = [img.convert("RGB") for img in image_list[0] if img] ## should assert bs=1 here
            all_images.extend(pil_images)
            img_path = ['placeholder'] * len(pil_images)  # Placeholder for image paths, if needed
            image_placeholders = " ".join(["<image_placeholder>"] * len(pil_images))
            conversation = [
                {
                    "role": "User",
                    "content": f"{image_placeholders}\n{context}",
                    "images": img_path,
                },
                {"role": "Assistant", "content": ""},
            ]

            # Prepare inputs for the batch
            prepare_inputs = self.processor(
                conversations=conversation,
                images=all_images,
                force_batchify=True
            ).to(self.model.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # Generate response
            try:
                with torch.no_grad():
                    inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                    outputs = self.model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=self.tokenizer.eos_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],

                        visual_masks=prepare_inputs['input_ids'] == 100015,
                        src_images=pil_images,
                        image_processor=self.processor.image_processor,
                        aligner=self.model.aligner,
                        vision_model=self.model.vision_model
                    )
            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                responses = [""]
            # Slice the output tokens to remove the prompt
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), responses)


            res.extend(responses)
            pbar.update(1)

        pbar.close()
        res = re_ords.get_original(res)
        return res

    def _model_call(self, inps):
        # Not used for this model's implementation
        pass

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for DeepSeek-vl")
