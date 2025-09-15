#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from ...llm_src import LlamaForCausalLM, LlamaModel

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        image_token_ids: Optional[List[torch.LongTensor]] = None,
        roi_target_map: Optional[torch.LongTensor] = None,
        src_images: Optional[List] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_token_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_token_ids=image_token_ids,
            roi_target_map=roi_target_map,
            src_images=src_images,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        raw_image = kwargs.pop("raw_image", None)
        aux_imgs_list = kwargs.pop("aux_imgs_list", None)
        src_images = kwargs.pop("src_images", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                image_token_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                raw_images=raw_image,
                aux_imgs_list=aux_imgs_list
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            image_token_ids = []

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_token_ids=image_token_ids,
            src_images=src_images,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        inputs['image_token_ids'] = kwargs.get("image_token_ids", [])
        inputs['src_images'] = kwargs.get("src_images", None)
        return inputs
    def load_twig_weights_from_original_model(self, model_args):
        """
        Loads weights INTO the existing twig_layers, twig_norm, and twig_head
        from a separately loaded "original" base model specified by
        model_args.model_name_or_path.

        Assumes that self.model.twig_layers, self.model.twig_norm, and self.twig_head
        have already been initialized (e.g., with random weights or via self.post_init()).

        Args:
            model_args (ModelArguments): Dataclass containing model arguments,
                                         especially model_name_or_path for the original model.
        """
        if not self.model.enable_twig or not self.enable_twig:
            print("Twig is not enabled in the configuration. Skipping loading twig weights.")
            return

        if not model_args.model_name_or_path:
            raise ValueError(
                "model_args.model_name_or_path must be provided to load original model weights for the twig.")
        if getattr(model_args, 'twig_init', True): #init from llava
            twig_init = model_args.twig_init
        else:
            print("Skipping init twig weights from pretrained model.")
            twig_init = False
        if self.model.twig_layers is not None and len(self.model.twig_layers) == self.model.twig_T:
            for i in range(self.model.twig_T):
                if not twig_init: continue
                original_layer_idx = self.model.twig_K + i
                if original_layer_idx < len(self.model.layers):
                    try:
                        # Source of weights is self.model.layers
                        state_dict_to_load = self.model.layers[original_layer_idx].state_dict()
                        self.model.twig_layers[i].load_state_dict(state_dict_to_load)
                        print(f"  Loaded weights from original_model.model.layers[{original_layer_idx}] ")
                    except Exception as e:
                        raise RuntimeError(f"Failed to load state_dict for twig_layer {i} "
                                           f"from self.model.layers[{original_layer_idx}]: {e}")
                else:
                    raise ValueError(
                        f"Not enough layers in self.model.layers ({len(self.model.layers)} layers) "
                        f"to initialize self.model.twig_layers[{i}]. Needed layer index {original_layer_idx}."
                    )
            if self.model.twig_K + self.model.twig_T < len(self.model.layers):
                print(f"  Pruning self.model.layers from index {self.model.twig_K + self.model.twig_T} onwards.")
                print(f"    Original number of layers: {len(self.model.layers)}")
                self.model.layers = self.model.layers[:self.model.twig_K + self.model.twig_T]
        else:
            print(
                f"self.model.twig_layers is None or has incorrect length. Expected {self.model.twig_T} layers. Skipping twig_layers initialization.")


        # 2. Load weights for twig_norm in self.model
        if self.model.twig_norm is not None and self.model.norm is not None:
            try:
                if twig_init:
                    state_dict_to_load = self.model.norm.state_dict()
                    self.model.twig_norm.load_state_dict(state_dict_to_load)
                    print(f"  Loaded weights from original_model.model.norm into self.model.twig_norm")
                    self.model.norm = None
            except Exception as e:
                raise RuntimeError(f"Failed to load state_dict for self.model.twig_norm: {e}")


        # 3. Load weights for twig_head in self (LlamaForCausalLM)
        if self.twig_head is not None and self.lm_head is not None:
            try:
                if twig_init:
                    state_dict_to_load = self.lm_head.state_dict()
                    self.twig_head.load_state_dict(state_dict_to_load)
                    print(f"  Loaded weights from original_model.lm_head into self.twig_head")
                    self.lm_head = None
            except Exception as e:
                raise RuntimeError(f"Failed to load state_dict for self.twig_head: {e}")

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
