
import torch
import os
import transformers
from llava.model import LlavaLlamaForCausalLM, LlavaConfig  # Adjust this import based on your LLaVA model structure
# Ensure you have the correct imports for LLaVA models and configs
# from llava.model import LlavaLlamaForCausalLM # Or your specific model class
# from llava.model.language_model.llava_llama import LlavaConfig # Or your specific config class
# Using transformers AutoClasses as placeholders - replace with actual LLaVA classes if needed
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# --- Configuration ---
# You MUST set this correctly based on your twig model's architecture
# TWIG_K is the number of initial layers from the base model that were KEPT and FROZEN.
# Layers from original[TWIG_K] to original[TWIG_K + TWIG_T - 1] were used to init twig_layers.
# Layers from original[TWIG_K] onwards (if not used for twig_layers init) were pruned from self.model.layers.
# Or, more simply, if self.model.layers was truncated to self.model.layers[:TWIG_K], then we need to fill from TWIG_K onwards.
TWIG_K = 15  # !!! EXAMPLE VALUE - SET THIS TO YOUR ACTUAL TWIG_K !!!
TWIG_T = 3  # This is the number of layers that were used to initialize the twig layers.
FINETUNED_MODEL_PATH = f"./checkpoints/llava-v1.5-7b-roi-K{TWIG_K}T{TWIG_T}-152k-v1bf16Mheads-twiginit"
ORIGINAL_MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DSTINATION_MODEL_PATH = f"./checkpoints/llava-v1.5-7b-roi-K{TWIG_K}T{TWIG_T}-152k-v1bf16Mheads-twiginit-filled"
DEVICE = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16  # Or torch.float16, torch.float32 depending on your models


def run_fill_pruned_model():
    print(f"Starting to fill pruned parts of the model at {FINETUNED_MODEL_PATH}")
    print(f"Using TWIG_K = {TWIG_K}. Ensure this is correct.")
    print(f"Using device: {DEVICE}")

    # 1. Load Finetuned (Pruned) Model
    #    Load with a config that describes the FULL original architecture.
    #    If FINETUNED_MODEL_PATH's config.json was modified to be smaller, this is important.
    finetuned_config = transformers.AutoConfig.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)
    finetuned_config = type(finetuned_config).from_dict(finetuned_config.to_dict())

    finetuned_model = LlavaLlamaForCausalLM.from_pretrained(
        FINETUNED_MODEL_PATH,
        config=finetuned_config,  # Ensure we use the full config for architecture
        torch_dtype=MODEL_DTYPE,
    )
    finetuned_model.to(DEVICE)
    finetuned_model.eval()

    # For LLaVA, the main language model part is often finetuned_model.model
    # Adjust if your model structure is different.
    finetuned_llama_model = finetuned_model.model

    # 2. Load Original Base Model
    original_config = transformers.AutoConfig.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
    original_config = type(original_config).from_dict(original_config.to_dict())
    original_config.multi_tower_mode = False
    config_cls = LlavaConfig()
    original_config = config_cls.from_dict(original_config.to_dict())
    original_model = LlavaLlamaForCausalLM.from_pretrained(
        ORIGINAL_MODEL_PATH,
        config=original_config,  # Ensure we use the full config for architecture
        torch_dtype=MODEL_DTYPE,
    )
    original_model.to(DEVICE)
    original_model.eval()
    original_llama_model = original_model.model

    original_total_layers = original_config.num_hidden_layers
    print(f"Original model has {original_total_layers} layers.")
    print(
        f"Finetuned model currently has {len(finetuned_llama_model.layers)} layers (expected to be {original_total_layers} if loaded with full config).")

    # Safety check: from_pretrained should have built the full layer list based on config
    if len(finetuned_llama_model.layers) != original_total_layers:
        print("Error: Finetuned model was not loaded with the full number of layers based on original config.")
        print(f"  Expected {original_total_layers} layers, but found {len(finetuned_llama_model.layers)}.")
        print(
            "  This script assumes that `from_pretrained` with the original model's config creates the full architecture skeleton.")
        print("  The missing layers might be randomly initialized. We will attempt to fill them.")
        # This is a critical point. If the list is shorter, state_dict loading will fail for out-of-bounds indices.
        # However, HuggingFace usually initializes all layers defined in the config.
        # If it IS shorter, the config wasn't applied correctly or model init is different.

    # --- 3. Reconstruct and Repopulate Pruned Parts ---
    with torch.no_grad():  # No need for gradients during weight copying
        # Part A: Reconstruct/fill self.model.layers
        # These are layers in the LLaMA backbone that were pruned during twig saving.
        # We are filling layers from index TWIG_K up to original_total_layers - 1.
        print(
            f"Attempting to fill layers from index {TWIG_K} to {original_total_layers - 1} in finetuned_model.model.layers...")
        for i in range(TWIG_K, original_total_layers):
            if i < len(original_llama_model.layers) and i < len(finetuned_llama_model.layers):
                print(
                    f"  Copying weights for layer {i} from original_model.model.layers[{i}] to finetuned_model.model.layers[{i}]")
                try:
                    state_to_load = original_llama_model.layers[i].state_dict()
                    finetuned_llama_model.layers[i].load_state_dict(state_to_load)
                except Exception as e:
                    print(f"    Error loading state_dict for layer {i}: {e}")
            else:
                print(
                    f"  Skipping layer {i}: index out of bounds. Original has {len(original_llama_model.layers)}, Finetuned has {len(finetuned_llama_model.layers)}.")

        # Part B: Reconstruct/fill self.model.norm (final layer norm of LLaMA backbone)
        if hasattr(finetuned_llama_model, 'norm') and finetuned_llama_model.norm is not None and \
            hasattr(original_llama_model, 'norm') and original_llama_model.norm is not None:
            print("Copying weights for model.model.norm...")
            try:
                state_to_load = original_llama_model.norm.state_dict()
                finetuned_llama_model.norm.load_state_dict(state_to_load)
                print("  model.model.norm weights loaded.")
            except Exception as e:
                print(f"    Error loading state_dict for model.model.norm: {e}")
        else:
            print("Skipping model.model.norm: Missing in finetuned or original model structure.")

        # Part C: Reconstruct/fill self.lm_head (final output head of the LLaVA model)
        if hasattr(finetuned_model, 'lm_head') and finetuned_model.lm_head is not None and \
            hasattr(original_model, 'lm_head') and original_model.lm_head is not None:
            print("Copying weights for model.lm_head...")
            try:
                state_to_load = original_model.lm_head.state_dict()
                finetuned_model.lm_head.load_state_dict(state_to_load)
                print("  model.lm_head weights loaded.")
            except Exception as e:
                print(f"    Error loading state_dict for model.lm_head: {e}")
        else:
            print("Skipping model.lm_head: Missing in finetuned or original model structure.")

    # --- 4. Save the "Completed" Model ---
    print(f"Saving the completed model to: {DSTINATION_MODEL_PATH}")
    # Ensure model is on CPU before saving if it was moved to GPU and you want standard HF save
    finetuned_model.save_pretrained(DSTINATION_MODEL_PATH)
    #finetuned_tokenizer.save_pretrained(DSTINATION_MODEL_PATH)

    print("Done. The model should now have its pruned parts filled from the original.")


if __name__ == '__main__':
    # Before running, ensure TWIG_K is set correctly at the top of this script.
    # Also, ensure your LLaVA model class imports are correct if not using AutoClasses.
    # Example: from llava.model import LlavaLlamaForCausalLM
    # And adjust finetuned_llama_model = finetuned_model.model if necessary.
    os.makedirs(DSTINATION_MODEL_PATH, exist_ok=True)
    command = f"cp {FINETUNED_MODEL_PATH}/*.json {DSTINATION_MODEL_PATH}/"
    os.system(command)
    command = f"cp {FINETUNED_MODEL_PATH}/t* {DSTINATION_MODEL_PATH}/"
    os.system(command)
    run_fill_pruned_model()


