import os
# os.environ["UNSLOTH_ENABLE_LOGGING"] = "1" # Enable Unsloth's auto compiler logging
# os.environ["UNSLOTH_COMPILE_DISABLE"] = "1" # Attempt to disable Unsloth's JIT compilation
os.environ["UNSLOTH_COMPILE_DISABLE"] = "0" # ENABLE Unsloth's JIT compilation.
                                           # With core QKV/O/MLP layers now patched (due to lora_dropout=0.0),
                                           # the persistent 'attn_bias' error likely stems from these patched layers
                                           # not executing correctly. Unsloth's JIT compiler provides optimized
                                           # kernels essential for the proper functioning of its patched mechanisms,
                                           # especially when FA2/Xformers are unavailable.

import unsloth # Import unsloth first as per its recommendation
import torch
from retrain.model import get_model # Assuming get_model is accessible from retrain.model
from loguru import logger
import sys

# Configure Loguru for better output
logger.remove()
# Set default level to DEBUG to capture more detailed logs from the library
logger.add(sys.stderr, level="DEBUG")

def run_unsloth_lora_example():
    """
    Demonstrates loading an Unsloth model with LoRA configuration,
    applying PEFT, and checking trainable parameters.
    """
    logger.info("Starting Unsloth LoRA example...")

    # 1. Define Model and PEFT Configuration
    model_loader_type = "unsloth"
    # Using a smaller Qwen model based on user request and Qwen3 GRPO notebook example.
    # model_identifier = "unsloth/Qwen3-0.5B-Base" # Changed from Phi-3 to Qwen3-0.5B
    model_identifier = "unsloth/Qwen3-0.6B-Base" # Corrected to 0.6B based on HF model card.

    model_config_for_loader = {
        "load_in_4bit": False, # Kept False, similar to Qwen3 GRPO notebook for LoRA stability.
                               # A 0.5B model in 16-bit is small (approx 1GB weights).
        "max_seq_length": 2048,
        "dtype": None,  # Let Unsloth choose the optimal dtype
        # Consider adding fast_inference = True and max_lora_rank if using advanced Unsloth features like the notebook.
    }

    # LoRA configuration for PEFT, adapted from Qwen3 GRPO notebook
    lora_rank = 32 # From Qwen3 notebook, can be adjusted
    active_peft_config = {
        "r": lora_rank,
        "lora_alpha": lora_rank * 2, # Qwen3 notebook suggests lora_rank * 2
        "lora_dropout": 0.0,         # Kept 0.0 for Unsloth fast patching
        "bias": "none",
        "use_gradient_checkpointing": "unsloth", # Recommended by Unsloth
        "random_state": 3407,
        "target_modules": [ # From Qwen3 GRPO notebook
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # "use_rslora": False,
        # "loftq_config": None,
    }
    logger.info(f"Attempting to load model: {model_identifier} using {model_loader_type}")
    logger.info(f"Model config for loader: {model_config_for_loader}")
    logger.info(f"PEFT (LoRA) config: {active_peft_config}")

    try:
        logger.debug("Calling get_model function...") # Log before calling get_model
        # 2. Load Model and Apply PEFT using get_model
        # The get_model function should handle both loading the Unsloth model
        # and then applying the PEFT configuration.
        model, tokenizer = get_model(
            model_type=model_loader_type,
            model_name_or_path=model_identifier,
            model_config_overrides=model_config_for_loader,
            peft_config=active_peft_config
        )
        logger.debug("get_model function returned.") # Log after get_model returns

        logger.success("Model and tokenizer loaded and PEFT applied successfully!")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Tokenizer type: {type(tokenizer)}")

        # 3. Check Trainable Parameters
        # Unsloth PEFT models usually have a method to print trainable parameters.
        if hasattr(model, "print_trainable_parameters"):
            logger.info("Trainable parameters:")
            model.print_trainable_parameters()
        else:
            logger.warning("Model does not have 'print_trainable_parameters' method.")
            # Fallback: manual check (simplified)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Manually calculated trainable params: {trainable_params}")
            logger.info(f"Total params: {total_params}")
            if total_params > 0:
                logger.info(f"Percentage trainable: {(100 * trainable_params / total_params):.2f}%")


        # 4. (Optional) Basic Inference Test
        logger.info("Performing a basic inference test...")
        prompt = "What is the capital of France?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device if hasattr(model, 'device') else "cuda")
        
        # Generate text
        # Ensure model is in eval mode for generation
        model.eval()
        with torch.no_grad():
            # Unsloth models might have specific generation kwargs or prefer `generate_unsloth`
            # For simplicity, using standard generate.
            # Add generation config for better outputs if needed (e.g., max_new_tokens)
            #
            # Changed `use_cache=False` back to `use_cache=True`.
            # Previous attempts with use_cache=True failed when Unsloth's core patching
            # (QKV/O/MLP layers) was incomplete (due to lora_dropout != 0) or its
            # JIT compiler was disabled.
            # Now that Unsloth core patching is confirmed active and JIT is enabled,
            # we revisit use_cache=True. The model's "dynamic cache_implementation" warning
            # suggests it might internally expect cache-related logic to be active.
            # This change tests if a fully Unsloth-optimized model works correctly with KV caching enabled.
            outputs = model.generate(**inputs, max_new_tokens=50, use_cache=True)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated text: {generated_text}")

        logger.success("Unsloth LoRA example finished successfully!")

    except ImportError as e:
        logger.error(f"ImportError: {e}. Please ensure Unsloth and all dependencies are installed.")
        logger.error("You might need to run: pip install unsloth transformers torch peft")
    except Exception as e:
        logger.error(f"An error occurred during the Unsloth LoRA example: {e}", exc_info=True)

if __name__ == "__main__":
    run_unsloth_lora_example() 