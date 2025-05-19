import os
os.environ["UNSLOTH_ENABLE_LOGGING"] = "1" # Enable Unsloth's auto compiler logging
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1" # Attempt to disable Unsloth's JIT compilation

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
    # Using a publicly available Unsloth model.
    # Replace with your desired model if needed.
    # For this example, we use Phi-3-mini, which is relatively small.
    model_identifier = "unsloth/Phi-3-mini-4k-instruct" 

    model_config_for_loader = {
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "dtype": None,  # Let Unsloth choose the optimal dtype (e.g., torch.bfloat16 if available)
        # "token": "YOUR_HF_TOKEN" # Add Hugging Face token if needed for gated models
    }

    # LoRA configuration for PEFT
    # Unsloth's get_peft_model can often auto-detect target_modules.
    # If you need to specify them, you can add a 'target_modules' key, e.g.:
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] # Example for Llama
    # For Phi-3, common targets are ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    # If left unspecified, Unsloth will attempt to find them.
    active_peft_config = {
        "r": 16,                             # Rank of the LoRA matrices
        "lora_alpha": 32,                    # Alpha scaling factor for LoRA
        "lora_dropout": 0.05,                # Dropout probability for LoRA layers
        "bias": "none",                      # Whether to use bias in LoRA layers ("none", "all", or "lora_only")
        "use_gradient_checkpointing": "unsloth", # Recommended by Unsloth for memory saving
        "random_state": 3407,
        "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"], # Explicit for Phi-3
        # "use_rslora": False,               # Rank Stabilized LoRA
        # "loftq_config": None,              # LoFTQ configuration
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