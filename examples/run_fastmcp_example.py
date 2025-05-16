import asyncio
import os
import sys
from pathlib import Path
import yaml
from loguru import logger
from typing import Dict, Any, Optional, List

# Add the project root to sys.path to allow imports from the retrain package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from retrain
from retrain import run # The main entry point
from retrain.config_models import TrainingConfig # Re-add for Pydantic if needed for direct parsing, though run() handles it
from retrain.reward import reward # Import the decorator
from retrain.reward.types import ProcessedTrajectory # Type for reward function

# Ensure retrain and its submodules are importable, adjust path if necessary
# This assumes the script is run from the root of the 'retrain' project
# sys.path.append(str(Path(__file__).resolve().parent.parent)) # Already handled by project_root logic

# Custom Reward Function Definition (copied from run_example.py)
@reward(name="substring_match_reward")
def substring_match_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """
    Rewards if the completion contains an expected substring.
    Configurable via `config_params`:
        - `expected_substring` (str): The substring to search for.
        - `case_sensitive` (bool, optional): Whether the search is case-sensitive. Defaults to False.
    """
    expected_substring = config_params.get("expected_substring")
    if not expected_substring:
        logger.warning(f"substring_match_reward: 'expected_substring' not found in config_params for prompt: {prompt[:50]}...")
        return 0.0

    case_sensitive = config_params.get("case_sensitive", False)

    if case_sensitive:
        if expected_substring in completion:
            logger.debug(f"substring_match_reward: Found '{expected_substring}' in completion (case-sensitive).")
            return 1.0
    else:
        if expected_substring.lower() in completion.lower():
            logger.debug(f"substring_match_reward: Found '{expected_substring}' in completion (case-insensitive).")
            return 1.0
    
    logger.debug(f"substring_match_reward: Did not find '{expected_substring}' in completion.")
    return 0.0

# New reward function for tool usage
@reward(
    name="tool_usage_reward"
)
def tool_usage_reward(
    prompt: str,                                      # The prompt text for the current turn
    completion: str,                                  # The raw LLM output for the current turn
    config_params: Dict[str, Any],                    # Parameters from the YAML config for this reward function
    step_info: Optional[Dict[str, Any]] = None      # Step-specific info. From RewardCalculator, it's the full step_info.
                                                      # From TRL, it's the 'example' dict for the turn.
) -> float:
    """Rewards tool usage and penalizes a lack of tool use when prompt keywords suggest it.
    
    When called by RewardCalculator (initial processing):
      `step_info` contains `auxiliary_data` which contains `original_step_info` which contains `action_received`.
    When called by TRL's internal reward processing:
      `step_info` is the 'example' from TRL's dataset. This 'example' should have an 'auxiliary_data'
      field if we populated it correctly in the HF Dataset passed to TRL's `step` (or if TRL forwards all columns).
      However, if TRL uses its own train_dataset (initialized with only 'prompt'), then `step_info`
      will likely be `{'prompt': '...'}` and will lack our `auxiliary_data`.
    """
    
    # Default parameter values (can be overridden by config_params)
    default_tool_call_reward = 0.1
    default_no_tool_penalty = -0.2
    default_keywords = ["perform_operation", "calculate", "get_server_time", "what is", "how to"]

    tool_call_reward_val = config_params.get("tool_call_reward", default_tool_call_reward)
    no_tool_penalty_val = config_params.get("no_tool_penalty", default_no_tool_penalty)
    prompt_keywords_for_tool_val = config_params.get("prompt_keywords_for_tool", default_keywords)

    if not isinstance(prompt_keywords_for_tool_val, list):
        logger.warning(
            f"tool_usage_reward: 'prompt_keywords_for_tool' in config_params is not a list, "
            f"using defaults. Value: {prompt_keywords_for_tool_val}"
        )
        prompt_keywords_for_tool_val = default_keywords
    
    reward = 0.0
    action_type = None
    llm_action_dict = None

    if step_info:
        # When called from TRL, step_info is the 'example' dict.
        # We expect our auxiliary data to be a field within this 'example' dict.
        payload_aux_data = step_info.get("auxiliary_data")

        if isinstance(payload_aux_data, dict):
            # payload_aux_data is the dictionary we stored in the 'auxiliary_data' column of the HF Dataset.
            # This dictionary should contain 'original_step_info'.
            if "original_step_info" in payload_aux_data and \
               isinstance(payload_aux_data["original_step_info"], dict):
                original_step_info_data = payload_aux_data["original_step_info"]
                if isinstance(original_step_info_data.get("action_received"), dict):
                    llm_action_dict = original_step_info_data["action_received"]
                    # logger.debug(f"tool_usage_reward: Extracted action_received via auxiliary_data.original_step_info: {llm_action_dict}")
            # else:
                # logger.debug(f"tool_usage_reward: 'original_step_info' not found or not dict in payload_aux_data. Keys: {list(payload_aux_data.keys())}")
        # elif step_info.get("action_received"): # Fallback for direct step_info (initial calculation by RewardCalculator)
            # This case handles when RewardCalculator calls this function directly.
            # In that scenario, step_info *is* the original_step_info_data.
        #    if isinstance(step_info.get("action_received"), dict): # Check if action_received is in top-level step_info
        #        llm_action_dict = step_info["action_received"]
        #        logger.debug(f"tool_usage_reward: Extracted action_received directly from step_info: {llm_action_dict}")


    # Revised logic: If the above complex path via 'auxiliary_data' fails (e.g. TRL context with minimal step_info),
    # check if step_info *itself* contains 'action_received'. This covers the direct call from RewardCalculator.
    if not llm_action_dict and step_info and isinstance(step_info.get("action_received"), dict):
        llm_action_dict = step_info["action_received"]
        # logger.debug(f"tool_usage_reward: Extracted action_received directly from step_info (e.g. RewardCalculator call): {llm_action_dict}")


    if llm_action_dict:
        action_type = llm_action_dict.get('action_type')
    else:
        # This warning will be logged if extraction failed through all paths.
        log_message_detail = ""
        if step_info is None:
            log_message_detail = "step_info was None."
        elif isinstance(step_info, dict):
            log_message_detail = f"step_info was a dict with keys: {list(step_info.keys())}."
            # For very verbose logging if needed:
            # log_message_detail += f" Full step_info: {str(step_info)[:500]}"
        else:
            log_message_detail = f"step_info was of type {type(step_info)}, value: {str(step_info)[:200]}."

        logger.warning(
            f"tool_usage_reward: Could not extract 'action_received' dictionary. "
            f"Cannot determine action_type. Details: {log_message_detail}"
        )
        action_type = None # Ensure action_type is None if extraction failed
    
    # Apply reward/penalty based on action_type and prompt
    if action_type == "tool_call":
        reward += tool_call_reward_val
        # logger.debug(f"tool_usage_reward: Tool call detected. Reward: +{tool_call_reward_val}")
    elif any(keyword.lower() in prompt.lower() for keyword in prompt_keywords_for_tool_val):
        # Keywords suggest tool use, but action_type is not 'tool_call' (or is None)
        reward += no_tool_penalty_val
        logger.debug(
            f"tool_usage_reward: Prompt keywords {prompt_keywords_for_tool_val} suggest tool use, "
            f"but action_type is '{action_type}'. Penalty: {no_tool_penalty_val}. "
            f"Prompt: '{prompt[:100]}...' Completion: '{completion[:100]}...'"
        )
    # else:
        # logger.debug(f"tool_usage_reward: No tool call detected, no relevant keywords in prompt. Base reward: 0.0")

    return reward

async def main():
    logger.remove() 
    logger.add(sys.stderr, level="DEBUG") 
    
    logger.info("--- Retrain FastMCPEnv Example Runner (via retrain.run) ---")
    logger.info("NOTE: This example expects 'examples/sample_fastmcp_server.py' to be RUNNING separately.")
    logger.info("It also expects a real model (e.g., Qwen/Qwen3-0.6B) to be downloadable.")

    config_file_path = Path(__file__).parent / "fastmcp_example_config.yaml"
    logger.info(f"Attempting to load configuration from: {config_file_path}")

    if not config_file_path.exists():
        logger.error(f"Configuration file not found at: {config_file_path}")
        return

    try:
        with open(config_file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # The Pydantic model validation (TrainingConfig) will happen inside retrain.run()
        logger.info("Configuration dictionary loaded.")

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {e}")
        return
    except Exception as e: 
        logger.error(f"Error loading configuration dictionary: {e}")
        return

    logger.info("--- Starting retrain.run() with FastMCPEnv configuration ---")
    try:
        # The 'retrain.run' function will now handle:
        # - Parsing and validating the full config_dict against TrainingConfig.
        # - Initializing the Model (EleutherAI/pythia-14m in this config).
        # - Initializing FastMCPEnv (connecting to server_url, setting up tools via provider).
        # - Initializing PromptSource, RewardCalculator, and Trainer (GRPOTrainer).
        # - Executing the training loop (which includes rollouts in the FastMCPEnv).
        results = await run(config=config_dict)
        logger.info(f"Retrain run with FastMCPEnv completed. Results: {results}")
    except ValueError as e:
        logger.error(f"ERROR: A ValueError occurred during retrain.run: {e}")
    except ImportError as e:
        logger.error(f"ERROR: An ImportError occurred: {e}. Ensure all dependencies (like fastmcp, trl, transformers, accelerate) are installed.")
    except Exception as e:
        logger.error(f"ERROR: An unexpected error occurred during retrain.run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Optional: Pre-import reward functions if they are in separate modules and need registration,
    # similar to run_example.py. For this basic FastMCP test, the placeholder reward in YAML
    # might be sufficient for TRL to initialize if no complex reward logic is immediately triggered.
    # try:
    #     import retrain.reward.substring_match # Example if using this specific reward
    #     logger.debug("Ensured example reward functions are 'imported' for registration.")
    # except ImportError as e:
    #     logger.warning(f"Could not pre-import all standard reward modules: {e}")

    asyncio.run(main())
    logger.info("--- FastMCPEnv Example (via retrain.run) Script Completed ---") 