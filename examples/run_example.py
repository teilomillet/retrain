import asyncio
import sys
from pathlib import Path
import yaml
from typing import Dict, Any

# Add the project root to sys.path to allow imports from the retrain package
# This assumes the script is run from the 'examples' directory.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import from retrain
from retrain import run
# from retrain.config_models import TrainingConfig # Keep commented if direct validation is not used here
from retrain.reward import reward
from loguru import logger

# Custom Reward Function Definition
@reward(name="substring_match_reward")
def substring_match_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """
    Rewards if the completion contains an expected substring.
    Penalizes if a parsing error occurred in the environment for this step,
    based on information from the 'step_info' dictionary passed in kwargs.
    Configurable via `config_params`:
        - `expected_substring` (str): The substring to search for.
        - `case_sensitive` (bool, optional): Whether the search is case-sensitive. Defaults to False.
        - `parsing_error_penalty` (float, optional): The penalty to apply if a parsing error is detected. Defaults to -1.0.
    """
    step_info = kwargs.get("step_info")
    logger.debug(f"substring_match_reward: Received step_info: {step_info}")

    current_parsing_error_penalty = float(config_params.get("parsing_error_penalty", -1.0))

    if isinstance(step_info, dict):
        # Check for parsing errors in a more streamlined way
        error_message = None
        error_source_key = None

        # Check top-level key first (most common case from SmolAgentEnv)
        if "parsing_error_in_text_response" in step_info:
            error_message = step_info["parsing_error_in_text_response"]
            error_source_key = "parsing_error_in_text_response"
        # Then check within 'action_received'
        elif "action_received" in step_info and isinstance(step_info["action_received"], dict) and \
             "parsing_error_message" in step_info["action_received"]:
            error_message = step_info["action_received"]["parsing_error_message"]
            error_source_key = "action_received.parsing_error_message"
            logger.debug("substring_match_reward: Extracted error from step_info['action_received']['parsing_error_message']")
        # Fallback to 'step_error'
        elif "step_error" in step_info:
            error_message = step_info["step_error"]
            error_source_key = "step_error"

        if error_message:
            logger.debug(
                f"substring_match_reward: Applying penalty {current_parsing_error_penalty} "
                f"due to error (key: '{error_source_key}'): {error_message}"
            )
            return current_parsing_error_penalty

    # Original logic for substring matching (only reached if no error penalty was applied)
    expected_substring = config_params.get("expected_substring")
    if not expected_substring: # None or empty string
        logger.warning(
            f"substring_match_reward: 'expected_substring' not found or empty in config_params for prompt: {prompt[:50]}..."
        )
        return 0.0
    
    # Ensure expected_substring is a string for comparison, though .get should provide it or None
    expected_substring = str(expected_substring)

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

async def main():
    logger.info("--- Retrain Example Runner ---")

    config_file_path = Path(__file__).parent / "simple_grpo_config.yaml"
    logger.info(f"Attempting to load configuration from: {config_file_path}")

    if not config_file_path.exists():
        logger.error(f"Configuration file not found at: {config_file_path}")
        return

    try:
        with open(config_file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {e}")
        return
    except Exception as e: # Catches other errors during loading/validation
        logger.error(f"Error loading or validating configuration: {e}")
        return

    logger.info("--- Starting retrain.run() ---")
    try:
        # The 'run' function can find 'substring_match_reward' as it's defined and decorated in this file.
        results = await run(config=config_dict)
        logger.info(f"Retrain run completed. Results: {results}")
    except Exception as e: # Simplified error handling for retrain.run()
        logger.error(f"ERROR: An unexpected error occurred during retrain.run: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Explicit reward imports removed for simplicity.
    # Assuming reward functions in the retrain.reward package (like arithmetic, exact_match)
    # are registered via the package's __init__.py.
    # If issues arise with missing rewards, ensure proper registration in the retrain package.

    # For Windows, the default asyncio event loop policy might need to be changed
    # if you encounter issues with subprocesses or certain async operations.
    # if sys.platform == "win32":
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
    logger.info("--- Example run script completed ---") 