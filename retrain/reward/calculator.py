import asyncio
from typing import List, Dict, Any, Callable, Optional, Union

# Types from this module
from .types import RawRolloutData, ProcessedTrajectory, SingleTurnDataForTrainer
# Remove the TypedDict import, we'll use the Pydantic model for config type hint
# from .types import RewardFunctionConfig as RewardFunctionConfigTypedDict 

# Import Pydantic model for config type hint
from ..config_models import RewardFunctionConfig as RewardFunctionConfigPydantic

# Reward/Verifier registries and utilities
from .reward import create_grpo_batch_reward_func, BatchRewardFunction, get_reward_function
from ..utils.logging_utils import get_logger # Using shared logger
# from ..verifier.verifier import VERIFIER_REGISTRY # If directly using verifier functions outside of create_grpo_batch_reward_func

# For reconstructing prompts if needed (tokenizer might be passed or part of model_utils)
# from ..utils.tokenizer_utils import format_conversation_for_llm # Example
TokenizerObject = Any # Placeholder

# Assuming LLMAction is defined in retrain.environment.types
from retrain.environment.types import LLMAction # Corrected import

logger = get_logger(__name__) # Initialize logger for this module

class RewardCalculator:
    """
    Processes raw rollout data from an environment to calculate final rewards
    for each turn, ready for a trainer.
    It can combine intrinsic environment rewards, step-level external rewards (with verifiers),
    and rollout-level external rewards.
    """

    def __init__(self,
                 step_reward_configs: Optional[Dict[str, RewardFunctionConfigPydantic]] = None, # Use Pydantic model in type hint
                 rollout_reward_configs: Optional[Dict[str, RewardFunctionConfigPydantic]] = None, # Use Pydantic model
                 tokenizer: Optional[Any] = None, # TokenizerObject placeholder was Any
                 # Configuration for how to combine rewards, e.g., weights
                 reward_combination_strategy: Optional[Dict[str, Any]] = None):
        
        self.step_reward_processors: List[BatchRewardFunction] = []
        if step_reward_configs:
            for name, reward_func_cfg_item in step_reward_configs.items(): # reward_func_cfg_item is a Pydantic model
                try:
                    # Access Pydantic model fields using attribute access
                    # Pydantic models handle defaults if fields are not present (based on Field definition)
                    params_val = reward_func_cfg_item.params
                    verifiers_val = reward_func_cfg_item.verifiers or [] # Pydantic defaults Optional to None
                    penalty_val = reward_func_cfg_item.verifier_penalty

                    prepared_config_for_wrapper = {
                        "name": name,
                        "params": params_val,
                        "verifiers": verifiers_val,
                        "verifier_penalty": penalty_val
                    }
                    
                    processor = create_grpo_batch_reward_func(prepared_config_for_wrapper)
                    if processor:
                        self.step_reward_processors.append(processor)
                        logger.info(f"RewardCalculator: Loaded step-reward processor for: {name}")
                    else:
                        logger.warning(f"RewardCalculator: Failed to create step-reward processor for {name}. Function returned None.")
                except Exception as e:
                    logger.error(f"RewardCalculator: Error loading step-reward processor for config '{name}': {e}", exc_info=True)

        self.rollout_reward_functions: List[Callable[[RawRolloutData, List[Dict[str,Any]]], float]] = []
        self.rollout_reward_distribution: List[str] = [] 
        if rollout_reward_configs:
            for name, config_pydantic_item in rollout_reward_configs.items(): # config_pydantic_item is a Pydantic model
                try:
                    reward_func = get_reward_function(name) 
                    if reward_func is None:
                        logger.error(f"RewardCalculator: Rollout reward function '{name}' not found in REWARD_REGISTRY.")
                        continue
                    self.rollout_reward_functions.append(reward_func)
                    
                    # Access Pydantic model field. Pydantic defaults Optional[str] to None if not provided.
                    # The `or "last_step"` handles the case where it's None or an empty string (though empty string is less likely for this field).
                    distribution_strategy_val = config_pydantic_item.distribution_strategy or "last_step"
                                        
                    # Ensure it's a valid string, primarily for type checking, though Pydantic should ensure it's str or None.
                    if not isinstance(distribution_strategy_val, str):
                        logger.warning(f"RewardCalculator: 'distribution_strategy' for {name} resolved to non-string ({distribution_strategy_val}), defaulting to 'last_step'.")
                        distribution_strategy_val = "last_step"
                    self.rollout_reward_distribution.append(distribution_strategy_val)
                    logger.info(f"RewardCalculator: Loaded rollout-reward function: {name}")
                except Exception as e:
                    logger.error(f"RewardCalculator: Error loading rollout-reward function for config '{name}': {e}", exc_info=True)

        self.tokenizer = tokenizer
        self.reward_combination_strategy = reward_combination_strategy or {}

    def _reconstruct_prompt_for_turn(self, 
                                     full_conversation_history: List[Dict[str,str]], 
                                     llm_action_index: int) -> str:
        """
        Reconstructs the prompt that led to the LLM action at the given index.
        The LLM action itself is at `full_conversation_history[llm_action_index]` if it was an 'assistant' turn.
        The prompt is everything before it.
        """
        # An LLM action is an "assistant" message. The prompt is the history *before* this.
        # The `executed_llm_actions` list corresponds to "assistant" turns.
        # If `llm_action_index` is the index in `executed_llm_actions`,
        # we need to find the corresponding point in `full_conversation_history`.

        assistant_message_count = 0
        prompt_end_index_in_history = -1
        for i, message in enumerate(full_conversation_history):
            if message["role"] == "assistant":
                if assistant_message_count == llm_action_index:
                    prompt_end_index_in_history = i
                    break
                assistant_message_count += 1
        
        if prompt_end_index_in_history == -1:
            # Should not happen if indices are correct
            # This might occur if llm_action_index is out of bounds for assistant turns in history
            logger.error(f"[RewardCalculator] Failed to reconstruct prompt for action index {llm_action_index} in history (len {len(full_conversation_history)}). The assistant action count did not match the index. Returning empty prompt.")
            return "" # Or raise an error

        prompt_messages = full_conversation_history[:prompt_end_index_in_history]

        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template") and callable(getattr(self.tokenizer, "apply_chat_template")):
            try:
                return self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True # Important for some models
                )
            except Exception as e:
                logger.warning(f"[RewardCalculator] Error applying chat template during prompt reconstruction: {e}. Falling back to simple join.", exc_info=True)
        
        # Fallback simple join
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt_messages])


    async def _calculate_rewards_for_one_rollout(self, raw_rollout: RawRolloutData) -> ProcessedTrajectory:
        logger.debug("[RewardCalculator._calculate_rewards_for_one_rollout] Received raw_rollout dict.")
        # ADDED DEBUG LOG: Log the number of executed LLM actions received
        logger.debug(f"  Length of raw_rollout['executed_llm_actions']: {len(raw_rollout.get('executed_llm_actions', []))}")
        
        # Access attributes by key from the RawRolloutData dictionary
        try:
            history: List[Dict[str, str]] = raw_rollout["full_conversation_history"]
            llm_actions_list: List[LLMAction] = raw_rollout["executed_llm_actions"]
            intrinsic_rewards_list: List[float] = raw_rollout["intrinsic_rewards_per_turn"]
            step_info_dicts_list: List[Dict[str, Any]] = raw_rollout["step_info_dicts_per_turn"]
        except KeyError as e:
            logger.error(f"  Error accessing keys from raw_rollout dict. Type: {type(raw_rollout)}. Missing key: {e}", exc_info=True)
            logger.error(f"  raw_rollout keys: {list(raw_rollout.keys()) if isinstance(raw_rollout, dict) else 'Not a dict'}")
            return [] 
        except TypeError as e_type:
             logger.error(f"  raw_rollout is not a dictionary or key access failed. Type: {type(raw_rollout)}. Error: {e_type}", exc_info=True)
             return []

        logger.debug(f"  Accessed RawRolloutData keys: history len={len(history)}, llm_actions_list len={len(llm_actions_list)}, intrinsic_rewards_list len={len(intrinsic_rewards_list)}, step_info_dicts_list len={len(step_info_dicts_list)}")

        processed_trajectory: ProcessedTrajectory = []
        num_llm_actions = len(llm_actions_list)
        num_step_infos = len(step_info_dicts_list)

        if num_llm_actions != num_step_infos:
            logger.warning(
                f"[RewardCalculator] Mismatch in lengths: num_llm_actions ({num_llm_actions}) != num_step_infos ({num_step_infos}). "
                f"This may lead to missing step_info for some turns or other inconsistencies."
            )
            if abs(num_llm_actions - num_step_infos) > 0: 
                 logger.debug(f"  Contents of llm_actions_list (first 2 relevant entries): {[{k: str(v)[:50] + '...' if isinstance(v, str) and len(v) > 50 else v for k,v in action.items() if k in ['action_type', 'text', 'raw_llm_output']} for action in llm_actions_list[:2]]}")
                 logger.debug(f"  Contents of step_info_dicts_list (first relevant portion): {step_info_dicts_list[:min(num_step_infos, num_llm_actions + 2)]}")

        if num_llm_actions == 0:
            logger.debug("[RewardCalculator] No LLM actions in this rollout, returning empty trajectory.")
            return []

        ext_step_rewards_per_turn: List[List[float]] = [[0.0] * len(self.step_reward_processors) for _ in range(num_llm_actions)]

        # Loop through each turn (LLM action)
        for i in range(num_llm_actions):
            # Fetch turn-specific data once
            current_llm_action: LLMAction = llm_actions_list[i]
            prompt_text_for_turn: str = self._reconstruct_prompt_for_turn(history, i)
            intrinsic_reward_for_turn: float = intrinsic_rewards_list[i] if i < len(intrinsic_rewards_list) else 0.0
            
            current_step_info: Dict[str, Any] = {}
            if i < num_step_infos:
                current_step_info = step_info_dicts_list[i]
            else:
                # This warning is already covered by the overall mismatch warning, 
                # but good to note if it leads to empty current_step_info here.
                logger.debug(f"[RewardCalculator] Turn {i}: No corresponding step_info found (index {i}, num_step_infos {num_step_infos}). Using empty dict for step_info.")

            # Check for malformed LLM action
            is_malformed_action = not isinstance(current_llm_action, dict) or "raw_llm_output" not in current_llm_action
            if is_malformed_action:
                logger.warning(f"[RewardCalculator] Malformed LLMAction at turn {i} (index in llm_actions_list). Skipping reward calculation for this turn. Action: {current_llm_action}")
                processed_trajectory.append(SingleTurnDataForTrainer(
                    prompt_text=prompt_text_for_turn,
                    completion_text="[MALFORMED_ACTION_SKIPPED_IN_REWARD_CALC]",
                    final_combined_reward=0.0, # Default reward for malformed action
                    auxiliary_data={"error": "Malformed LLMAction during reward processing", 
                                    "original_step_info": current_step_info},
                    old_per_token_logps=None
                ))
                continue # Move to the next LLM action

            # If action is not malformed, proceed with reward calculation
            completion_for_llm_turn: str = current_llm_action["raw_llm_output"]
            action_logprobs = current_llm_action.get("old_per_token_logps")

            # ADDED DEBUG LOG: Confirming before appending to processed_trajectory
            logger.debug(f"  Turn {i}: Preparing to append SingleTurnDataForTrainer. Prompt snippet: '{prompt_text_for_turn[:50]}...', Completion snippet: '{completion_for_llm_turn[:50]}...'")

            # Calculate step-level external rewards
            for processor_idx, processor_fn in enumerate(self.step_reward_processors):
                try:
                    rewards_tensor_list = processor_fn(
                        prompts=[prompt_text_for_turn],
                        completions=[completion_for_llm_turn],
                        infos=[current_step_info] # Use pre-fetched current_step_info
                    )
                    if rewards_tensor_list and isinstance(rewards_tensor_list, list) and len(rewards_tensor_list) > 0:
                        ext_step_rewards_per_turn[i][processor_idx] = float(rewards_tensor_list[0].item())
                    else:
                        logger.warning(f"Step reward processor {processor_idx} for turn {i} returned unexpected result: {rewards_tensor_list}. Defaulting to 0.0.")
                        ext_step_rewards_per_turn[i][processor_idx] = 0.0 
                except Exception as e_proc:
                    logger.error(f"Error calling step reward processor {processor_idx} for turn {i}: {e_proc}", exc_info=True)
                    ext_step_rewards_per_turn[i][processor_idx] = 0.0
            
            # Combine rewards for the current turn
            final_reward = intrinsic_reward_for_turn
            for step_ext_reward in ext_step_rewards_per_turn[i]:
                final_reward += step_ext_reward
            
            # Auxiliary data for the trainer
            aux_data: Dict[str, Any] = {"original_step_info": current_step_info}
            aux_data["intrinsic_env_reward"] = intrinsic_reward_for_turn
            aux_data["external_step_rewards_components"] = ext_step_rewards_per_turn[i]
            # Rollout scores are added to aux_data with the last step later.

            # Create SingleTurnDataForTrainer object
            # ADDED DEBUG LOG: Confirming before appending to processed_trajectory
            logger.debug(f"  Turn {i}: Appending SingleTurnDataForTrainer with reward {final_reward:.2f}. Prompt: '{prompt_text_for_turn[:30]}...', Completion: '{completion_for_llm_turn[:30]}...'") 
            single_turn_data = SingleTurnDataForTrainer(
                prompt_text=prompt_text_for_turn,
                completion_text=completion_for_llm_turn,
                final_combined_reward=final_reward,
                auxiliary_data=aux_data,
                old_per_token_logps=action_logprobs 
            )
            processed_trajectory.append(single_turn_data)

        rollout_level_scores: List[float] = []
        if self.rollout_reward_functions:
            rollout_reward_tasks = []
            # Pass step_info_dicts_list for rollout functions that might need all step infos
            infos_for_rollout_rewards = step_info_dicts_list if step_info_dicts_list else []
            for rr_func in self.rollout_reward_functions:
                if asyncio.iscoroutinefunction(rr_func):
                    task = rr_func(raw_rollout, infos_for_rollout_rewards) 
                else:
                    task = asyncio.to_thread(rr_func, raw_rollout, infos_for_rollout_rewards)
                rollout_reward_tasks.append(task)
            
            calculated_rollout_scores = await asyncio.gather(*rollout_reward_tasks, return_exceptions=True)
            for score_idx, score_result in enumerate(calculated_rollout_scores):
                if isinstance(score_result, Exception):
                    func_name_for_log = getattr(self.rollout_reward_functions[score_idx], '__name__', f'rollout_reward_function_{score_idx}')
                    logger.warning(f"[RewardCalculator] Rollout reward function '{func_name_for_log}' failed: {score_result}", exc_info=True)
                    rollout_level_scores.append(0.0)
                elif score_result is None:
                    func_name_for_log = getattr(self.rollout_reward_functions[score_idx], '__name__', f'rollout_reward_function_{score_idx}')
                    logger.warning(f"[RewardCalculator] Rollout reward function '{func_name_for_log}' returned None. Treating as 0.0.")
                    rollout_level_scores.append(0.0)
                else:
                    try:
                        if not isinstance(score_result, BaseException): 
                            rollout_level_scores.append(float(score_result))
                        else: 
                            func_name_for_log = getattr(self.rollout_reward_functions[score_idx], '__name__', f'rollout_reward_function_{score_idx}')
                            logger.error(f"[RewardCalculator] Rollout reward function '{func_name_for_log}' result was an unexpected exception type not caught earlier: {score_result}. Treating as 0.0.")
                            rollout_level_scores.append(0.0)
                    except (ValueError, TypeError) as e_float_conv:
                        func_name_for_log = getattr(self.rollout_reward_functions[score_idx], '__name__', f'rollout_reward_function_{score_idx}')
                        logger.warning(f"[RewardCalculator] Rollout reward function '{func_name_for_log}' returned non-floatable value: {score_result} (type: {type(score_result)}). Error: {e_float_conv}. Treating as 0.0.")
                        rollout_level_scores.append(0.0)

            for i in range(num_llm_actions):
                current_llm_action: LLMAction = llm_actions_list[i]
                prompt_text_for_turn: str = self._reconstruct_prompt_for_turn(history, i)
                
                # Safely get intrinsic reward for the turn
                intrinsic_reward_for_turn: float = intrinsic_rewards_list[i] if i < len(intrinsic_rewards_list) else 0.0
                
                # Safely get current_step_info for the turn
                current_step_info: Dict[str, Any]
                if i < num_step_infos:
                    current_step_info = step_info_dicts_list[i]
                else:
                    current_step_info = {}
                    # This specific warning for a turn might be too verbose if the general mismatch is already logged.
                    # Let's rely on the initial mismatch warning. Can be changed to debug if needed.
                    logger.debug(f"[RewardCalculator] Turn {i}: Accessing step_info_dicts_list out of bounds (index {i}, len {num_step_infos}). Defaulting step_info to empty dict for this turn.")

                # Check for malformed LLM action (must have "raw_llm_output")
                if not isinstance(current_llm_action, dict) or "raw_llm_output" not in current_llm_action:
                    logger.warning(f"[RewardCalculator] Malformed LLMAction at turn {i} (missing 'raw_llm_output' or not a dict). Skipping reward calculation. Action: {str(current_llm_action)[:200]}")
                    processed_trajectory.append(SingleTurnDataForTrainer(
                        prompt_text=prompt_text_for_turn, # Use pre-fetched prompt
                        completion_text="[MALFORMED_ACTION_SKIPPED_IN_REWARD_CALC]",
                        final_combined_reward=0.0, # Default reward
                        auxiliary_data={"error": "Malformed LLMAction during reward processing", 
                                        "original_step_info": current_step_info}, # Use pre-fetched step_info
                        old_per_token_logps=None
                    ))
                    continue # Skip to the next LLM action

                # If action is valid, get completion and logprobs
                completion_for_llm_turn: str = current_llm_action["raw_llm_output"]
                action_logprobs = current_llm_action.get("old_per_token_logps")

                # Calculate step-level external rewards for this turn
                # ext_step_rewards_per_turn was initialized for all turns already
                for processor_idx, processor_fn in enumerate(self.step_reward_processors):
                    try:
                        # Call the synchronous processor for the current turn (batch size of 1)
                        rewards_tensor_list = processor_fn(
                            prompts=[prompt_text_for_turn],       # Use pre-fetched prompt
                            completions=[completion_for_llm_turn],
                            infos=[current_step_info]             # Use pre-fetched step_info
                        )
                        if rewards_tensor_list and isinstance(rewards_tensor_list, list) and len(rewards_tensor_list) > 0:
                            ext_step_rewards_per_turn[i][processor_idx] = float(rewards_tensor_list[0].item())
                        else:
                            logger.warning(f"Step reward processor {processor_idx} for turn {i} returned unexpected result: {rewards_tensor_list}. Defaulting to 0.0.")
                            ext_step_rewards_per_turn[i][processor_idx] = 0.0 
                    except Exception as e_proc:
                        logger.error(f"Error calling step reward processor {processor_idx} for turn {i}: {e_proc}", exc_info=True)
                        ext_step_rewards_per_turn[i][processor_idx] = 0.0
                
                # Combine rewards: intrinsic + sum of step external rewards
                final_reward = intrinsic_reward_for_turn
                for step_ext_reward in ext_step_rewards_per_turn[i]:
                    final_reward += step_ext_reward
                
                # Distribute rollout-level rewards (calculated once before this loop)
                for rollout_idx, rollout_score in enumerate(rollout_level_scores):
                    distribution_strategy = self.rollout_reward_distribution[rollout_idx]
                    if distribution_strategy == "last_step" and i == num_llm_actions - 1:
                        final_reward += rollout_score
                    elif distribution_strategy == "all_steps_average" and num_llm_actions > 0:
                        final_reward += rollout_score / num_llm_actions
                    elif distribution_strategy == "all_steps_equal":
                         final_reward += rollout_score

                # Prepare auxiliary data
                aux_data: Dict[str, Any] = {"original_step_info": current_step_info}
                aux_data["intrinsic_env_reward"] = intrinsic_reward_for_turn 
                aux_data["external_step_rewards_components"] = ext_step_rewards_per_turn[i] 
                if i == num_llm_actions -1 and rollout_level_scores: 
                     aux_data["rollout_level_rewards_total_components"] = rollout_level_scores

                processed_trajectory.append(SingleTurnDataForTrainer(
                    prompt_text=prompt_text_for_turn,
                    completion_text=completion_for_llm_turn,
                    final_combined_reward=final_reward,
                    auxiliary_data=aux_data,
                    old_per_token_logps=action_logprobs 
                ))
        logger.debug(f"[RewardCalculator._calculate_rewards_for_one_rollout] Processed trajectory with {len(processed_trajectory)} turns.")
        return processed_trajectory

    async def process_rollouts(
        self, 
        batch_of_raw_rollout_data: List[RawRolloutData]
    ) -> List[ProcessedTrajectory]: # Return a list of trajectories, one per input rollout
        """
        Processes a batch of raw rollout data objects into a list of processed trajectories.
        Handles exceptions during individual rollout processing and logs them.
        """
        if not batch_of_raw_rollout_data:
            return []

        tasks = []
        for raw_rollout_item in batch_of_raw_rollout_data:
            # Ensure _calculate_rewards_for_one_rollout is robust enough to always return ProcessedTrajectory or handle its own errors
            # by returning an empty list or a list with error markers.
            tasks.append(self._calculate_rewards_for_one_rollout(raw_rollout_item))

        # Use return_exceptions=True to get exceptions instead of having gather raise them.
        results_or_exceptions: List[Any] = await asyncio.gather(*tasks, return_exceptions=True)

        processed_trajectories_batch: List[ProcessedTrajectory] = []
        for i, res_or_exc in enumerate(results_or_exceptions):
            if isinstance(res_or_exc, Exception):
                # Log the exception with full traceback
                logger.error(
                    f"[RewardCalculator] Error calculating rewards for rollout at batch index {i}: {res_or_exc}",
                    exc_info=True
                )
                processed_trajectories_batch.append([]) 
            elif isinstance(res_or_exc, list): 
                processed_trajectories_batch.append(res_or_exc)
            else:
                logger.error(
                    f"[RewardCalculator] Unexpected result type for rollout at batch index {i}: {type(res_or_exc)}. Result: {res_or_exc}"
                )
                processed_trajectories_batch.append([])

        return processed_trajectories_batch

# End of RewardCalculator class and process_rollouts method. 
# The comments below regarding RewardFunctionConfig are historical and can be removed. 