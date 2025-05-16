import json
import importlib # Added for dynamic provider loading
from typing import List, Dict, Any, Optional, Tuple, Literal
import torch # Added for logprobs
from loguru import logger # Added for logging
import re

# Retrain specific imports
from retrain.environment.environment import Environment
from retrain.environment.types import LLMAction, EnvironmentObservation, ToolCallAction, ToolResultObservation
from retrain.environment.tool.registry import get_tool_info
from retrain.environment.tool.tool import Tool as RetrainTool # Alias to avoid confusion
from retrain.environment.tool.provider.tool_provider import ToolProvider # Added for tool providers
from retrain.utils.parser.smol_agent_parser import SmolAgentParser, ParsedSmolLLMOutput
from retrain.reward.types import RawRolloutData # Import RawRolloutData

# Placeholder for actual model and tokenizer types from retrain.model
ModelObject = Any 
TokenizerObject = Any
# Placeholder for SamplingParams if retrain defines it, otherwise use a Dict
SamplingParams = Dict[str, Any]


MAX_STEPS_DEFAULT = 10
DEFAULT_PROMPT = "What do you want to do?"

class SmolAgentEnv(Environment):
    """
    An environment that simulates a smol-agent-like interaction where an LLM
    can use tools over multiple turns. It uses SmolAgentParser to parse LLM outputs 
    and retrain's tool system (registry and providers) to execute tools.

    The environment maintains the state of the conversation. An 'action' from the
    LLM is its textual response. The environment processes this, potentially
    calls a tool, and the resulting observation prompts the LLM for its next action.
    """

    def __init__(self,
                 env_specific_config: Dict[str, Any], # Changed from tool_names, max_steps
                 **kwargs: Any):
        super().__init__(**kwargs) # Pass kwargs to base Environment constructor
        
        self._env_config = env_specific_config
        self.parser = SmolAgentParser()
        
        # Tools will be loaded in the async setup() method
        self.active_tools: Dict[str, RetrainTool] = {}
        # Schemas for prompting the LLM about available tools
        self.tool_schemas_for_prompt: List[Dict[str, Any]] = [] 
        
        # Configuration for the environment's behavior
        self.max_steps: int = self._env_config.get("max_steps", MAX_STEPS_DEFAULT)
        self._initial_prompt_template: str = self._env_config.get("initial_prompt_template", DEFAULT_PROMPT)
        
        # Internal state
        self._conversation_history: List[Dict[str, str]] = []
        self._current_step = 0
        
        logger.debug(f"[SmolAgentEnv] Initialized. Max steps: {self.max_steps}. Tools will be loaded in setup().")

    async def setup(self) -> None:
        """
        Asynchronously sets up the environment, primarily by loading tools
        based on the 'tools' section of the environment_specific_config.
        """
        logger.debug("[SmolAgentEnv.setup] Starting environment setup...")
        tools_config: Dict[str, Any] = self._env_config.get("tools", {})
        
        loaded_tool_names = set()

        registry_keys: List[str] = tools_config.get("registry_keys", [])
        tool_override_configs: Dict[str, Dict[str, Any]] = tools_config.get("tool_configs", {})
        
        logger.debug(f"[SmolAgentEnv.setup] Attempting to load tools from registry: {registry_keys}")
        for key in registry_keys:
            try:
                tool_class, registered_name, registered_description = get_tool_info(key)
                specific_config = tool_override_configs.get(key, {})
                tool_instance = tool_class(name=registered_name, description=registered_description, **specific_config)
                
                if tool_instance.name in self.active_tools:
                    logger.warning(f"[SmolAgentEnv.setup] Tool '{tool_instance.name}' (from registry key '{key}') is overwriting an existing tool with the same name.")
                
                self.active_tools[tool_instance.name] = tool_instance
                try:
                    tool_schema = await tool_instance.get_schema()
                    self.tool_schemas_for_prompt.append(tool_schema)
                    logger.debug(f"[SmolAgentEnv.setup] Successfully loaded tool '{tool_instance.name}' from registry and fetched its schema.")
                except Exception as e_schema:
                    logger.error(f"[SmolAgentEnv.setup] Failed to get schema for tool '{tool_instance.name}' from registry: {e_schema}. Using basic info.")
                    self.tool_schemas_for_prompt.append({
                        "name": tool_instance.name, 
                        "description": tool_instance.description,
                        "error_getting_schema": str(e_schema)
                    })
                loaded_tool_names.add(tool_instance.name)
            except KeyError:
                logger.error(f"[SmolAgentEnv.setup] Tool key '{key}' not found in registry. Skipping.")
            except Exception as e:
                logger.error(f"[SmolAgentEnv.setup] Error instantiating tool '{key}' from registry: {e}")

        provider_configs: List[Any] = tools_config.get("providers", [])
        logger.debug(f"[SmolAgentEnv.setup] Attempting to load tools from {len(provider_configs)} providers.")
        
        for i, provider_conf_item in enumerate(provider_configs):
            provider_instance: Optional[ToolProvider] = None
            provider_id_for_logging = f"provider_config_index_{i}"

            try:
                if isinstance(provider_conf_item, ToolProvider):
                    provider_instance = provider_conf_item
                    provider_id_for_logging = f"pre_instantiated_{type(provider_instance).__name__}"
                    logger.info(f"[SmolAgentEnv.setup] Using pre-instantiated tool provider: {provider_id_for_logging}")
                elif isinstance(provider_conf_item, dict):
                    provider_class_path = provider_conf_item.get("provider_type")
                    provider_init_args = provider_conf_item.get("config", {})
                    provider_id_for_logging = provider_class_path or provider_id_for_logging

                    if not provider_class_path:
                        logger.error(f"[SmolAgentEnv.setup] Tool provider config at index {i} is a dict but missing 'provider_type'. Skipping.")
                        continue
                    
                    logger.debug(f"[SmolAgentEnv.setup] Attempting to load tool provider from class path: {provider_class_path}")
                    module_path, class_name = provider_class_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    ProviderClass = getattr(module, class_name)
                    
                    if not issubclass(ProviderClass, ToolProvider):
                        logger.error(f"[SmolAgentEnv.setup] Class '{provider_class_path}' is not a subclass of ToolProvider. Skipping.")
                        continue
                        
                    provider_instance = ProviderClass(**provider_init_args)
                    logger.debug(f"[SmolAgentEnv.setup] Instantiated tool provider '{provider_class_path}' with config: {provider_init_args}")
                else:
                    logger.error(f"[SmolAgentEnv.setup] Invalid item in 'providers' list at index {i}: {type(provider_conf_item)}. Must be a ToolProvider instance or a config dict. Skipping.")
                    continue

                if provider_instance:
                    discovered_tools = await provider_instance.discover_tools()
                    logger.debug(f"[SmolAgentEnv.setup] Provider '{provider_id_for_logging}' discovered {len(discovered_tools)} tools.")
                    for tool_instance in discovered_tools:
                        if not isinstance(tool_instance, RetrainTool):
                            logger.warning(f"[SmolAgentEnv.setup] Item discovered by provider '{provider_id_for_logging}' is not a RetrainTool instance: {type(tool_instance)}. Skipping.")
                            continue
                        
                        if tool_instance.name in self.active_tools:
                            logger.warning(f"[SmolAgentEnv.setup] Tool '{tool_instance.name}' (from provider '{provider_id_for_logging}') is overwriting an existing tool.")
                        
                        self.active_tools[tool_instance.name] = tool_instance
                        try:
                            tool_schema = await tool_instance.get_schema()
                            self.tool_schemas_for_prompt.append(tool_schema)
                            logger.debug(f"[SmolAgentEnv.setup] Successfully loaded tool '{tool_instance.name}' from provider '{provider_id_for_logging}' and fetched schema.")
                        except Exception as e_schema:
                            logger.error(f"[SmolAgentEnv.setup] Failed to get schema for tool '{tool_instance.name}' from provider '{provider_id_for_logging}': {e_schema}. Using basic info.")
                            self.tool_schemas_for_prompt.append({
                                "name": tool_instance.name,
                                "description": tool_instance.description,
                                "error_getting_schema": str(e_schema)
                            })
                        loaded_tool_names.add(tool_instance.name)
            except ImportError as e:
                logger.error(f"[SmolAgentEnv.setup] Failed to import module or class for provider '{provider_id_for_logging}': {e}. Skipping provider.")
            except AttributeError as e: 
                logger.error(f"[SmolAgentEnv.setup] Failed to get class for provider '{provider_id_for_logging}': {e}. Skipping provider.")
            except TypeError as e: 
                logger.error(f"[SmolAgentEnv.setup] TypeError instantiating provider '{provider_id_for_logging}': {e}. Skipping provider.")
            except Exception as e:
                logger.error(f"[SmolAgentEnv.setup] Error processing tool provider '{provider_id_for_logging}': {e}. Skipping provider.")
        
        logger.info(f"[SmolAgentEnv.setup] Setup complete. Total active tools loaded: {len(self.active_tools)}. Names: {list(self.active_tools.keys()) if len(self.active_tools) < 10 else str(len(self.active_tools)) + ' tools'}")
        if not self.active_tools:
            logger.warning("[SmolAgentEnv.setup] No tools were loaded into SmolAgentEnv. The agent will not be able to use any tools.")

    def _get_formatted_tool_descriptions(self) -> str:
        """
        Formats the schemas of available tools into a JSON string for the LLM prompt.
        """
        if not self.tool_schemas_for_prompt:
            return "No tools available."
        
        try:
            return json.dumps(self.tool_schemas_for_prompt, indent=2)
        except Exception as e:
            logger.error(f"[SmolAgentEnv] Error formatting tool schemas to JSON: {e}")
            fallback_desc = []
            for schema in self.tool_schemas_for_prompt:
                name = schema.get("name", "Unknown tool")
                desc = schema.get("description", "No description available.")
                fallback_desc.append(f"- {name}: {desc}")
            return "Error serializing tool schemas. Basic list:\n" + "\n".join(fallback_desc)

    async def _execute_tool(self, tool_name: str, tool_args_dict: Optional[Dict[str, Any]]) -> ToolResultObservation:
        """Executes a tool and returns its string output."""
        if tool_name not in self.active_tools:
            # Warning is suitable here, as the LLM might hallucinate tools.
            logger.warning(f"[SmolAgentEnv._execute_tool] Attempted to execute unavailable tool: '{tool_name}'")
            return ToolResultObservation(
                tool_name=tool_name, 
                tool_output=f"Error: Tool '{tool_name}' is not available or not recognized.",
                status="error",
                error_message=f"Tool '{tool_name}' not available."
            )
        
        tool_instance = self.active_tools[tool_name]
        try:
            logger.debug(f"[SmolAgentEnv._execute_tool] Executing tool '{tool_name}' with args: {tool_args_dict}")
            result = await tool_instance.execute(tool_input=tool_args_dict)
            logger.debug(f"[SmolAgentEnv._execute_tool] Tool '{tool_name}' executed successfully. Result snippet: {str(result)[:100]}...")
            return ToolResultObservation(
                tool_name=tool_name,
                tool_output=str(result),
                status="success",
                error_message=None
            )
        except Exception as e:
            logger.error(f"[SmolAgentEnv._execute_tool] Error during tool '{tool_name}' execution with args {tool_args_dict}: {e}", exc_info=True)
            error_msg = f"Error: Tool {tool_name} failed with error: {str(e)}"
            return ToolResultObservation(
                tool_name=tool_name,
                tool_output=error_msg, 
                status="error",
                error_message=str(e) 
            )

    async def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[EnvironmentObservation, Dict[str, Any]]:
        """
        Resets the environment and starts a new conversation.
        The initial prompt can be passed via options['initial_prompt'] or from env_specific_config.
        """
        await super().reset(seed=seed, options=options)

        self._current_step = 0
        self._conversation_history = []
        
        initial_user_prompt = self._initial_prompt_template
        if options and "initial_prompt" in options and isinstance(options["initial_prompt"], str):
            initial_user_prompt = options["initial_prompt"]
        
        self._conversation_history.append({"role": "user", "content": initial_user_prompt})
        
        current_available_tools_desc = list(self.tool_schemas_for_prompt)

        observation = EnvironmentObservation(
            observation_type="initial",
            content=None, 
            tool_result=None,
            current_conversation=list(self._conversation_history),
            available_tools=current_available_tools_desc, 
            requires_llm_action=True
        )
        info = {"message": "Environment reset, initial prompt provided."}
        logger.info(f"[SmolAgentEnv] Environment Reset. Initial prompt snippet: '{initial_user_prompt[:100]}...'. Tools available: {len(current_available_tools_desc)}")
        return observation, info

    async def step(self, action: LLMAction) -> Tuple[EnvironmentObservation, float, bool, bool, Dict[str, Any]]:
        """
        Processes the LLM's action (its textual response), potentially calls a tool,
        and returns the next state.
        """
        if not isinstance(action, dict) or "action_type" not in action or "raw_llm_output" not in action:
            logger.error(f"[SmolAgentEnv.step] Invalid action structure received: {action}")
            # This is a critical error in how actions are formed and an error observation is returned to the LLM.
            error_msg_for_llm = self.parser.format_environment_error("Internal error: Invalid action structure received by environment.")
            self._conversation_history.append({"role": "user", "content": error_msg_for_llm})
            
            obs = EnvironmentObservation(
                observation_type="environment_state", 
                content=error_msg_for_llm,
                tool_result=None,
                current_conversation=list(self._conversation_history),
                available_tools=list(self.tool_schemas_for_prompt),
                requires_llm_action=True # Allow LLM to try again
            )
            # Returns: obs, no reward, not terminated, not truncated, error info
            return obs, 0.0, False, False, {"step_error": "Invalid LLMAction structure", "received_action": action}

        self._current_step += 1
        self._conversation_history.append({"role": "assistant", "content": action["raw_llm_output"]})

        terminated = False
        truncated = False
        reward = 0.0  
        info: Dict[str, Any] = {"action_received": action} 
        
        current_observation_type: Literal["tool_result", "environment_state", "final_answer_feedback"] = "environment_state"
        tool_result_obs: Optional[ToolResultObservation] = None
        environment_content: Optional[Any] = None

        if action["action_type"] == "tool_call":
            tool_call_data = action.get("tool_call")
            if not tool_call_data or not isinstance(tool_call_data.get("tool_name"), str):
                error_message_for_llm = self.parser.format_environment_error("Invalid tool_call structure in your action.")
                self._conversation_history.append({"role": "user", "content": error_message_for_llm})
                environment_content = error_message_for_llm
                info["step_error"] = "Invalid tool_call structure in LLMAction"
            else:
                tool_name = tool_call_data["tool_name"]
                tool_input = tool_call_data.get("tool_input")
                
                logger.debug(f"[SmolAgentEnv.step] Executing tool: {tool_name} with input: {tool_input}")
                tool_result_obs = await self._execute_tool(tool_name, tool_input)
                current_observation_type = "tool_result"
                
                if tool_result_obs["status"] == "success":
                    formatted_tool_output = self.parser.format_successful_tool_result(str(tool_result_obs["tool_output"]))
                else:
                    formatted_tool_output = self.parser.format_tool_error_result(str(tool_result_obs["tool_output"]))
                self._conversation_history.append({"role": "user", "content": formatted_tool_output})
                info["tool_executed"] = tool_name
                info["tool_result_status"] = tool_result_obs["status"]
        
        elif action["action_type"] == "text_response":
            # The LLMAction's "raw_llm_output" is processed here.
            # The parser, when initially creating the LLMAction, might have extracted fields like reasoning/final_answer.
            # However, this step needs to confirm if a final answer is present to terminate the episode.
            # The LLMAction structure itself (derived from parser output) doesn't explicitly flag "is_final_answer",
            # so re-parsing the raw output is the most reliable way to check for a final answer tag here.
            parsed_output = self.parser.parse(action["raw_llm_output"])

            if parsed_output.final_answer is not None:
                terminated = True
                current_observation_type = "final_answer_feedback"
                environment_content = f"Final answer received: {parsed_output.final_answer}"
                info["final_answer_detected"] = parsed_output.final_answer
                logger.info(f"[SmolAgentEnv.step] Final answer detected.")
            elif parsed_output.parsing_error and not parsed_output.tool_command: # Error but not a tool
                # This means the LLM tried to say something, but it wasn't a tool or final answer, and had parsing issues.
                error_message_for_llm = self.parser.format_environment_error(
                    f"Parsing error in your response: {parsed_output.parsing_error}"
                )
                self._conversation_history.append({"role": "user", "content": error_message_for_llm})
                environment_content = error_message_for_llm
                info["parsing_error_in_text_response"] = parsed_output.parsing_error
                logger.warning(f"[SmolAgentEnv.step] Parsing error in LLM text response: {parsed_output.parsing_error}")
            else:
                # Regular text response (e.g. reasoning, thought), not a final answer.
                # The LLM's text is already in conversation_history.
                environment_content = "LLM provided a text response (thought/reasoning)."
                logger.debug("[SmolAgentEnv.step] LLM provided a text response (thought/reasoning).")

        else:
            unknown_action_msg = self.parser.format_environment_error(f"Unknown action_type: {action.get('action_type')}")
            self._conversation_history.append({"role": "user", "content": unknown_action_msg})
            environment_content = unknown_action_msg
            info["step_error"] = f"Unknown action_type: {action.get('action_type')}"
            logger.error(f"[SmolAgentEnv.step] Unknown action_type encountered: {action.get('action_type')}")

        requires_llm_action_next = True 
        if terminated or (self._current_step >= self.max_steps):
            truncated = not terminated 
            requires_llm_action_next = False 
            if truncated:
                 info["truncation_reason"] = "max_steps_reached"
                 logger.info(f"[SmolAgentEnv.step] Episode truncated at step {self._current_step} due to max_steps.")
            else:
                 logger.info(f"[SmolAgentEnv.step] Episode terminated at step {self._current_step}.")

        # Use self.tool_schemas_for_prompt, which is populated during setup()
        # The EnvironmentObservation type expects List[Dict[str,Any]] for available_tools
        # and self.tool_schemas_for_prompt is already in this format.
        current_available_tools_desc = list(self.tool_schemas_for_prompt)

        observation = EnvironmentObservation(
            observation_type=current_observation_type,
            content=environment_content,
            tool_result=tool_result_obs,
            current_conversation=list(self._conversation_history),
            available_tools=current_available_tools_desc,
            requires_llm_action=requires_llm_action_next
        )
        
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        print("\n--- SmolAgentEnv Conversation ---")
        for i, message in enumerate(self._conversation_history):
            print(f"[{i:02d}] {message['role']}: {message['content']}")
        print("-------------------------------")

    def close(self) -> None:
        # Cleanup if needed, e.g., close any resources held by tools if they have them.
        logger.info("[SmolAgentEnv] Closed.")
        pass

    async def rollout(
        self,
        initial_prompt: str, # Or List[Dict[str, str]] for initial messages
        llm_model: ModelObject, # Actual loaded model object
        tokenizer: TokenizerObject, # Tokenizer for the model
        sampling_params: SamplingParams,
        # max_tokens_to_generate: Optional[int] = None, # Now expected in sampling_params
        # temperature_for_logprobs: float = 0.7 # Now expected in sampling_params
    ) -> RawRolloutData:
        """
        Performs a full rollout in the environment using the provided LLM until the episode ends.
        This implementation calculates old_per_token_logps for the entire completion.
        """
        logger.debug("[SmolAgentEnv.rollout] Starting rollout...")
        
        # Prepare for generation if tokenizer has common methods, otherwise model might handle it
        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                logger.debug(f"[SmolAgentEnv.rollout] Tokenizer pad_token_id not set, using eos_token_id: {tokenizer.eos_token_id}")
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                logger.warning("[SmolAgentEnv.rollout] Tokenizer pad_token_id and eos_token_id are not set. Adding a PAD token.")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.debug(f"[SmolAgentEnv.rollout] New pad_token_id: {tokenizer.pad_token_id}")

        # If the model is a Hugging Face model, it likely has a .device attribute
        device = getattr(llm_model, 'device', 'cpu')
        logger.debug(f"[SmolAgentEnv.rollout] Using device: {device} for LLM operations.")

        obs_list: List[EnvironmentObservation] = []
        action_list: List[LLMAction] = []
        reward_list: List[float] = []
        info_dicts_per_step: List[Dict[str, Any]] = []

        current_observation, _ = await self.reset(options={"initial_prompt": initial_prompt})
        obs_list.append(current_observation)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {} # Initialize info

        # Determine max_tokens_for_this_generation
        # Priority: sampling_params -> env_config -> default
        max_tokens_from_sampling_params = sampling_params.get("max_tokens_to_generate")
        temperature_for_logprobs = sampling_params.get("temperature_for_logprobs", 0.7) # Get temp for logprobs

        if max_tokens_from_sampling_params is not None:
            max_tokens_for_this_generation = max_tokens_from_sampling_params
            logger.debug(f"[SmolAgentEnv.rollout] Using max_tokens_to_generate = {max_tokens_for_this_generation} (from sampling_params)")
        else:
            max_tokens_for_this_generation = self._env_config.get("max_tokens_per_llm_turn", 256)
            logger.debug(f"[SmolAgentEnv.rollout] Using max_tokens_to_generate = {max_tokens_for_this_generation} (from env_config or default)")

        logger.debug(f"[SmolAgentEnv.rollout] Entering main generation loop. Terminated: {terminated}, Truncated: {truncated}")
        while not (terminated or truncated):
            if current_observation["requires_llm_action"]:
                # 1. Format conversation for LLM input
                #    HF tokenizers typically need a string or list of strings.
                #    The chat template (if available) is the best way to format this.
                
                # Construct the prompt that includes tool descriptions
                # This is a critical part for enabling tool use.
                formatted_tools_string = self._get_formatted_tool_descriptions()
                
                # System prompt defining tool usage and output format expectations for the LLM.
                system_prompt_content = (
                    "You are a helpful assistant. Your response MUST ALWAYS START with <reasoning>, <tool>, or <answer>. NO other text is allowed before these starting tags."
                    + " To use a tool, you MUST output the tool name and arguments as a JSON string enclosed \n"
                    + "EXACTLY within <tool> and </tool> tags. For example:\n"
                    + "<tool>{{\"name\": \"tool_name_here\", \"args\": {{\"param1\": \"value1\", \"param2\": 123}}}}</tool>\n"
                    + "Do NOT put any other text before or after the <tool>...</tool> block if you are calling a tool (except for the JSON arguments themselves which follow the closing </tool> tag if the format implies that, though the example shows it inside).\n"
                    + "The JSON inside <tool> must have a \"name\" key for the tool and an optional \"args\" key for its arguments.\n"
                    + "Available tools are listed below. Only use these exact tool names.\n"
                    + f"{formatted_tools_string}\n\n"
                    + "If you are providing a final answer, use the <answer>YOUR_FINAL_ANSWER_HERE</answer> tag. Your response MUST start with this tag if it's a final answer.\n"
                    + "For intermediate reasoning or thoughts BEFORE deciding on a tool or an answer, use <reasoning>YOUR_THOUGHTS_HERE</reasoning>. Your response MUST start with this tag if you are just thinking. Any thinking or deliberation MUST be enclosed in <reasoning> tags if it is not part of a direct tool call or final answer structure.\n"
                    + "Tool results provided by the system will be in <result>...</result> tags.\n"
                )

                # Create a temporary conversation history for prompting, injecting the system prompt with tools.
                # Filter out any prior system messages to avoid duplication.
                prompting_conversation = [{"role": "system", "content": system_prompt_content}]
                for msg in current_observation["current_conversation"]:
                    if msg.get("role") != "system":
                        prompting_conversation.append(msg)

                try:
                    prompt_text_for_llm = tokenizer.apply_chat_template(
                        prompting_conversation,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    logger.debug(f"[SmolAgentEnv.rollout] Generated prompt for LLM using chat template (length {len(prompt_text_for_llm)}). Tools included: {formatted_tools_string[:100]}...")
                except Exception as e_template:
                    logger.warning(f"[SmolAgentEnv.rollout] Failed to apply chat template: {e_template}. Falling back to basic formatting.")
                    # Basic fallback formatting if chat template fails
                    formatted_messages = [f"{msg['role']}: {msg['content']}" for msg in prompting_conversation]
                    prompt_text_for_llm = "\n".join(formatted_messages)
                    if getattr(tokenizer, "bos_token", None):
                        prompt_text_for_llm = tokenizer.bos_token + prompt_text_for_llm
                    # add_generation_prompt=True behavior for some models is to append assistant role like "role: assistant\ncontent: "
                    # This simple join doesn't do that, so LLM might need to be robust.

                # 2. LLM generates text response.
                # Assumes a Hugging Face like model.generate() API.
                inputs = tokenizer(prompt_text_for_llm, return_tensors="pt", padding=True).to(device)
                
                # Ensure sampling_params are compatible with model.generate()
                # We remove our custom keys before passing to model.generate()
                model_generate_sampling_params = { 
                    k: v for k, v in sampling_params.items() 
                    if k not in ["max_tokens_to_generate", "temperature_for_logprobs"]
                }

                generate_kwargs = {
                    "max_new_tokens": max_tokens_for_this_generation,
                    "pad_token_id": tokenizer.pad_token_id,
                    **model_generate_sampling_params
                }
                raw_llm_text_output = "" # Initialize with a default
                output_sequences_obj = None # Initialize to avoid UnboundLocalError if model doesn't have generate

                if hasattr(llm_model, "generate"):
                    # Ensure scores are returned for logprob calculation
                    generate_kwargs_with_scores = {
                        **generate_kwargs,
                        "output_scores": True,
                        "return_dict_in_generate": True 
                    }
                    logger.debug(f"[SmolAgentEnv.rollout] Calling model.generate with kwargs (excluding inputs): {generate_kwargs_with_scores}")
                    with torch.no_grad(): # Ensure no gradients are computed during inference
                        output_sequences_obj = llm_model.generate(
                            **inputs, 
                            **generate_kwargs_with_scores
                        )
                    
                    # Decode only the generated part of the sequences.
                    # .sequences attribute is available if return_dict_in_generate=True was used.
                    generated_tokens_for_parse = output_sequences_obj.sequences[0, inputs.input_ids.shape[1]:]
                    raw_llm_text_output = tokenizer.decode(generated_tokens_for_parse, skip_special_tokens=True)
                else:
                    logger.error("[SmolAgentEnv.rollout] llm_model missing generate method. Cannot generate LLM response.")
                    # raw_llm_text_output remains empty, which will lead to parsing error downstream.

                # Pre-processing LLM output
                processed_llm_output = raw_llm_text_output.strip()
                original_processed_for_log = processed_llm_output # For logging if we strip further after <think> tags

                # 1. Strip leading <think>...</think> block or unclosed <think> tag
                if processed_llm_output.startswith("<think>"):
                    think_end_match = re.search(r"</think>", processed_llm_output)
                    if think_end_match:
                        processed_llm_output = processed_llm_output[think_end_match.end():].strip()
                        logger.debug(f"[SmolAgentEnv.rollout] Stripped complete leading <think>...</think> block. Output for parsing: '{processed_llm_output[:200]}...'")
                    else:
                        processed_llm_output = processed_llm_output[len("<think>"):].strip()
                        logger.warning(f"[SmolAgentEnv.rollout] Found starting <think> tag but no closing </think> tag. Stripped only the opening <think> tag. Output for parsing: '{processed_llm_output[:200]}...'")

                # 2. Strip any text before the first primary XML tag (<tool>, <reasoning>, <answer>)
                #    This helps if the LLM adds conversational prefixes before the structured output.
                primary_tags = ["<tool", "<reasoning", "<answer"]
                first_tag_index = -1
                
                indices = []
                for tag_prefix in primary_tags:
                    try:
                        indices.append(processed_llm_output.index(tag_prefix))
                    except ValueError:
                        pass # Tag not found
                
                if indices:
                    first_tag_index = min(indices)

                if first_tag_index > 0: # If a primary tag is found, but not at the beginning
                    stripped_prefix = processed_llm_output[:first_tag_index]
                    processed_llm_output = processed_llm_output[first_tag_index:]
                    logger.info(f"[SmolAgentEnv.rollout] Stripped leading text '{stripped_prefix[:100]}...' before first primary XML tag. Output for parsing: '{processed_llm_output[:200]}...'")
                elif first_tag_index == -1 and processed_llm_output and processed_llm_output != original_processed_for_log : # No primary tag, but <think> was stripped
                    logger.debug(f"[SmolAgentEnv.rollout] No primary XML tag found after <think> stripping. Original (pre-think-strip): '{original_processed_for_log[:200]}...'. Post-think-strip: '{processed_llm_output[:200]}...'")
                elif first_tag_index == -1 and processed_llm_output: # No primary tag, and no <think> was stripped (or it was empty after)
                    logger.debug(f"[SmolAgentEnv.rollout] No primary XML tag found in LLM output. Output for parsing: '{processed_llm_output[:200]}...'")

                
                logger.debug(f"[SmolAgentEnv.rollout] Feeding to parser: [START_OF_STRING]{processed_llm_output}[END_OF_STRING]")
                parsed_smol_output: ParsedSmolLLMOutput = self.parser.parse(processed_llm_output)

                # Calculate log probabilities
                action_logprobs: Optional[torch.Tensor] = None
                if output_sequences_obj and hasattr(output_sequences_obj, 'scores'):
                    # output_sequences_obj.scores is a tuple of tensors, one for each generated token
                    # Each tensor is [batch_size, vocab_size]
                    # We need to stack them and get the log_softmax
                    stacked_scores = torch.stack(output_sequences_obj.scores, dim=1) # [batch_size, seq_len, vocab_size]
                    log_probs_full_vocab = torch.nn.functional.log_softmax(stacked_scores, dim=-1)
                    
                    # Get the IDs of the generated tokens.
                    # generated_tokens_for_parse was already derived from output_sequences_obj.sequences
                    # We need it as [batch_size=1, seq_len_generated] to match log_probs_full_vocab
                    generated_ids_for_logprobs = generated_tokens_for_parse.unsqueeze(0).to(log_probs_full_vocab.device) # Ensure same device
                    
                    # Gather the log-probabilities of the actually generated tokens
                    # generated_ids_for_logprobs needs to be [batch_size, seq_len, 1] for gather
                    action_logprobs_batched = torch.gather(log_probs_full_vocab, -1, generated_ids_for_logprobs.unsqueeze(-1)).squeeze(-1)
                    
                    if action_logprobs_batched.shape[0] == 1: # Remove batch dim if 1
                        action_logprobs = action_logprobs_batched.squeeze(0).cpu()
                    else: # Should not happen with current rollout logic (batch_size=1)
                        action_logprobs = action_logprobs_batched.cpu()
                    logger.trace(f"[SmolAgentEnv.rollout] Calculated action_logprobs (shape: {action_logprobs.shape if action_logprobs is not None else 'None'}) for raw output: {raw_llm_text_output[:50]}...")
                elif output_sequences_obj is None: # Case where llm_model.generate was not called
                    logger.warning("[SmolAgentEnv.rollout] llm_model.generate was not called (e.g. missing method). Logprobs cannot be calculated.")
                else: # Case where output_sequences_obj exists but .scores is missing
                    logger.warning("[SmolAgentEnv.rollout] model.generate did not return 'scores'. Logprobs cannot be calculated.")


                # Construct LLMAction based on parsing result
                llm_action: Optional[LLMAction] = None
                if parsed_smol_output.tool_command:
                    try:
                        tool_command_dict = json.loads(parsed_smol_output.tool_command)
                        tool_name = tool_command_dict.get("name")
                        tool_input_args = tool_command_dict.get("args")
                        if isinstance(tool_name, str):
                            tool_call_action = ToolCallAction(tool_name=tool_name, tool_input=tool_input_args)
                            llm_action = LLMAction(
                                action_type="tool_call", 
                                tool_call=tool_call_action, 
                                text=parsed_smol_output.reasoning, # Include reasoning if present with tool call
                                raw_llm_output=processed_llm_output,
                                reasoning=parsed_smol_output.reasoning, # Explicitly provide reasoning
                                old_per_token_logps=action_logprobs
                            )
                        else:
                            logger.warning(f"[SmolAgentEnv.rollout] Tool name in parsed command is not a string: '{tool_name}'. Command: {parsed_smol_output.tool_command}")
                            raise ValueError("Tool name missing or not a string in parsed command.")
                    except (json.JSONDecodeError, ValueError) as e:
                        # Malformed tool command JSON or structure
                        logger.warning(f"[SmolAgentEnv.rollout] Error parsing tool command JSON: '{parsed_smol_output.tool_command}'. Error: {e}")
                        # Fallback: treat as text response with error content, or let step handle raw output
                        llm_action = LLMAction(
                            action_type="text_response",
                            text=f"Parsing Error: Your attempt to call a tool was not formatted correctly. The command was: {processed_llm_output}. Please ensure the <tool> tag contains valid JSON with 'name' and 'args'. Error details: {e}",
                            tool_call=None,
                            raw_llm_output=processed_llm_output,
                            reasoning=None, # Explicitly None
                            old_per_token_logps=action_logprobs
                        )
                elif parsed_smol_output.final_answer is not None:
                    llm_action = LLMAction(
                        action_type="text_response", 
                        text=parsed_smol_output.final_answer, 
                        tool_call=None,
                        raw_llm_output=processed_llm_output,
                        reasoning=parsed_smol_output.reasoning, # Pass reasoning if present with final answer
                        old_per_token_logps=action_logprobs
                    )
                elif parsed_smol_output.reasoning is not None:
                    llm_action = LLMAction(
                        action_type="text_response",
                        text=parsed_smol_output.reasoning, # This is just a thought/reasoning step
                        tool_call=None,
                        raw_llm_output=processed_llm_output,
                        reasoning=parsed_smol_output.reasoning, # Reasoning is the main content here
                        old_per_token_logps=action_logprobs
                    )
                elif parsed_smol_output.parsing_error:
                     # If parser explicitly found an error and nothing else, pass raw output to step for error handling
                    logger.warning(f"[SmolAgentEnv.rollout] SmolAgentParser returned parsing_error: '{parsed_smol_output.parsing_error}' for output: {processed_llm_output}")
                    llm_action = LLMAction(
                        action_type="text_response", 
                        text=processed_llm_output, 
                        tool_call=None, 
                        raw_llm_output=processed_llm_output,
                        reasoning=None, # No specific reasoning if parsing error
                        old_per_token_logps=action_logprobs
                    )
                else: # Empty or unparseable by SmolAgentParser into distinct fields
                    logger.info(f"[SmolAgentEnv.rollout] LLM output was not parsed into specific smol-agent fields (tool/answer/reasoning). Treating as simple text response: {processed_llm_output[:100]}...")
                    llm_action = LLMAction(
                        action_type="text_response",
                        text=processed_llm_output, # Pass through raw if no specific structure found
                        tool_call=None,
                        raw_llm_output=processed_llm_output,
                        reasoning=None, # No specific reasoning
                        old_per_token_logps=action_logprobs
                    )
                
                action_list.append(llm_action)
                # Environment processes this structured action
                next_observation, reward, terminated, truncated, info = await self.step(llm_action)
                
                obs_list.append(next_observation)
                reward_list.append(reward)
                info_dicts_per_step.append(info)
                current_observation = next_observation
            else:
                # Should not happen if requires_llm_action is managed correctly in step()
                logger.warning("[SmolAgentEnv.rollout] Loop continued but environment does not require LLM action. Breaking.")
                break
        
        logger.debug(f"[SmolAgentEnv.rollout] Exited main generation loop. Terminated: {terminated}, Truncated: {truncated}, Current Step: {self._current_step}")
        # Add final observation and info if loop exited due to requires_llm_action = False
        if not current_observation.get("requires_llm_action", True) and (terminated or truncated):
             obs_list.append(current_observation)
             info_dicts_per_step.append(info)

        for action_data in action_list:
            if "old_per_token_logps" not in action_data:
                action_data["old_per_token_logps"] = None # Explicitly set if missing
            if "reasoning" not in action_data: # Ensure reasoning is present, even if None
                 action_data["reasoning"] = None


        final_env_obs: Optional[EnvironmentObservation] = obs_list[-1] if obs_list else None
        if final_env_obs is None:
            # This case should ideally not happen if a rollout produces at least one observation.
            # Creating a minimal fallback if it does.
            logger.warning("[SmolAgentEnv.rollout] obs_list is empty, creating a default final_environment_observation.")
            # Attempt to use current_observation if available, otherwise create a very basic one.
            if 'current_observation' in locals() and current_observation:
                final_env_obs = current_observation
            else: # Fallback to a truly minimal structure if current_observation is also somehow unavailable
                final_env_obs = EnvironmentObservation(
                    observation_type="environment_state", 
                    content="Rollout ended with no observations.",
                    tool_result=None,
                    current_conversation=list(self._conversation_history), # Use the latest history
                    available_tools=list(self.tool_schemas_for_prompt),
                    requires_llm_action=False 
                )
        
        logger.debug(f"[SmolAgentEnv.rollout] Preparing RawRolloutData. Length of action_list: {len(action_list)}")
        # Construct and return RawRolloutData dictionary
        raw_rollout_result: RawRolloutData = {
            "full_conversation_history": self._conversation_history, # Use the final state of internal history
            "executed_llm_actions": action_list,
            "intrinsic_rewards_per_turn": reward_list,
            "final_environment_observation": final_env_obs,
            "step_info_dicts_per_turn": info_dicts_per_step,
            "rollout_metadata": {} # Add empty dict for now, can be populated later if needed
        }
        logger.info("[SmolAgentEnv.rollout] Rollout complete.")
        return raw_rollout_result