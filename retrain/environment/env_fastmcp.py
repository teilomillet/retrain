# Standard library imports
import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING, Literal

# Retrain library imports
from .environment import Environment
from .types import LLMAction, EnvironmentObservation, ToolCallAction, ToolResultObservation

# Corrected and more explicit imports
from .tool.tool import Tool
from .tool.error import ToolExecutionError
from .tool.registry import get_tool_info
from .tool.provider import ToolProvider

# Assuming FastMCPToolProvider will be the primary provider example here
from .tool.provider import FastMCPToolProvider, ManualToolProvider 
from retrain.reward.types import RawRolloutData # For rollout return type
from retrain.utils.parser.smol_agent_parser import SmolAgentParser, ParsedSmolLLMOutput # For parsing LLM output
import torch # For Optional[torch.Tensor] in RawRolloutData and logprobs
from loguru import logger # For logging within rollout
import json # For formatting tool descriptions

# Try to import FastMCPClient for dynamic provider setup
try:
    from fastmcp.client import Client as FastMCPClientTypeInternal
except ImportError:
    FastMCPClientTypeInternal = None
    logger.warning("fastmcp.client.Client could not be imported. FastMCPToolProvider cannot be auto-initialized by FastMCPEnv if server_url is given without an explicit provider instance.")

if TYPE_CHECKING:
    # Placeholder for actual FastMCPClient type to avoid runtime dependency if not installed
    # In a real setup, you'd do: from fastmcp.client import Client as FastMCPClientType
    FastMCPClientType = Any
    # Placeholder for actual model and tokenizer types from retrain.model
    ModelObject = Any 
    TokenizerObject = Any
    # Placeholder for SamplingParams if retrain defines it, otherwise use a Dict
    SamplingParams = Dict[str, Any]

#  Example User-Defined Tool (Registered via @tool)
# @tool # Name will be derived as 'simple_calculator_tool', description from docstring.
# class SimpleCalculatorTool(Tool):
#     """A basic calculator tool that adds two numbers x and y."""
#     
#     # BaseTool's __init__ takes name and description. The @tool decorator handles providing these
#     # to the environment when it instantiates the tool using get_tool_info.
#     # If this tool needed its own specific config, it would define its own __init__ like:
#     # def __init__(self, name: str, description: str, precision: int = 2):
#     #     super().__init__(name, description)
#     #     self.precision = precision
# 
#     async def execute(self, tool_input: Optional[Dict[str, Any]]) -> Any:
#         if tool_input is None or 'x' not in tool_input or 'y' not in tool_input:
#             raise ToolExecutionError("Input must include 'x' and 'y' for addition.")
#         try:
#             x = float(tool_input['x'])
#             y = float(tool_input['y'])
#             result = x + y
#             # Example of returning a structured result, could also be just the number
#             return {"sum": result, "inputs": {"x": x, "y": y}}
#         except ValueError as e:
#             raise ToolExecutionError(f"Invalid number input for addition: {e}") from e
#         except Exception as e:
#             raise ToolExecutionError(f"Calculator tool error: {e}") from e

# Main Environment Definition
class FastMCPEnv(Environment):
    """
    An environment designed to interact with a FastMCP server, demonstrating
    the use of dynamically discovered tools (from FastMCP) and registered tools.
    """
    def __init__(
        self,
        tool_registry_keys: Optional[List[str]] = None,
        tool_configs: Optional[Dict[str, Any]] = None,
        tool_providers: Optional[List[ToolProvider]] = None,
        # Add any other environment-specific parameters here
        initial_prompt_template: str = "Welcome! What would you like to do?",
        max_steps: int = 10, # Added max_steps for rollout termination
        server_url: Optional[str] = None # Added server_url
    ):
        super().__init__() # BaseEnvironment __init__ currently does nothing with kwargs
        
        self._tool_registry_keys = tool_registry_keys if tool_registry_keys is not None else []
        self._tool_configs = tool_configs if tool_configs is not None else {}
        self._tool_providers = list(tool_providers) if tool_providers is not None else [] # Ensure it's a list
        
        self.active_tools: Dict[str, Tool] = {}
        self.tool_schemas_for_prompt: List[Dict[str, Any]] = [] # For storing tool schemas
        self.initial_prompt_template = initial_prompt_template
        self.max_steps = max_steps # Store max_steps
        self.current_turn = 0 # Renamed from _current_step for consistency with existing code
        self._conversation_history: List[Dict[str, str]] = [] # For storing conversation
        self.parser = SmolAgentParser() # Initialize parser
        self.server_url = server_url # Store server_url

        # If server_url is provided and no FastMCPToolProvider is in the list, try to add one.
        # This part is moved to setup() to allow async client instantiation if needed,
        # and to keep __init__ synchronous.
        # For now, we'll prepare it here if the client import was successful.

    async def setup(self):
        """
        Asynchronously initializes and loads all tools for the environment.
        This method should be called after the environment is instantiated.
        """
        logger.debug("[FastMCPEnv] Starting environment setup...")
        self.active_tools = {}
        self.tool_schemas_for_prompt = []

        if self.server_url and FastMCPClientTypeInternal:
            has_fmp_provider = any(isinstance(p, FastMCPToolProvider) for p in self._tool_providers)
            if not has_fmp_provider:
                logger.info(f"[FastMCPEnv] server_url '{self.server_url}' provided and no FastMCPToolProvider found. Attempting to auto-initialize one.")
                try:
                    mcp_client = FastMCPClientTypeInternal(transport=self.server_url)
                    auto_provider = FastMCPToolProvider(client=mcp_client)
                    self._tool_providers.append(auto_provider)
                    logger.info(f"[FastMCPEnv] Auto-initialized FastMCPToolProvider for server: {self.server_url}")
                except Exception as e_client:
                    logger.error(f"[FastMCPEnv] Failed to auto-initialize FastMCPToolProvider with server_url '{self.server_url}': {e_client}")
            elif has_fmp_provider:
                 logger.debug(f"[FastMCPEnv] FastMCPToolProvider already present. Skipping auto-initialization for server_url: {self.server_url}")
        elif self.server_url and not FastMCPClientTypeInternal:
            logger.warning(f"[FastMCPEnv] server_url '{self.server_url}' was provided, but FastMCPClient could not be imported. Cannot auto-initialize FastMCPToolProvider.")

        for tool_key in self._tool_registry_keys:
            try:
                ToolClass, reg_name, reg_desc = get_tool_info(tool_key)
                config = self._tool_configs.get(tool_key, {})
                tool_instance = ToolClass(name=reg_name, description=reg_desc, **config)
                
                if reg_name in self.active_tools:
                    logger.warning(f"[FastMCPEnv] Tool '{reg_name}' (from registry) overwriting existing tool.")
                self.active_tools[reg_name] = tool_instance
                try:
                    schema = await tool_instance.get_schema()
                    self.tool_schemas_for_prompt.append(schema)
                    logger.debug(f"[FastMCPEnv] Loaded registered tool: {reg_name}") 
                except Exception as e_schema:
                    logger.error(f"[FastMCPEnv] Failed to get schema for tool '{reg_name}' from registry: {e_schema}. Using basic info.")
                    self.tool_schemas_for_prompt.append({"name": reg_name, "description": reg_desc, "error_getting_schema": str(e_schema)})
            except KeyError:
                logger.warning(f"[FastMCPEnv] Tool key '{tool_key}' not found in global tool registry. Skipping.")
            except Exception as e:
                logger.error(f"[FastMCPEnv] Error instantiating registered tool '{tool_key}': {e}. Skipping.")

        for provider in self._tool_providers:
            try:
                discovered_tools: List[Tool] = await provider.discover_tools()
                for tool_instance in discovered_tools: 
                    if not isinstance(tool_instance, Tool):
                        logger.warning(f"[FastMCPEnv] Item from provider {type(provider).__name__} (name: {getattr(tool_instance, 'name', 'Unknown')}) is not a Tool. Type: {type(tool_instance)}. Skipping.")
                        continue
                    
                    tool_name = tool_instance.name
                    if tool_name in self.active_tools:
                        logger.warning(
                            f"[FastMCPEnv] Tool '{tool_name}' (from provider {type(provider).__name__}) "
                            f"is overwriting an existing tool."
                        )
                    self.active_tools[tool_name] = tool_instance
                    try:
                        schema = await tool_instance.get_schema()
                        self.tool_schemas_for_prompt.append(schema)
                        logger.debug(f"[FastMCPEnv] Loaded tool from provider {type(provider).__name__}: {tool_name}")
                    except Exception as e_schema:
                        logger.error(f"[FastMCPEnv] Failed to get schema for tool '{tool_name}' from provider {type(provider).__name__}: {e_schema}. Using basic info.")
                        self.tool_schemas_for_prompt.append({"name": tool_name, "description": tool_instance.description, "error_getting_schema": str(e_schema)})
            except Exception as e:
                logger.error(f"[FastMCPEnv] Error discovering tools from provider {type(provider).__name__}: {e}. Skipping provider.")
        
        logger.debug(f"[FastMCPEnv] Setup complete. Active tools ({len(self.active_tools)}): {list(self.active_tools.keys())}")
        if not self.active_tools:
            logger.warning("[FastMCPEnv] No tools were loaded during setup.")

    def _get_formatted_tool_descriptions(self) -> str:
        """Formats the schemas of available tools into a JSON string for the LLM prompt."""
        if not self.tool_schemas_for_prompt:
            return "No tools available."
        try:
            return json.dumps(self.tool_schemas_for_prompt, indent=2)
        except Exception as e:
            # Error log is appropriate here.
            logger.error(f"[FastMCPEnv] Error formatting tool schemas to JSON: {e}")
            return "Error serializing tool schemas. Basic list:\n" + "\n".join(
                [f"- {s.get('name', 'Unknown')}: {s.get('description', 'N/A')}" for s in self.tool_schemas_for_prompt]
            )
            
    # Abstract method implementations from BaseEnvironment
    async def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[EnvironmentObservation, Dict[str, Any]]:
        self.current_turn = 0
        self._conversation_history = [] 
        
        prompt_to_use = options.get("initial_prompt", self.initial_prompt_template) if options else self.initial_prompt_template
        
        self._conversation_history.append({"role": "user", "content": prompt_to_use})
        # INFO for significant lifecycle event. Snippet of prompt is fine.
        logger.info(f"[FastMCPEnv] Environment Reset. Initial prompt snippet: '{prompt_to_use[:100]}...'")
        
        initial_obs = EnvironmentObservation(
            observation_type="initial",
            content=None, 
            tool_result=None,
            current_conversation=list(self._conversation_history),
            available_tools=list(self.tool_schemas_for_prompt), 
            requires_llm_action=True
        )
        return initial_obs, {"message": "Environment reset, initial prompt provided."}

    async def step(self, action: LLMAction) -> Tuple[EnvironmentObservation, float, bool, bool, Dict[str, Any]]:
        if not isinstance(action, dict) or "action_type" not in action or "raw_llm_output" not in action:
            # Error log is appropriate for invalid input structure.
            logger.error(f"[FastMCPEnv.step] Invalid action structure received: {action}")
            error_msg_for_llm = self.parser.format_environment_error("Internal error: Invalid action structure.")
            self._conversation_history.append({"role": "user", "content": error_msg_for_llm})
            obs = EnvironmentObservation(
                observation_type="environment_state", content=error_msg_for_llm, tool_result=None,
                current_conversation=list(self._conversation_history),
                available_tools=list(self.tool_schemas_for_prompt), requires_llm_action=True
            )
            return obs, 0.0, False, False, {"step_error": "Invalid LLMAction structure"}

        self.current_turn += 1
        self._conversation_history.append({"role": "assistant", "content": action["raw_llm_output"]})

        terminated = False
        truncated = False
        reward = 0.0  
        info: Dict[str, Any] = {"turn": self.current_turn, "action_received": action} # Action can be large, consider logging snippet or excluding from default logs
        
        current_observation_type: str = "environment_state"
        tool_result_obs: Optional[ToolResultObservation] = None
        environment_content: Optional[Any] = None

        if action['action_type'] == "tool_call":
            tool_call_data = action.get("tool_call")
            if not tool_call_data or not isinstance(tool_call_data.get("tool_name"), str):
                error_message_for_llm = self.parser.format_environment_error("Invalid tool_call structure.")
                self._conversation_history.append({"role": "user", "content": error_message_for_llm})
                environment_content = error_message_for_llm
                info["step_error"] = "Invalid tool_call structure"
            else:
                tool_name = tool_call_data['tool_name']
                tool_input = tool_call_data.get('tool_input')

                if tool_name in self.active_tools:
                    tool_to_execute = self.active_tools[tool_name]
                    # DEBUG for routine operation. Input can be verbose, log snippet or just name.
                    logger.debug(f"[FastMCPEnv.step] Executing tool: '{tool_name}' with input snippet: {str(tool_input)[:100]}...")
                    try:
                        tool_output_raw = await tool_to_execute.execute(tool_input)
                        # DEBUG for routine success. Output can be verbose, log snippet.
                        logger.debug(f"[FastMCPEnv.step] Tool '{tool_name}' executed. Output snippet: {str(tool_output_raw)[:100]}...")
                        tool_result_obs = ToolResultObservation(
                            tool_name=tool_name, tool_output=str(tool_output_raw),
                            status="success", error_message=None
                        )
                        formatted_tool_output = self.parser.format_successful_tool_result(str(tool_output_raw))
                        reward = 0.1 
                    except ToolExecutionError as e:
                        # Error log for tool execution failure.
                        logger.error(f"[FastMCPEnv.step] Tool '{tool_name}' execution error: {e}")
                        tool_result_obs = ToolResultObservation(
                            tool_name=tool_name, tool_output=None, status="error", error_message=str(e)
                        )
                        formatted_tool_output = self.parser.format_tool_error_result(str(e))
                        reward = -0.5 
                else:
                    # Warning for trying to use an unknown tool.
                    logger.warning(f"[FastMCPEnv.step] Unknown tool called: {tool_name}")
                    tool_result_obs = ToolResultObservation(
                        tool_name=tool_name, tool_output=None, status="error",
                        error_message=f"Tool '{tool_name}' not available."
                    )
                    formatted_tool_output = self.parser.format_tool_error_result(f"Tool '{tool_name}' not available.")
                    reward = -0.2
                
                current_observation_type = "tool_result"
                self._conversation_history.append({"role": "user", "content": formatted_tool_output})
                info["tool_executed"] = tool_name
                info["tool_result_status"] = tool_result_obs["status"] if tool_result_obs else "unknown"
        
        elif action['action_type'] == "text_response":
            parsed_output = self.parser.parse(action["raw_llm_output"])
            if parsed_output.final_answer is not None:
                terminated = True
                current_observation_type = "final_answer_feedback"
                environment_content = f"Final answer: {parsed_output.final_answer}"
                info["final_answer_detected"] = parsed_output.final_answer
                reward = 1.0 # Example reward for final answer
            elif parsed_output.parsing_error and not parsed_output.tool_command:
                error_msg_for_llm = self.parser.format_environment_error(f"Parse error: {parsed_output.parsing_error}")
                self._conversation_history.append({"role": "user", "content": error_msg_for_llm})
                environment_content = error_msg_for_llm
            else:
                environment_content = "LLM provided text response (thought/reasoning)."
                # The raw text is already in history.
        else:
            unknown_action_msg = self.parser.format_environment_error(f"Unknown action: {action.get('action_type')}")
            self._conversation_history.append({"role": "user", "content": unknown_action_msg})
            environment_content = unknown_action_msg
            info["step_error"] = f"Unknown action_type: {action.get('action_type')}"

        requires_llm_action_next = True
        if terminated or (self.current_turn >= self.max_steps): # Using self.max_steps
            truncated = not terminated
            requires_llm_action_next = False
            if truncated: 
                info["truncation_reason"] = "max_steps_reached"
        
        observation = EnvironmentObservation(
            observation_type=current_observation_type, content=environment_content,
            tool_result=tool_result_obs, current_conversation=list(self._conversation_history),
            available_tools=list(self.tool_schemas_for_prompt), requires_llm_action=requires_llm_action_next
        )
        return observation, reward, terminated, truncated, info

    def render(self) -> None: # Added render method
        print("\n--- FastMCPEnv Conversation ---")
        for i, message in enumerate(self._conversation_history):
            print(f"[{i:02d}] {message['role']}: {message['content']}")
        print("----------------------------")

    def close(self) -> None: # Added close method
        print("FastMCPEnv closed.")
        pass

    async def rollout(
        self,
        initial_prompt: str,
        llm_model: 'ModelObject',
        tokenizer: 'TokenizerObject',
        sampling_params: 'SamplingParams',
        max_tokens_to_generate: int = 256,
    ) -> RawRolloutData:
        logger.debug(f"[FastMCPEnv.rollout] Starting rollout. Initial prompt snippet: '{initial_prompt[:100]}...'")
        logger.trace(f"[FastMCPEnv.rollout] Sampling_params: {sampling_params}")

        default_for_max_tokens = max_tokens_to_generate
        actual_max_tokens_for_generation: int = default_for_max_tokens
        if isinstance(sampling_params, dict):
            retrieved_max_tokens = sampling_params.get("max_tokens_to_generate")
            if retrieved_max_tokens is not None:
                try:
                    actual_max_tokens_for_generation = int(retrieved_max_tokens)
                    logger.trace(f"[FastMCPEnv.rollout] Using 'max_tokens_to_generate={actual_max_tokens_for_generation}' from sampling_params.")
                except (ValueError, TypeError):
                    logger.warning(
                        f"[FastMCPEnv.rollout] Invalid 'max_tokens_to_generate={retrieved_max_tokens}' in sampling_params. "
                        f"Falling back to default: {default_for_max_tokens}."
                    )
                    actual_max_tokens_for_generation = default_for_max_tokens
            else:
                logger.trace(f"[FastMCPEnv.rollout] 'max_tokens_to_generate' not in sampling_params. Using default: {default_for_max_tokens}.")
        else:
            logger.trace(f"[FastMCPEnv.rollout] sampling_params not dict. Using default max_tokens_to_generate: {default_for_max_tokens}.")
        
        logger.debug(f"[FastMCPEnv.rollout] LLM max_new_tokens for generation: {actual_max_tokens_for_generation}")

        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                logger.trace(f"[FastMCPEnv.rollout] Tokenizer pad_token_id set to eos_token_id: {tokenizer.eos_token_id}")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.warning(f"[FastMCPEnv.rollout] Tokenizer missing pad_token_id and eos_token_id. Added new pad_token: {tokenizer.pad_token_id}. This might affect model behavior.")
        
        device = getattr(llm_model, 'device', 'cpu')
        logger.trace(f"[FastMCPEnv.rollout] Device for LLM ops: {device}.")

        obs_list: List[EnvironmentObservation] = []
        action_list: List[LLMAction] = []
        reward_list: List[float] = []
        info_dicts_per_step: List[Dict[str, Any]] = []
        
        current_observation, info_reset = await self.reset(options={"initial_prompt": initial_prompt})
        obs_list.append(current_observation)

        terminated = False
        truncated = False
        turn_number = 0
        info: Dict[str, Any] = {} # Initialize info dict to be available after the loop

        while not (terminated or truncated):
            turn_number += 1
            logger.trace(f"[FastMCPEnv.rollout] ----- Turn {turn_number} -----")
            if current_observation["requires_llm_action"]:
                formatted_tools_string = self._get_formatted_tool_descriptions()
                system_prompt_content = (
                    "You are a helpful assistant interacting with FastMCP. "
                    "To use a tool, you MUST output the tool name and arguments as a JSON string enclosed \n"
                    "EXACTLY within <tool> and </tool> tags. For example:\n"
                    "<tool>{\"name\": \"tool_name_here\", \"args\": {\"param1\": \"value1\", \"param2\": 123}}</tool>\n"
                    "Do NOT put any other text before or after the <tool>...</tool> block if you are calling a tool.\n"
                    "The JSON inside <tool> must have a \"name\" key for the tool and an optional \"args\" key for its arguments.\n"
                    "Available tools are listed below. Only use these exact tool names.\n"
                    f"{formatted_tools_string}\n\n"
                    "If you are providing a final answer, use the <answer>YOUR_FINAL_ANSWER_HERE</answer> tag.\n"
                    "For intermediate reasoning or thoughts, use <reasoning>YOUR_THOUGHTS_HERE</reasoning>.\n"
                    "Tool results provided by the system will be in <result>...</result> tags."
                )
                
                prompting_conversation = [{"role": "system", "content": system_prompt_content}]
                for msg in current_observation["current_conversation"]:
                    if msg.get("role") != "system":
                        prompting_conversation.append(msg)
                
                try:
                    prompt_text_for_llm = tokenizer.apply_chat_template(
                        prompting_conversation, tokenize=False, add_generation_prompt=True
                    )
                    log_prompt_snippet = prompt_text_for_llm[:500] + ("..." if len(prompt_text_for_llm) > 500 else "")
                    logger.trace(f"[FastMCPEnv.rollout] Prompt for LLM (len {len(prompt_text_for_llm)}):\n{log_prompt_snippet}")
                except Exception as e_template:
                    logger.warning(f"[FastMCPEnv.rollout] Failed to apply chat template: {e_template}. Using basic join for prompt construction.")
                    prompt_text_for_llm = "\n".join([f"{m['role']}: {m['content']}" for m in prompting_conversation])

                inputs = tokenizer(prompt_text_for_llm, return_tensors="pt", padding=True).to(device)
                
                hf_generate_params = { 
                    k: v for k, v in sampling_params.items() 
                    if k not in ["max_tokens_to_generate", "temperature_for_logprobs"]
                }
                hf_generate_params.setdefault("temperature", 0.7)
                hf_generate_params.setdefault("do_sample", True)

                generate_kwargs = {
                    "max_new_tokens": actual_max_tokens_for_generation,
                    "pad_token_id": tokenizer.pad_token_id,
                    **hf_generate_params 
                }
                if "max_tokens_to_generate" in generate_kwargs: 
                    del generate_kwargs["max_tokens_to_generate"]

                logger.debug(f"[FastMCPEnv.rollout] Calling model.generate with effective kwargs: {generate_kwargs}")
                
                raw_llm_text_output_for_parse: str = ""
                output_sequences_obj: Optional[Any] = None

                try:
                    with torch.no_grad():
                        output_sequences_obj = llm_model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            return_dict_in_generate=True, 
                            output_scores=True,           
                            **generate_kwargs,
                        )
                    
                    if output_sequences_obj is None or not hasattr(output_sequences_obj, 'sequences'):
                        raise ValueError("LLM generate call did not return expected sequence data.")

                    generated_tokens_for_parse = output_sequences_obj.sequences[0, inputs.input_ids.shape[-1]:]
                    raw_llm_text_output_for_parse = tokenizer.decode(generated_tokens_for_parse, skip_special_tokens=True)

                    full_raw_output_log = tokenizer.decode(generated_tokens_for_parse, skip_special_tokens=False)
                    logger.trace(f"[FastMCPEnv.rollout] Turn {turn_number}: Raw LLM Output (with special tokens) for parsing: {full_raw_output_log[:500]}...")
                    logger.debug(f"[FastMCPEnv.rollout] Turn {turn_number}: Decoded LLM Output (for parsing, special tokens skipped): '{raw_llm_text_output_for_parse[:200]}...'")

                except Exception as e_generate:
                    logger.error(f"[FastMCPEnv.rollout] LLM generate call failed: {e_generate}", exc_info=True)
                    raw_llm_text_output_for_parse = ""

                processed_llm_output_for_parser = raw_llm_text_output_for_parse.strip()
                original_for_comparison_after_think_strip = processed_llm_output_for_parser
                
                primary_tags_for_prefix_strip = ["<tool", "<reasoning", "<answer"]
                first_tag_index = -1
                
                indices_for_prefix_strip = []
                for tag_prefix in primary_tags_for_prefix_strip:
                    try:
                        indices_for_prefix_strip.append(processed_llm_output_for_parser.index(tag_prefix))
                    except ValueError:
                        pass 
                
                if indices_for_prefix_strip:
                    first_tag_index = min(indices_for_prefix_strip)

                if first_tag_index > 0: 
                    stripped_prefix_text = processed_llm_output_for_parser[:first_tag_index]
                    processed_llm_output_for_parser = processed_llm_output_for_parser[first_tag_index:]
                    logger.info(f"[FastMCPEnv.rollout] Stripped leading text ('{stripped_prefix_text[:100]}...') before first primary XML tag. Output for parsing: '{processed_llm_output_for_parser[:200]}...'")
                elif first_tag_index == -1 and processed_llm_output_for_parser and processed_llm_output_for_parser != original_for_comparison_after_think_strip:
                    logger.debug(f"[FastMCPEnv.rollout] No primary XML tag found after potential <think> stripping. Original (pre-think-strip): '{original_for_comparison_after_think_strip[:200]}...'. Current for parsing: '{processed_llm_output_for_parser[:200]}...'")
                elif first_tag_index == -1 and processed_llm_output_for_parser: 
                    logger.debug(f"[FastMCPEnv.rollout] No primary XML tag found in LLM output to guide stripping. Original for parsing: '{processed_llm_output_for_parser[:200]}...'")

                parsed_output_after_stripping: ParsedSmolLLMOutput = self.parser.parse(processed_llm_output_for_parser)
                
                action_logprobs: Optional[torch.Tensor] = None
                if output_sequences_obj is not None and hasattr(output_sequences_obj, 'scores'):
                    scores_tuple = getattr(output_sequences_obj, 'scores', None)
                    if scores_tuple is not None:
                        stacked_scores = torch.stack(scores_tuple, dim=1)
                        log_probs_full_vocab = torch.nn.functional.log_softmax(stacked_scores, dim=-1)
                        
                        generated_ids_for_logprobs = generated_tokens_for_parse.unsqueeze(0).to(log_probs_full_vocab.device)
                        
                        action_logprobs_batched = torch.gather(log_probs_full_vocab, -1, generated_ids_for_logprobs.unsqueeze(-1)).squeeze(-1)
                        
                        if action_logprobs_batched.shape[0] == 1:
                            action_logprobs = action_logprobs_batched.squeeze(0).cpu()
                        else: 
                            action_logprobs = action_logprobs_batched.cpu()
                        logger.trace(f"[FastMCPEnv.rollout] Calculated action_logprobs (shape: {action_logprobs.shape if action_logprobs is not None else 'None'}) for raw output snippet: '{raw_llm_text_output_for_parse[:50]}...'")
                    else:
                        logger.warning("[FastMCPEnv.rollout] 'scores' attribute was None in LLM output despite hasattr check. Logprobs cannot be calculated.")
                else:
                    missing_reason = "output_sequences_obj was None (e.g., LLM generation failed)" if output_sequences_obj is None else "output_sequences_obj did not have 'scores' attribute"
                    logger.warning(f"[FastMCPEnv.rollout] Logprobs cannot be calculated: {missing_reason}.")

                logger.debug(f"[FastMCPEnv.rollout] Turn {self.current_turn + 1}: Parsed LLM Output from stripped string: {parsed_output_after_stripping}")
                
                action_type_str: Optional[str] = None
                text_content: Optional[str] = None
                tool_call_payload: Optional[ToolCallAction] = None
                reasoning_content: Optional[str] = parsed_output_after_stripping.reasoning

                if parsed_output_after_stripping.tool_command:
                    action_type_str = "tool_call"
                    try:
                        tool_cmd_dict = json.loads(parsed_output_after_stripping.tool_command)
                        tool_name = tool_cmd_dict.get("name")
                        tool_args = tool_cmd_dict.get("args")

                        if not isinstance(tool_name, str) or not tool_name:
                            raise ValueError("Tool name missing or not a string in parsed command.")
                        
                        tool_call_payload = ToolCallAction(tool_name=tool_name, tool_input=tool_args)
                        text_content = parsed_output_after_stripping.reasoning
                        logger.debug(f"[FastMCPEnv.rollout] Successfully parsed tool call: {tool_name} with args: {tool_args}")

                    except json.JSONDecodeError as e_json:
                        action_type_str = "text_response"
                        tool_call_payload = None
                        error_detail = f"Invalid JSON in <tool> command: {e_json}. Command content: '{parsed_output_after_stripping.tool_command}'"
                        text_content = f"ERROR: Your instruction to use a tool was malformed. {error_detail}"
                        logger.warning(f"[FastMCPEnv.rollout] {error_detail}")
                    except ValueError as e_val:
                        action_type_str = "text_response"
                        tool_call_payload = None
                        error_detail = f"Invalid tool command structure: {e_val}. Command content: '{parsed_output_after_stripping.tool_command}'"
                        text_content = f"ERROR: Your instruction to use a tool had an invalid structure. {error_detail}"
                        logger.warning(f"[FastMCPEnv.rollout] {error_detail}")
                    except Exception as e_tool_parse: 
                        action_type_str = "text_response" 
                        tool_call_payload = None
                        error_detail = f"Unexpected error parsing tool command: {e_tool_parse}. Command: '{parsed_output_after_stripping.tool_command}'"
                        text_content = f"ERROR: An unexpected error occurred while processing your tool command. {error_detail}"
                        logger.error(f"[FastMCPEnv.rollout] {error_detail}", exc_info=True)

                elif parsed_output_after_stripping.final_answer is not None:
                    action_type_str = "text_response"
                    text_content = parsed_output_after_stripping.final_answer
                elif parsed_output_after_stripping.reasoning is not None: 
                    action_type_str = "text_response"
                    text_content = parsed_output_after_stripping.reasoning
                elif parsed_output_after_stripping.parsing_error:
                    action_type_str = "text_response"
                    text_content = f"ERROR: Could not parse your response due to: {parsed_output_after_stripping.parsing_error}. Original response: {raw_llm_text_output_for_parse}"
                    # Using f"""...""" for robustness if error messages or output contain quotes
                    logger.warning(f"""[FastMCPEnv.rollout] SmolAgentParser indicated parsing error: '{parsed_output_after_stripping.parsing_error}'. Raw LLM output snippet: '{raw_llm_text_output_for_parse[:200]}...'""")
                else: 
                      action_type_str = "text_response"
                      text_content = raw_llm_text_output_for_parse 
                      if not text_content.strip() and not reasoning_content: 
                        logger.debug("[FastMCPEnv.rollout] LLM output was empty or whitespace after initial stripping and parsing. Passing as empty text_response.")
                        text_content = "" 
                      else:
                        logger.info(f"[FastMCPEnv.rollout] LLM output not parsed into tool/answer/reasoning. Treating as simple text response: '{text_content[:100]}...'")
                
                final_action_type: Literal["tool_call", "text_response"]
                if action_type_str == "tool_call":
                    final_action_type = "tool_call"
                elif action_type_str == "text_response":
                    final_action_type = "text_response"
                else: 
                    logger.error(f"[FastMCPEnv.rollout] CRITICAL: action_type_str has unexpected value '{action_type_str}'. Defaulting to error text_response.")
                    final_action_type = "text_response"
                    text_content = "INTERNAL ERROR: Could not determine action type due to unexpected state."
                
                llm_action_to_step = LLMAction(
                    action_type=final_action_type,
                    text=text_content,
                    tool_call=tool_call_payload,
                    raw_llm_output=raw_llm_text_output_for_parse, 
                    reasoning=reasoning_content,
                    old_per_token_logps=action_logprobs
                )
                
                action_list.append(llm_action_to_step)
                next_observation, reward, terminated, truncated, info = await self.step(llm_action_to_step)
                
                obs_list.append(next_observation)
                reward_list.append(reward)
                info_dicts_per_step.append(info)
                current_observation = next_observation
            else: 
                logger.debug("[FastMCPEnv.rollout] Loop condition met: current_observation does not require LLM action. Terminating rollout.")
                break 
        
        # Ensure final_observation_for_rollout_data is defined using the last state of current_observation
        final_observation_for_rollout_data = current_observation 
        
        # Ensure info is the one from the last step if loop executed, or the initial one if not.
        # The variable `info` will hold the info from the last call to self.step, 
        # or it will be the initial empty dict if the loop didn't run (e.g. initial obs says no action needed).
        rollout_completion_info = info.copy() if info else {} # Make a copy, ensure it's a dict
        
        if terminated:
            rollout_completion_info["rollout_status"] = "terminated"
            logger.debug(f"[FastMCPEnv.rollout] Rollout ended: Terminated. Total turns: {turn_number}")
        elif truncated:
            rollout_completion_info["rollout_status"] = "truncated"
            rollout_completion_info.setdefault("truncation_reason", "max_steps_reached_or_no_action_required")
            logger.debug(f"[FastMCPEnv.rollout] Rollout ended: Truncated. Total turns: {turn_number}. Reason: {rollout_completion_info.get('truncation_reason')}")
        else:
            # This case might occur if the loop breaks because requires_llm_action is false initially or after a step
            # and it wasn't a standard termination/truncation from the step method.
            rollout_completion_info.setdefault("rollout_status", "ended_no_action_required")
            logger.debug(f"[FastMCPEnv.rollout] Rollout ended. Status: {rollout_completion_info['rollout_status']}. Total turns: {turn_number}")

        raw_rollout_data = RawRolloutData(
            full_conversation_history=list(final_observation_for_rollout_data["current_conversation"]),
            executed_llm_actions=action_list,
            intrinsic_rewards_per_turn=reward_list,
            final_environment_observation=final_observation_for_rollout_data,
            step_info_dicts_per_turn=info_dicts_per_step,
            rollout_metadata=rollout_completion_info
        )
        logger.info(f"[FastMCPEnv.rollout] Completed. Actions: {len(action_list)}, History msgs: {len(final_observation_for_rollout_data['current_conversation'])}. Status: {rollout_completion_info.get('rollout_status', 'unknown')}")

        return raw_rollout_data

# Main example execution block (self-executable part)
async def main_example():
    print("--- Running FastMCPEnv Example ---")

    # 1. Setup Mock FastMCP Client (placeholder)
    class MockFastMCPClient:
        async def list_tools(self):
            print("[MockFastMCPClient] list_tools() called")
            # Simulate tools discovered from an MCP server
            return [
                type('MockMCPToolInfo', (), {'name': 'mcp_sendCommand', 'description': 'Sends a command via MCP.'})(),
                type('MockMCPToolInfo', (), {'name': 'mcp_getPlayerHealth', 'description': 'Gets player health via MCP.'})()
            ]

        async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None):
            print(f"[MockFastMCPClient] call_tool(name='{name}', arguments={arguments}) called")
            if name == "mcp_sendCommand":
                return f"Command '{arguments.get('command_string', '') if arguments else ''}' executed via MCP."
            elif name == "mcp_getPlayerHealth":
                return {"player_id": arguments.get('player', 'default_player') if arguments else 'default_player', "health": 20, "max_health": 20}
            return f"Unknown MCP tool '{name}' called on mock client."

    mock_mcp_client = MockFastMCPClient()

    # 2. Instantiate Providers
    # Note: FastMCPToolProvider expects the actual client type if type hints were strict.
    # We are using `Any` as a placeholder for FastMCPClientType for now.
    mcp_provider = FastMCPToolProvider(client=mock_mcp_client) 

    # Manually instantiate a tool for ManualToolProvider
    # manual_calc_instance = SimpleCalculatorTool(name="manual_calculator", description="A manually added calculator.")
    # manual_provider = ManualToolProvider(tools=[manual_calc_instance])
    manual_provider = ManualToolProvider(tools=[]) # Initialize with empty list if no manual_calc_instance

    # 3. Instantiate the Environment
    env = FastMCPEnv(
        tool_registry_keys=["simple_calculator_tool"], # From @tool decoration (assuming one is still registered elsewhere)
        tool_configs={
            # "simple_calculator_tool": {"precision": 4} # If SimpleCalculatorTool took config
        },
        # tool_providers=[mcp_provider, manual_provider]
        tool_providers=[mcp_provider] # Use only mcp_provider if manual_provider is empty
    )

    # 4. Setup tools in the environment (this is crucial)
    await env.setup()

    # 5. Simulate interactions
    # Interaction 1: Reset and call calculator
    print("\n--- Interaction 1: Reset and Call Calculator ---")
    obs, info = await env.reset(options={"initial_prompt": "Calculate 10 + 5."})
    print(f"Reset Obs: {obs}, Info: {info}")

    # calc_action = LLMAction(
    #     action_type="tool_call",
    #     tool_call=ToolCallAction(tool_name="simple_calculator_tool", tool_input={"x": 10, "y": 5}),
    #     raw_llm_output="<tool_call name='simple_calculator_tool'><input x='10' y='5'/></tool_call>"
    # )
    # obs, reward, term, trunc, info = await env.step(calc_action)
    # print(f"Step Obs: {obs}\nReward: {reward}, Term: {term}, Trunc: {trunc}, Info: {info}")

    # Interaction 2: Call a discovered FastMCP tool
    print("\n--- Interaction 2: Call FastMCP Tool (sendCommand) ---")
    mcp_command_action = LLMAction(
        action_type="tool_call",
        tool_call=ToolCallAction(tool_name="fastmcp_mcp_sendCommand", tool_input={"command_string": "time set day"}),
        raw_llm_output="<tool_call name='fastmcp_mcp_sendCommand'><input command_string='time set day'/></tool_call>",
        text=None, # Added text=None
        reasoning=None, # Added reasoning=None
        old_per_token_logps=None # Added old_per_token_logps=None
    )
    obs, reward, term, trunc, info = await env.step(mcp_command_action)
    print(f"Step Obs: {obs}\nReward: {reward}, Term: {term}, Trunc: {trunc}, Info: {info}")

    # Interaction 3: Call an unknown tool
    print("\n--- Interaction 3: Call Unknown Tool ---")
    unknown_tool_action = LLMAction(
        action_type="tool_call",
        tool_call=ToolCallAction(tool_name="non_existent_tool", tool_input={}),
        raw_llm_output="<tool_call name='non_existent_tool'/>",
        text=None, # Added text=None
        reasoning=None, # Added reasoning=None
        old_per_token_logps=None # Added old_per_token_logps=None
    )
    obs, reward, term, trunc, info = await env.step(unknown_tool_action)
    print(f"Step Obs: {obs}\nReward: {reward}, Term: {term}, Trunc: {trunc}, Info: {info}")

    # Interaction 4: Text response from LLM
    print("\n--- Interaction 4: Text Response ---")
    text_action = LLMAction(
        action_type="text_response",
        text="Okay, I think I am done now.",
        tool_call=None,
        raw_llm_output="Okay, I think I am done now.",
        reasoning=None, # Added reasoning=None
        old_per_token_logps=None # Added old_per_token_logps=None
    )
    obs, reward, term, trunc, info = await env.step(text_action)
    print(f"Step Obs: {obs}\nReward: {reward}, Term: {term}, Trunc: {trunc}, Info: {info}")

if __name__ == "__main__":
    asyncio.run(main_example()) 