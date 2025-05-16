from typing import TypedDict, Literal, Optional, Dict, Any, List
import torch # ADDED for torch.Tensor

# Represents a parsed tool call intent from the LLM
class ToolCallAction(TypedDict):
    tool_name: str
    tool_input: Optional[Dict[str, Any]] # Arguments for the tool

# Represents the agent's (LLM's) action after parsing
class LLMAction(TypedDict):
    action_type: Literal["text_response", "tool_call"]
    text: Optional[str]          # Content for a text_response
    tool_call: Optional[ToolCallAction] # Structured tool call
    raw_llm_output: str          # The original string from the LLM, for logging/debugging
    reasoning: Optional[str]     # Added: Reasoning text, if separate from main text/answer
    old_per_token_logps: Optional[torch.Tensor]
    # Optional: could also store parsed_llm_output (like ParsedSmolLLMOutput) directly if needed
    # parsed_output: Optional[Dict[str, Any]] 

# Represents the outcome of a tool execution by the environment
class ToolResultObservation(TypedDict):
    tool_name: str
    tool_output: Any # Serialized output from the tool
    status: Literal["success", "error"]
    error_message: Optional[str] # Error message if status is "error"

# Represents the observation provided by the environment to the agent
class EnvironmentObservation(TypedDict):
    observation_type: Literal["tool_result", "environment_state", "initial", "final_answer_feedback"]
    content: Optional[Any] # General content (e.g., text description of state, feedback)
    tool_result: Optional[ToolResultObservation]
    current_conversation: List[Dict[str, str]] # Added: Full history for the LLM to consume
    available_tools: List[Dict[str, str]]     # Added: So LLM knows what it can call
    requires_llm_action: bool                 # Added: True if the LLM should generate a response next
    # Potentially other fields like current_step_number, etc. can be added later. 