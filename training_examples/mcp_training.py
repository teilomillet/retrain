#!/usr/bin/env python3
"""
MCP Training Script for Retrain

Simple script to train Qwen3-0.6B to use MCP tools effectively.
Follows the same pattern as examples/run_example.py
"""

import asyncio
import sys
from pathlib import Path
import yaml
from typing import Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrain import run
from retrain.reward import reward
from loguru import logger

# MCP Training Reward Functions
@reward(name="tool_discovery_reward")
def tool_discovery_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """Rewards tool discovery and exploration."""
    step_info = kwargs.get("step_info", {})
    
    # Check for parsing errors
    if isinstance(step_info, dict):
        error_message = None
        if "parsing_error_in_text_response" in step_info:
            error_message = step_info["parsing_error_in_text_response"]
        elif "action_received" in step_info and isinstance(step_info["action_received"], dict):
            error_message = step_info["action_received"].get("parsing_error_message")
        
        if error_message:
            return -0.5
    
    discovery_keywords = config_params.get("discovery_keywords", [])
    tool_names = config_params.get("tool_names", [])
    discovery_bonus = float(config_params.get("discovery_bonus", 0.5))
    
    reward = 0.0
    prompt_lower = prompt.lower()
    
    if any(keyword.lower() in prompt_lower for keyword in discovery_keywords):
        reward += 0.5
        completion_lower = completion.lower()
        if any(keyword.lower() in completion_lower for keyword in discovery_keywords):
            reward += 0.3
        if any(tool_name.lower() in completion_lower for tool_name in tool_names):
            reward += discovery_bonus
    
    return reward

@reward(name="tool_usage_reward")
def tool_usage_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """Rewards appropriate tool usage."""
    step_info = kwargs.get("step_info", {})
    
    tool_call_reward = float(config_params.get("tool_call_reward", 0.3))
    no_tool_penalty = float(config_params.get("no_tool_penalty", -0.2))
    prompt_keywords = config_params.get("prompt_keywords_for_tool", [])
    
    reward = 0.0
    action_type = None
    
    if isinstance(step_info, dict):
        if "action_received" in step_info and isinstance(step_info["action_received"], dict):
            action_type = step_info["action_received"].get("action_type")
    
    if action_type == "tool_call":
        reward += tool_call_reward
    elif any(keyword.lower() in prompt.lower() for keyword in prompt_keywords):
        reward += no_tool_penalty
    
    return reward

@reward(name="correct_tool_reward")
def correct_tool_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """Rewards using correct tool for task."""
    step_info = kwargs.get("step_info", {})
    
    tool_mapping = config_params.get("tool_mapping", {})
    correct_tool_bonus = float(config_params.get("correct_tool_bonus", 0.4))
    wrong_tool_penalty = float(config_params.get("wrong_tool_penalty", -0.3))
    
    reward = 0.0
    tool_name = None
    
    if isinstance(step_info, dict):
        if "action_received" in step_info and isinstance(step_info["action_received"], dict):
            action_data = step_info["action_received"]
            if action_data.get("action_type") == "tool_call":
                tool_name = action_data.get("tool_name")
    
    if tool_name:
        prompt_lower = prompt.lower()
        expected_tools = []
        
        for keyword, tools in tool_mapping.items():
            if keyword.lower() in prompt_lower:
                expected_tools.extend(tools)
        
        if expected_tools:
            if tool_name in expected_tools:
                reward += correct_tool_bonus
            else:
                reward += wrong_tool_penalty
    
    return reward

@reward(name="tool_parameter_reward")
def tool_parameter_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """Rewards correct tool parameters."""
    step_info = kwargs.get("step_info", {})
    
    valid_operations = config_params.get("valid_operations", [])
    parameter_quality_bonus = float(config_params.get("parameter_quality_bonus", 0.2))
    invalid_parameter_penalty = float(config_params.get("invalid_parameter_penalty", -0.2))
    
    reward = 0.0
    
    if isinstance(step_info, dict):
        if "action_received" in step_info and isinstance(step_info["action_received"], dict):
            action_data = step_info["action_received"]
            if action_data.get("action_type") == "tool_call":
                tool_params = action_data.get("tool_params", {})
                
                if action_data.get("tool_name") == "perform_operation":
                    operation = tool_params.get("operation")
                    if operation in valid_operations:
                        reward += parameter_quality_bonus
                    elif operation:
                        reward += invalid_parameter_penalty
                
                if "operand1" in tool_params and "operand2" in tool_params:
                    try:
                        float(tool_params["operand1"])
                        float(tool_params["operand2"])
                        reward += parameter_quality_bonus
                    except (ValueError, TypeError):
                        reward += invalid_parameter_penalty
    
    return reward

@reward(name="multi_step_reward")
def multi_step_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """Rewards multi-step task completion."""
    step_info = kwargs.get("step_info", {})
    
    step_completion_bonus = float(config_params.get("step_completion_bonus", 0.3))
    
    reward = 0.0
    multi_step_keywords = ["first", "then", "finally", "and then", "after"]
    
    if any(keyword in prompt.lower() for keyword in multi_step_keywords):
        reward += step_completion_bonus
        
        if isinstance(step_info, dict):
            if "action_received" in step_info and step_info["action_received"]:
                reward += step_completion_bonus
    
    return reward

@reward(name="task_completion_reward")
def task_completion_reward(raw_rollout: Dict[str, Any], infos: List[Dict[str, Any]], **kwargs) -> float:
    """Rollout-level reward for task completion."""
    # Extract config parameters from the rollout data or use defaults
    # Since rollout rewards don't get config_params passed directly, we'll use defaults
    completion_threshold = 0.7
    completion_bonus = 0.5
    exploration_bonus = 0.3
    
    # Extract the completion from the rollout data
    # The completion is the last LLM action's raw_llm_output
    executed_llm_actions = raw_rollout.get("executed_llm_actions", [])
    if not executed_llm_actions:
        return 0.0
    
    # Get the last completion
    last_action = executed_llm_actions[-1]
    if not isinstance(last_action, dict) or "raw_llm_output" not in last_action:
        return 0.0
    
    completion = last_action["raw_llm_output"]
    
    reward = 0.0
    completion_indicators = ["result", "answer", "calculated", "time is", "capabilities"]
    completion_lower = completion.lower()
    
    completion_score = sum(1 for indicator in completion_indicators if indicator in completion_lower)
    completion_score /= len(completion_indicators)
    
    if completion_score >= completion_threshold:
        reward += completion_bonus
    
    exploration_indicators = ["available", "tools", "capabilities", "can do"]
    if any(indicator in completion_lower for indicator in exploration_indicators):
        reward += exploration_bonus
    
    return reward

async def main():
    """Main training function."""
    logger.info("=== MCP Training with Retrain ===")
    
    # Load configuration
    config_file_path = Path(__file__).parent / "mcp_config.yaml"
    logger.info(f"Loading configuration from: {config_file_path}")
    
    if not config_file_path.exists():
        logger.error(f"Configuration file not found at: {config_file_path}")
        return
    
    try:
        with open(config_file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        return
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Start training
    logger.info("Starting MCP training...")
    try:
        results = await run(config=config_dict)
        logger.info("✅ MCP training completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 