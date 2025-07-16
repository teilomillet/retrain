"""
Format compliance reward functions for enforcing proper XML format in model outputs.

This module provides rewards that heavily penalize format violations to encourage
the model to follow the expected XML structure for tool usage and responses.
"""

import re
from typing import Dict, Any
from .reward import reward
from ..utils.parser.smol_agent_parser import SmolAgentParser, ParsedSmolLLMOutput


@reward("format_compliance")
def format_compliance(prompt: str, completion: str, **kwargs) -> float:
    """
    Heavily penalizes format violations and rewards proper XML format compliance.
    
    Expected formats:
    1. <think>reasoning</think> followed by <tool>JSON</tool>
    2. <think>reasoning</think> followed by <answer>content</answer>
    3. <tool>JSON</tool> only
    4. <answer>content</answer> only
    5. <think>reasoning</think> only (partial, small penalty)
    
    Heavy penalties for:
    - Parsing errors (-2.0)
    - Malformed XML tags (-1.5)
    - Content outside XML tags (-1.0)
    - Missing closing tags (-1.5)
    
    Rewards for:
    - Perfect format compliance (+1.0)
    - Partial compliance (+0.3 to +0.7)
    """
    parser = SmolAgentParser()
    
    try:
        parsed_output: ParsedSmolLLMOutput = parser.parse(completion)
        
        # Heavy penalty for parsing errors
        if parsed_output.parsing_error:
            if "No recognized <tool>, <answer>, or <think> tags found" in parsed_output.parsing_error:
                return -2.0  # Heaviest penalty for complete format violation
            elif "not structured correctly" in parsed_output.parsing_error:
                return -1.5  # Heavy penalty for incorrect structure
            elif "No <tool> or <answer> tag found as the primary action" in parsed_output.parsing_error:
                return -0.8  # Penalty for incomplete action (only thinking)
            else:
                return -1.0  # General parsing error penalty
        
        # Check for format compliance beyond basic parsing
        completion_stripped = completion.strip()
        
        # Reward successful parsing with proper structure
        base_reward = 0.0
        
        # Perfect structure: reasoning + action or just action
        if (parsed_output.reasoning and (parsed_output.tool_command or parsed_output.final_answer)) or \
           (not parsed_output.reasoning and (parsed_output.tool_command or parsed_output.final_answer)):
            base_reward = 1.0
        elif parsed_output.reasoning and not parsed_output.tool_command and not parsed_output.final_answer:
            # Only reasoning, partial reward
            base_reward = 0.3
        
        # Additional checks for XML tag quality
        format_quality_bonus = _check_xml_tag_quality(completion_stripped)
        
        # Check for content outside XML tags (should be minimal)
        outside_content_penalty = _check_content_outside_tags(completion_stripped)
        
        final_reward = base_reward + format_quality_bonus + outside_content_penalty
        
        # Cap the reward to reasonable bounds
        return max(-2.0, min(1.5, final_reward))
        
    except Exception as e:
        # Parser itself failed, heavy penalty
        return -2.0


def _check_xml_tag_quality(completion: str) -> float:
    """Check the quality of XML tag formatting."""
    quality_score = 0.0
    
    # Check for properly closed tags
    think_matches = re.findall(r'<think>.*?</think>', completion, re.DOTALL)
    tool_matches = re.findall(r'<tool>.*?</tool>', completion, re.DOTALL)
    answer_matches = re.findall(r'<answer>.*?</answer>', completion, re.DOTALL)
    
    # Reward for properly closed tags
    if think_matches:
        quality_score += 0.1
    if tool_matches:
        quality_score += 0.2
    if answer_matches:
        quality_score += 0.2
    
    # Check for unclosed tags (penalty)
    unclosed_patterns = [
        r'<think>(?!.*</think>)',  # <think> without closing
        r'<tool>(?!.*</tool>)',    # <tool> without closing
        r'<answer>(?!.*</answer>)' # <answer> without closing
    ]
    
    for pattern in unclosed_patterns:
        if re.search(pattern, completion, re.DOTALL):
            quality_score -= 0.5  # Heavy penalty for unclosed tags
    
    return quality_score


def _check_content_outside_tags(completion: str) -> float:
    """Check for significant content outside XML tags."""
    penalty = 0.0
    
    # Remove all recognized XML tag content
    cleaned = completion
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<tool>.*?</tool>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<answer>.*?</answer>', '', cleaned, flags=re.DOTALL)
    
    # What's left should be minimal (whitespace, minimal text)
    remaining = cleaned.strip()
    
    if remaining:
        # Count non-whitespace characters outside tags
        non_whitespace_count = len(re.sub(r'\s', '', remaining))
        
        if non_whitespace_count > 50:
            penalty -= 0.8  # Heavy penalty for lots of content outside tags
        elif non_whitespace_count > 20:
            penalty -= 0.4  # Medium penalty
        elif non_whitespace_count > 5:
            penalty -= 0.1  # Light penalty
    
    return penalty


@reward("xml_structure_enforcement") 
def xml_structure_enforcement(prompt: str, completion: str, **kwargs) -> float:
    """
    Extreme penalty reward function specifically for enforcing XML structure.
    
    This is designed to be used alongside other rewards to strongly discourage
    any deviation from the expected XML format.
    
    Returns:
    - +0.5 for perfect XML structure
    - 0.0 for acceptable structure
    - -3.0 to -5.0 for violations (scaled by severity)
    """
    completion_stripped = completion.strip()
    
    # Check for basic XML tag presence
    has_think = '<think>' in completion_stripped and '</think>' in completion_stripped
    has_tool = '<tool>' in completion_stripped and '</tool>' in completion_stripped  
    has_answer = '<answer>' in completion_stripped and '</answer>' in completion_stripped
    
    # Check for unclosed tags (critical violation)
    unclosed_think = '<think>' in completion_stripped and '</think>' not in completion_stripped
    unclosed_tool = '<tool>' in completion_stripped and '</tool>' not in completion_stripped
    unclosed_answer = '<answer>' in completion_stripped and '</answer>' not in completion_stripped
    
    if unclosed_think or unclosed_tool or unclosed_answer:
        return -5.0  # Extreme penalty for unclosed tags
    
    # Check for no XML tags at all
    if not (has_think or has_tool or has_answer):
        return -4.0  # Very heavy penalty for no XML structure
    
    # Check for content that starts without proper XML structure
    # The output should start with whitespace + XML tag
    first_tag_match = re.search(r'<(think|tool|answer)>', completion_stripped)
    if not first_tag_match:
        return -3.0  # Heavy penalty for no proper opening tag
    
    # Check if there's significant content before the first XML tag
    content_before_tag = completion_stripped[:first_tag_match.start()].strip()
    if content_before_tag and len(content_before_tag) > 10:
        return -2.0  # Penalty for content before XML structure
    
    # If we get here, structure is acceptable
    # Check for perfect structure
    parser = SmolAgentParser()
    try:
        parsed = parser.parse(completion)
        if not parsed.parsing_error:
            return 0.5  # Reward for perfect structure
        else:
            return 0.0  # Acceptable but not perfect
    except:
        return -1.0  # Some issue with parsing 