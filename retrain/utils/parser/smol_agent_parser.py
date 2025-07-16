import re
from typing import NamedTuple, Optional, Any

from .parser import Parser

# This data structure will hold the parsed output from the LLM
# It's designed to capture the distinct elements of a smol-agent's response.
class ParsedSmolLLMOutput(NamedTuple):
    reasoning: Optional[str] = None
    tool_command: Optional[str] = None # Stores the raw JSON string from within the <tool> tag
    final_answer: Optional[str] = None
    parsing_error: Optional[str] = None # Captures issues if parsing fails

class SmolAgentParser(Parser):
    """
    Parses LLM output formatted in a smol-agent-like XML dialect.
    It also provides methods to format tool results and errors back into
    the expected string format for the LLM.
    """

    # Regex to find <tool>TOOL_COMMAND_JSON_STRING</tool>
    # - The content of the tool tag is expected to be a JSON string
    #   representing the tool name and its arguments.
    TOOL_PATTERN = re.compile(r'<tool>(?P<tool_command_content>.*?)</tool>', re.DOTALL)

    # Regex for <answer>THE_FINAL_ANSWER</answer>
    ANSWER_PATTERN = re.compile(r'<answer>(?P<answer_content>.*?)</answer>', re.DOTALL)

    # Regex for <think>REASONING_TEXT</think>
    # This tag can appear independently or alongside tool/answer tags.
    REASONING_PATTERN = re.compile(r'<think>(?P<reasoning_content>.*?)</think>', re.DOTALL)

    # Define structured patterns for re.match. These assume the string starts with these structures.
    # Pattern 1: Optional Reasoning, then Tool
    # Allows optional whitespace around and between tags.
    # The .* at the end consumes any trailing characters after the primary tag, which are ignored.
    REASONING_THEN_TOOL_PATTERN = re.compile(
        r"\s*(?:<think>(?P<reasoning_content>.*?)</think>\s*)?"
        r"<tool>(?P<tool_command_content>.*?)</tool>.*", 
        re.DOTALL
    )
    # Pattern 2: Optional Reasoning, then Answer
    REASONING_THEN_ANSWER_PATTERN = re.compile(
        r"\s*(?:<think>(?P<reasoning_content>.*?)</think>\s*)?"
        r"<answer>(?P<answer_content>.*?)</answer>.*",
        re.DOTALL
    )
    # Pattern 3: Tool only (no preceding reasoning block explicitly matched by this pattern)
    TOOL_ONLY_PATTERN = re.compile(
        r"\s*<tool>(?P<tool_command_content>.*?)</tool>.*", 
        re.DOTALL
    )
    # Pattern 4: Answer only
    ANSWER_ONLY_PATTERN = re.compile(
        r"\s*<answer>(?P<answer_content>.*?)</answer>.*",
        re.DOTALL
    )
    # Pattern 5: Reasoning only (entire string, after stripping, is just a reasoning block)
    REASONING_ONLY_PATTERN = re.compile(
        r"\s*<think>(?P<reasoning_content>.*?)</think>\s*$", # Anchor to end of string
        re.DOTALL
    )

    def parse(self, llm_output: str) -> ParsedSmolLLMOutput:
        """
        Parses the raw LLM output string.
        The method tries to match the beginning of the stripped llm_output against several structured patterns.
        It prioritizes patterns with <tool> or <answer> tags.
        If only a <think> tag is found, that's captured.
        If no primary tag structure is matched at the start, it's a parsing error.
        """
        stripped_output = llm_output.strip()

        # Priority 1: Try to match optional reasoning followed by a tool tag
        match = self.REASONING_THEN_TOOL_PATTERN.match(stripped_output)
        if match:
            reasoning = match.group("reasoning_content")
            tool_command = match.group("tool_command_content")
            return ParsedSmolLLMOutput(
                reasoning=reasoning.strip() if reasoning else None,
                tool_command=tool_command.strip() if tool_command else None
            )

        # Priority 2: Try to match optional reasoning followed by an answer tag
        match = self.REASONING_THEN_ANSWER_PATTERN.match(stripped_output)
        if match:
            reasoning = match.group("reasoning_content")
            answer = match.group("answer_content")
            return ParsedSmolLLMOutput(
                reasoning=reasoning.strip() if reasoning else None,
                final_answer=answer.strip() if answer else None
            )
        
        # Priority 3: Try to match tool tag only (if not caught by REASONING_THEN_TOOL)
        # This handles cases where there's no explicit <think> tag before <tool>
        match = self.TOOL_ONLY_PATTERN.match(stripped_output)
        if match:
            tool_command = match.group("tool_command_content")
            return ParsedSmolLLMOutput(
                tool_command=tool_command.strip() if tool_command else None
                # Reasoning is None as this pattern doesn't capture a separate reasoning block
            )

        # Priority 4: Try to match answer tag only
        match = self.ANSWER_ONLY_PATTERN.match(stripped_output)
        if match:
            answer = match.group("answer_content")
            return ParsedSmolLLMOutput(
                final_answer=answer.strip() if answer else None
                # Reasoning is None
            )

        # Priority 5: Try to match reasoning tag only (as the entire significant content)
        match = self.REASONING_ONLY_PATTERN.match(stripped_output)
        if match:
            reasoning = match.group("reasoning_content")
            # If only reasoning is found, it's a valid thought but not a complete action.
            return ParsedSmolLLMOutput(
                reasoning=reasoning.strip() if reasoning else None,
                parsing_error="No <tool> or <answer> tag found as the primary action after reasoning."
            )

        # If none of the structured patterns match from the beginning:
        # This means the output doesn't start with a recognized primary tag structure.
        # We can also check if ANY recognized tag is present, to give a more specific error.
        
        has_tool_tag = self.TOOL_PATTERN.search(stripped_output)
        has_answer_tag = self.ANSWER_PATTERN.search(stripped_output)
        has_reasoning_tag = self.REASONING_PATTERN.search(stripped_output)

        # Check for common XML formatting issues
        has_unclosed_tool = '<tool>' in stripped_output and not self.TOOL_PATTERN.search(stripped_output)
        has_unclosed_answer = '<answer>' in stripped_output and not self.ANSWER_PATTERN.search(stripped_output)
        has_unclosed_think = '<think>' in stripped_output and not self.REASONING_PATTERN.search(stripped_output)

        if not has_tool_tag and not has_answer_tag and not has_reasoning_tag:
            if has_unclosed_tool or has_unclosed_answer or has_unclosed_think:
                return ParsedSmolLLMOutput(parsing_error="Invalid LLM output: Found opening XML tags but they are not properly closed. Remember to close all tags (e.g., <tool>...</tool>, <answer>...</answer>, <think>...</think>). Each tag must have a matching closing tag with a forward slash.")
            else:
                return ParsedSmolLLMOutput(parsing_error="Invalid LLM output: No recognized <tool>, <answer>, or <think> tags found anywhere in the output. You must use one of these XML tag formats.")
        else:
            # One or more tags are present, but not in a recognized primary structure at the beginning.
            error_details = []
            
            if has_unclosed_tool or has_unclosed_answer or has_unclosed_think:
                error_details.append("Some XML tags are not properly closed - ensure all tags end with </tag>")
            
            # Check for multiple unclosed tool tags (common issue)
            if stripped_output.count('<tool>') > stripped_output.count('</tool>'):
                tool_starts = stripped_output.count('<tool>')
                tool_ends = stripped_output.count('</tool>')
                error_details.append(f"Found {tool_starts} <tool> opening tags but only {tool_ends} closing </tool> tags")
            
            if error_details:
                specific_error = ". ".join(error_details)
                return ParsedSmolLLMOutput(parsing_error=f"Invalid LLM output: {specific_error}. Each <tool> call must be properly closed with </tool> before starting a new one.")
            else:
                return ParsedSmolLLMOutput(parsing_error="Invalid LLM output: Found <tool>, <answer>, or <think> tags, but they are not structured correctly at the beginning of the response as the primary element.")

    def _escape_xml_content(self, content: str) -> str:
        """Basic XML escaping for content.
           For robust XML, a dedicated library is preferred.
        """
        return content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def format_llm_input(self, data: Any, context: Optional[str] = None) -> str:
        """
        Formats structured data (e.g., tool output, error message) into a string
        to be fed back to the LLM using <result> tags.

        Args:
            data: The data to format. Expected to be a string.
            context: Guides formatting. Expected values:
                     "tool_result_success", "tool_result_error", "env_error".
        """
        content_str = str(data)
        
        if context == "tool_result_error" and not content_str.lower().startswith("error:"):
            # If it's an error from a tool, ensure it starts with "Error:"
            content_str = f"Error: {content_str}"
        elif context == "env_error":
            # If it's an error from the environment/parser, prepend "Error:"
            content_str = f"Error: {content_str}"
        # For "tool_result_success", we use content_str as is.
        
        escaped_content = self._escape_xml_content(content_str)
        return f"<result>{escaped_content}</result>"

    # Convenience wrappers for specific formatting tasks
    def format_successful_tool_result(self, tool_output_str: str) -> str:
        """Formats a successful tool output string for the LLM."""
        return self.format_llm_input(tool_output_str, context="tool_result_success")

    def format_tool_error_result(self, tool_error_str: str) -> str:
        """Formats a tool error string (an error originating from the tool itself) for the LLM."""
        return self.format_llm_input(tool_error_str, context="tool_result_error")

    def format_environment_error(self, error_message: str) -> str:
        """Formats an environment-level error (e.g., LLM output parsing error) for the LLM."""
        return self.format_llm_input(error_message, context="env_error") 