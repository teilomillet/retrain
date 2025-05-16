from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Optional

class BaseParsedOutput(NamedTuple):
    """
    A generic base for structured output from a parser.
    Individual parsers can return this or their own more specific NamedTuple.
    """
    parsing_error: Optional[str] = None
    # Potentially other common fields like 'raw_text' could be added if truly universal.

class Parser(ABC):
    """
    Abstract base class for parsers that convert between LLM string outputs
    and structured data, and format structured data back into strings for the LLM.
    """

    @abstractmethod
    def parse(self, llm_output: str) -> Any:
        """
        Parses the raw LLM output string into a structured representation.
        The concrete implementation should define the exact structure of the returned object
        (e.g., a specific NamedTuple like ParsedSmolLLMOutput).
        """
        pass

    @abstractmethod
    def format_llm_input(self, data: Any, context: Optional[str] = None) -> str:
        """
        Formats structured data into a string to be fed back to the LLM.

        Args:
            data: The structured data to format (e.g., tool output, error message).
            context: An optional string providing context about the data
                     (e.g., "tool_result_success", "tool_result_error", "env_error")
                     to guide formatting.
        """
        pass 