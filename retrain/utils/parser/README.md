# Parser Utilities (`retrain.utils.parser`)

This module provides utilities for parsing structured text output from Language Models (LLMs) and for formatting data back into string representations suitable for LLM input.

## Core Components

-   **`Parser` (in `parser.py`):**
    An abstract base class defining a common interface for parsers. It mandates a `parse` method to convert LLM string output to a structured format and a `format_llm_input` method to convert structured data back into a string for the LLM.

-   **`BaseParsedOutput` (in `parser.py`):**
    A generic `NamedTuple` that can serve as a base for specific structured outputs from parsers. It includes a `parsing_error` field.

-   **`SmolAgentParser` (in `smol_agent_parser.py`):**
    A concrete implementation of `BaseParser` designed to handle the XML-like dialect used by `smol-agents`. It parses LLM outputs containing tags like `<reasoning>`, `<tool>`, and `<answer>` into a `ParsedSmolLLMOutput` structure. It also formats tool results and errors into `<result>...</result>` tags for the LLM.

-   **`ParsedSmolLLMOutput` (in `smol_agent_parser.py`):**
    A `NamedTuple` that holds the structured data extracted by `SmolAgentParser`, including fields for reasoning, tool name, tool arguments (as a JSON string), final answer, and any parsing errors.

## Usage

Parsers from this module are typically used within `Environment` implementations to interpret an agent's actions or to prepare feedback for the agent.

To use a specific parser:

```python
from retrain.utils.parser import SmolAgentParser, ParsedSmolLLMOutput

parser = SmolAgentParser()
llm_response_text = "<reasoning>I should use a tool.</reasoning><tool name=\"my_tool\" args='{\"param\": \"value\"}'></tool>"

parsed_action: ParsedSmolLLMOutput = parser.parse(llm_response_text)

if parsed_action.tool_name:
    print(f"Tool call: {parsed_action.tool_name}")
    # ... execute tool ...
    tool_result_str = "Tool executed successfully."
    feedback_for_llm = parser.format_successful_tool_result(tool_result_str)
elif parsed_action.final_answer:
    print(f"Final answer: {parsed_action.final_answer}")
    # ... handle final answer ...
elif parsed_action.parsing_error:
    print(f"Parsing error: {parsed_action.parsing_error}")
    feedback_for_llm = parser.format_environment_error(parsed_action.parsing_error)

```

This structure allows for the addition of new parsers for different LLM output formats by inheriting from `BaseParser` and implementing its abstract methods. 