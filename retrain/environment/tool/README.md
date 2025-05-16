# Tool Definition and Registration (`retrain.environment.tool`)

This subpackage helps `retrain` define, find, and use "tools." Think of tools as special abilities for language models, allowing them to interact with the outside world, run specific tasks, or get fresh information.

## What's Inside?

This part of the project organizes how tools are created and made available.

### 1. `Tool` (the blueprint for actions - in `tool.py`)

The `Tool` class is like a template. Any new tool you want to create (e.g., a tool to search the web or a tool to calculate something) must follow this template.

*   **Key Idea:** Every tool needs a `name` (so we can call it) and a `description` (so the language model knows what it does). Most importantly, each tool must have an `execute` method, which contains the actual logic for what the tool does when called.

### 2. Tool Registration (finding and organizing tools - in `registry.py`)

This part keeps track of all available tools. It uses a special `@tool` decorator to easily add new tools to a central list (the `GLOBAL_TOOL_REGISTRY`).

*   **Key Idea:** When you create a new tool class and mark it with `@tool`, the system automatically notes its name (usually from the class name) and its description (from the class's help text/docstring). This makes it easy for other parts of `retrain` to find and use the tool.
*   There's also a helper `get_tool_info` to look up a tool's details from the registry.

### 3. Error Handling (what happens when tools go wrong - in `error.py`)

Sometimes, a tool might run into a problem while trying to do its job.

*   **Key Idea:** If a tool fails, it should raise a `ToolExecutionError`. This is a specific type of error that helps the system understand that the problem happened within a tool.

## Example: A Simple "Echo" Tool

Here's how you might define a basic tool that just repeats back what you tell it:

```python
# in my_custom_tools.py
from typing import Dict, Any, Optional
from retrain.environment.tool import Tool, tool # Main components
from retrain.environment.tool.error import ToolExecutionError # For error handling

@tool # This "registers" our tool
class SimpleEchoTool(Tool):
    """
    A simple tool that echoes back its input.
    It expects a 'message' key in the tool_input dictionary.
    """
    # The 'name' will be 'simple_echo_tool' (from the class name)
    # The 'description' is taken from the text above.

    async def execute(self, tool_input: Optional[Dict[str, Any]]) -> Any:
        if tool_input is None or "message" not in tool_input:
            raise ToolExecutionError("SimpleEchoTool requires a 'message' in tool_input.")
        
        message_to_echo = tool_input["message"]
        return f"Echo: {message_to_echo}"

# After this, other parts of the system can find and use 'simple_echo_tool'.
```

This system allows `retrain` to easily work with a variety of tools, making language models more powerful and interactive. 