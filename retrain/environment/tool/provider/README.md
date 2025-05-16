# Tool Providers (`retrain.environment.tool.providers`)

This subpackage defines the concept of "Tool Providers." Tool Providers are responsible for discovering, loading, or otherwise supplying a list of `BaseTool` instances that an environment can then make available for use, often by an LLM.

They offer a flexible way to source tools, whether they are manually defined, registered globally, or discovered from external services.

## Core Components

### 1. `ToolProvider` (in `base.py`)

`ToolProvider` is an abstract base class (ABC) that all specific tool provider implementations must inherit from. It defines a standard interface for tool discovery.

**Key Interface Members:**

*   `async def discover_tools(self, **kwargs: Any) -> List[BaseTool]`:
    *   An abstract asynchronous method that concrete providers must implement.
    *   Its purpose is to find, instantiate (if necessary), and return a list of objects that are instances of classes inheriting from `BaseTool`.
    *   `**kwargs`: Allows passing provider-specific arguments that might be needed for the discovery process (e.g., API clients, configuration details).

### 2. `ManualToolProvider` (in `manual_provider.py`)

`ManualToolProvider` is a straightforward, concrete implementation of `ToolProvider`. It allows you to directly provide a list of already instantiated `BaseTool` objects.

**Key Features:**

*   `__init__(self, tools: List[BaseTool])`:
    *   The constructor takes a list of `BaseTool` instances.
    *   It performs a check to ensure all provided items are indeed instances of `BaseTool`.
*   `async def discover_tools(self, **kwargs: Any) -> List[BaseTool]`:
    *   Simply returns the list of tools that was provided during its initialization.
    *   `**kwargs` are ignored by this provider.

**Use Cases:**

*   Providing tools that are already configured and instantiated elsewhere in the application.
*   Injecting mock or test-specific tool implementations into an environment.
*   Using tools that are created dynamically and are not part of the `GLOBAL_TOOL_REGISTRY` (defined in `retrain.environment.tool.registry`).

## Extensibility and Usage

Environments can be designed to accept one or more `ToolProvider` instances. This allows an environment to aggregate tools from various sources:

*   Tools registered via the `@tool` decorator (which can be wrapped in a provider if needed, or accessed directly by the environment).
*   Tools explicitly provided via `ManualToolProvider`.
*   Tools discovered dynamically from external systems, for which custom providers can be built. For example, the `retrain.environment.tool.providers.fastmcp` subpackage contains a `FastMCPToolProvider` designed to interface with a `fastmcp` service to discover its available tools.

This provider model promotes modularity by decoupling the environment's tool usage logic from the specifics of how and where those tools are defined or discovered.

## Example Scenario

An environment could be initialized with a `ManualToolProvider` for some basic, always-available tools, and also with a `FastMCPToolProvider` to dynamically fetch tools from a remote service:

```python
# Hypothetical environment setup
# from retrain.environment import BaseEnvironment, LLMAction, EnvironmentObservation
# from retrain.environment.tool import BaseTool, tool
# from retrain.environment.tool.providers import ManualToolProvider
# from retrain.environment.tool.providers.fastmcp import FastMCPToolProvider # Assuming it exists

# @tool
# class MyLocalTool(BaseTool):
#     async def execute(self, ...): ...

# local_tool_instance = MyLocalTool()
# manual_provider = ManualToolProvider(tools=[local_tool_instance])

# fastmcp_client = ... # Initialize your fastmcp client
# fastmcp_provider = FastMCPToolProvider(client=fastmcp_client)

# my_env = MyCoolEnvironment(tool_providers=[manual_provider, fastmcp_provider])
# await my_env.setup() # The environment would call discover_tools() on each provider
``` 