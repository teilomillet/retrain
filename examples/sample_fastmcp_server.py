import asyncio
from fastmcp import FastMCP
from loguru import logger

# 1. Create FastMCP Server Instance
mcp_server = FastMCP(name="RetrainExampleServer")

# 2. Define a simple tool
@mcp_server.tool()
def get_server_time(timezone: str = "UTC") -> str:
    """Returns the current (simulated) server time in the specified timezone."""
    logger.info(f"[SampleFastMCPServer] Tool 'get_server_time' called with timezone: {timezone}")
    # In a real scenario, you'd get actual time. Here, it's fixed for predictability.
    return f"Simulated server time: 10:30 AM {timezone}"

# 3. Define a new calculator-like tool
@mcp_server.tool()
def perform_operation(operation: str, operand1: float, operand2: float) -> str:
    """Performs a specified arithmetic operation on two numbers. 
    Valid operations are 'add', 'subtract', 'multiply', 'divide'.
    """
    logger.info(f"[SampleFastMCPServer] Tool 'perform_operation' called with: {operation}, {operand1}, {operand2}")
    
    if operation == "add":
        result = operand1 + operand2
    elif operation == "subtract":
        result = operand1 - operand2
    elif operation == "multiply":
        result = operand1 * operand2
    elif operation == "divide":
        if operand2 == 0:
            logger.warning("[SampleFastMCPServer] Division by zero attempt.")
            return "Error: Division by zero."
        result = operand1 / operand2
    else:
        logger.warning(f"[SampleFastMCPServer] Invalid operation: {operation}")
        return f"Error: Invalid operation '{operation}'. Must be one of 'add', 'subtract', 'multiply', 'divide'."
    
    logger.info(f"[SampleFastMCPServer] 'perform_operation' result: {result}")
    return f"Result: {result}"

@mcp_server.tool()
def list_server_capabilities() -> dict:
    """Lists the (mocked) capabilities of this sample server."""
    logger.info("[SampleFastMCPServer] Tool 'list_server_capabilities' called.")
    return {
        "name": mcp_server.name,
        "description": "A sample FastMCP server for retrain examples.",
        "available_tools": [tool.name for tool in mcp_server.tools.values()]
    }

# 3. Run the server
if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 8765 # Using a slightly different port to avoid common conflicts
    
    logger.info(f"Starting SampleFastMCPServer on http://{HOST}:{PORT}")
    # Commenting out potentially problematic logging for linter peace
    # if hasattr(mcp_server, 'tools') and isinstance(mcp_server.tools, dict):
    #     logger.info("Available tools on this server (before FastMCPToolProvider prefixing):")
    #     for tool_name in mcp_server.tools: # Iterating dict keys
    #         logger.info(f"  - {tool_name}")
    # else:
    #     logger.info("Could not directly list tools from mcp_server.tools for pre-run logging.")
    logger.info(f"Server will expose tools like 'get_server_time' and 'list_server_capabilities'.")
    
    # To make it accessible via HTTP for the retrain example client
    mcp_server.run(transport="streamable-http", host=HOST, port=PORT)
    # Note: mcp_server.run() is blocking. 