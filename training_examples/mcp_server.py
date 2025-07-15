#!/usr/bin/env python3
"""
Simple MCP Server for Training

This script starts a FastMCP server with tools for training.
Follows the same pattern as examples/sample_fastmcp_server.py
"""

import asyncio
from fastmcp import FastMCP
from loguru import logger

# Create FastMCP Server Instance
mcp_server = FastMCP(name="RetrainMCPServer")

# Define tools for training
@mcp_server.tool()
def get_server_time(timezone: str = "UTC") -> str:
    """Returns the current server time in the specified timezone."""
    logger.info(f"[MCPServer] Tool 'get_server_time' called with timezone: {timezone}")
    return f"Current server time: 10:30 AM {timezone}"

@mcp_server.tool()
def perform_operation(operation: str, operand1: float, operand2: float) -> str:
    """Performs arithmetic operations on two numbers.
    Valid operations: 'add', 'subtract', 'multiply', 'divide'
    """
    logger.info(f"[MCPServer] Tool 'perform_operation' called: {operation}({operand1}, {operand2})")
    
    if operation == "add":
        result = operand1 + operand2
    elif operation == "subtract":
        result = operand1 - operand2
    elif operation == "multiply":
        result = operand1 * operand2
    elif operation == "divide":
        if operand2 == 0:
            return "Error: Division by zero."
        result = operand1 / operand2
    else:
        return f"Error: Invalid operation '{operation}'. Must be 'add', 'subtract', 'multiply', or 'divide'."
    
    return f"Result: {result}"

@mcp_server.tool()
def list_server_capabilities() -> dict:
    """Lists the capabilities and available tools of this server."""
    logger.info("[MCPServer] Tool 'list_server_capabilities' called.")
    return {
        "name": mcp_server.name,
        "description": "A training server for MCP tool usage",
        "available_tools": [
            "get_server_time - Get current server time",
            "perform_operation - Perform arithmetic operations",
            "list_server_capabilities - List server capabilities"
        ],
        "supported_operations": ["add", "subtract", "multiply", "divide"]
    }

@mcp_server.tool()
def search_tools(query: str) -> dict:
    """Search for tools based on a query."""
    logger.info(f"[MCPServer] Tool 'search_tools' called with query: {query}")
    
    all_tools = {
        "time": ["get_server_time"],
        "calculate": ["perform_operation"],
        "math": ["perform_operation"],
        "arithmetic": ["perform_operation"],
        "capabilities": ["list_server_capabilities"],
        "search": ["search_tools"]
    }
    
    query_lower = query.lower()
    matching_tools = []
    
    for keyword, tools in all_tools.items():
        if keyword in query_lower:
            matching_tools.extend(tools)
    
    return {
        "query": query,
        "matching_tools": list(set(matching_tools)),
        "all_tools": list(all_tools.keys())
    }

# Run the server
if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 8765
    
    logger.info(f"Starting MCP Training Server on http://{HOST}:{PORT}")
    logger.info("Available tools:")
    logger.info("  - get_server_time(timezone)")
    logger.info("  - perform_operation(operation, operand1, operand2)")
    logger.info("  - list_server_capabilities()")
    logger.info("  - search_tools(query)")
    logger.info("")
    logger.info("Server is ready for training. Keep this running while training.")
    
    try:
        mcp_server.run(transport="streamable-http", host=HOST, port=PORT)
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"Server error: {e}") 