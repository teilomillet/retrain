from typing import Dict, Any, Union, Literal

from retrain.environment.tool.tool import Tool
from retrain.environment.tool.registry import tool # For registering

@tool
class SimpleCalculatorTool(Tool):
    """
    Performs addition, subtraction, multiplication, or division on two numbers. Operations must be 'add', 'subtract', 'multiply', or 'divide'.
    """
    # Removed class-level 'name' and 'description' attributes.
    # The instance's name and description will be set by Tool.__init__ via values from @tool decorator
    # and used by self.name and self.description (properties from base Tool class) in get_schema.

    async def execute(self, tool_input: Dict[str, Any]) -> Union[float, str]:
        """
        Executes the calculation.
        Args:
            tool_input: A dictionary containing:
                - operation (Literal['add', 'subtract', 'multiply', 'divide']): The operation to perform.
                - a (Union[int, float]): The first number.
                - b (Union[int, float]): The second number.
        Returns:
            The result of the calculation, or an error message string if input is invalid.
        """
        operation = tool_input.get("operation")
        a = tool_input.get("a")
        b = tool_input.get("b")

        if not isinstance(operation, str) or not operation in ["add", "subtract", "multiply", "divide"]:
            return "Error: Invalid 'operation'. Must be 'add', 'subtract', 'multiply', or 'divide'."
        
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return "Error: Invalid input for 'a' or 'b'. Both must be numbers."

        a = float(a)
        b = float(b)

        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero."
            return a / b
        else:
            # This case should ideally be caught by the validation above, but as a safeguard:
            return "Error: Unknown operation. This should not happen."

    async def get_schema(self) -> Dict[str, Any]:
        """
        Returns the JSON schema for this tool, compatible with OpenAI's function calling format.
        """
        return {
            "name": self.name,  # This will now use the registered name (e.g., simple_calculator_tool)
            "description": self.description, # This will use the registered description (from docstring)
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform.",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    },
                    "a": {
                        "type": "number",
                        "description": "The first operand."
                    },
                    "b": {
                        "type": "number",
                        "description": "The second operand (divisor in case of division)."
                    }
                },
                "required": ["operation", "a", "b"]
            }
        } 