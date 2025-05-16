from typing import Dict, Type, Callable, Tuple, Optional, Union
import inspect
import re
from loguru import logger

from .tool import Tool

# Stores: registration_key -> (ToolClass, registered_name, registered_description)
GLOBAL_TOOL_REGISTRY: Dict[str, Tuple[Type[Tool], str, str]] = {}

def _to_snake_case(name: str) -> str:
    """Converts a CamelCase name to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _perform_tool_registration(
    cls_to_decorate: Type[Tool], 
    name_param: Optional[str],
    description_param: Optional[str]
) -> Type[Tool]:
    """Internal helper to perform the actual registration logic."""
    if not inspect.isclass(cls_to_decorate) or not issubclass(cls_to_decorate, Tool):
        raise TypeError(
            f"The @tool decorator can only be used on classes that inherit from Tool. "
            f"Attempted to decorate {cls_to_decorate}."
        )

    actual_name = name_param if name_param is not None else _to_snake_case(cls_to_decorate.__name__)
    
    actual_description = description_param
    if actual_description is None:
        docstring = inspect.getdoc(cls_to_decorate)
        actual_description = docstring.strip() if docstring else ""
    
    if not actual_description and description_param is None: # No explicit desc and no docstring
        logger.warning(
            f"[ToolRegistry] Tool class {cls_to_decorate.__module__}.{cls_to_decorate.__name__} (registered as '{actual_name}') "
            f"has no docstring and no explicit description provided to @tool. "
            f"It will be registered with an empty description."
        )

    if actual_name in GLOBAL_TOOL_REGISTRY:
        existing_cls, _, _ = GLOBAL_TOOL_REGISTRY[actual_name]
        logger.warning(
            f"[ToolRegistry] Tool name '{actual_name}' is already registered for "
            f"{existing_cls.__module__}.{existing_cls.__name__}. "
            f"Overwriting with {cls_to_decorate.__module__}.{cls_to_decorate.__name__}."
        )
    
    GLOBAL_TOOL_REGISTRY[actual_name] = (cls_to_decorate, actual_name, actual_description)
    logger.debug(f"[ToolRegistry] Registered tool: '{actual_name}' -> {cls_to_decorate.__module__}.{cls_to_decorate.__name__}")
    return cls_to_decorate


def tool(
    _cls_or_none: Optional[Type[Tool]] = None, 
    *, 
    name: Optional[str] = None, 
    description: Optional[str] = None
) -> Union[Callable[[Type[Tool]], Type[Tool]], Type[Tool]]:
    """
    Decorator to register a tool class in the GLOBAL_TOOL_REGISTRY.

    Can be used as `@tool` (name/description derived automatically) or 
    `@tool(name=..., description=...)`.

    The decorated class must inherit from Tool.
    - Name: Derived from class name (snake_case) if not provided.
    - Description: Derived from class docstring if not provided.

    Args (if used with parentheses):
        name: Explicit name for the tool.
        description: Explicit description for the tool.

    Returns:
        Either the decorated class (if @tool used directly) or a decorator function.
    """
    if _cls_or_none is not None and inspect.isclass(_cls_or_none):
        return _perform_tool_registration(_cls_or_none, name_param=name, description_param=description)
    else:
        def decorator_wrapper(cls_to_decorate: Type[Tool]) -> Type[Tool]:
            return _perform_tool_registration(cls_to_decorate, name_param=name, description_param=description)
        return decorator_wrapper


def get_tool_info(tool_registration_key: str) -> Tuple[Type[Tool], str, str]:
    """
    Retrieves registered tool information (Class, name, description) by its registration key.

    Args:
        tool_registration_key: The key (usually snake_case name) the tool class was registered under.

    Returns:
        A tuple: (ToolClass, registered_name, registered_description).

    Raises:
        KeyError: If no tool class is registered with the given key.
    """
    if tool_registration_key not in GLOBAL_TOOL_REGISTRY:
        raise KeyError(
            f"No tool class registered with key: '{tool_registration_key}'. "
            f"Available registered tools: {list(GLOBAL_TOOL_REGISTRY.keys())}"
        )
    return GLOBAL_TOOL_REGISTRY[tool_registration_key] 