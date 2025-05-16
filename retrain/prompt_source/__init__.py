"""Makes PromptSource and its factory available under the retrain.prompt_source namespace."""

from ..utils.prompt_source import PromptSource, CfgPromptSource, get_prompt_source

__all__ = [
    "PromptSource",
    "CfgPromptSource",
    "get_prompt_source",
] 