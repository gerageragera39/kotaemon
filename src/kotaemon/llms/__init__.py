"""LLM implementations with optional providers loaded on demand."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseMessage": ("kotaemon.base.schema", "BaseMessage"),
    "HumanMessage": ("kotaemon.base.schema", "HumanMessage"),
    "AIMessage": ("kotaemon.base.schema", "AIMessage"),
    "SystemMessage": ("kotaemon.base.schema", "SystemMessage"),
    "BaseLLM": (".base", "BaseLLM"),
    "ChatLLM": (".chats.base", "ChatLLM"),
    "EndpointChatLLM": (".chats", "EndpointChatLLM"),
    "AzureChatOpenAI": (".chats", "AzureChatOpenAI"),
    "ChatOpenAI": (".chats", "ChatOpenAI"),
    "StructuredOutputChatOpenAI": (".chats", "StructuredOutputChatOpenAI"),
    "LCAnthropicChat": (".chats", "LCAnthropicChat"),
    "LCGeminiChat": (".chats", "LCGeminiChat"),
    "LCCohereChat": (".chats", "LCCohereChat"),
    "LCOllamaChat": (".chats", "LCOllamaChat"),
    "LCAzureChatOpenAI": (".chats", "LCAzureChatOpenAI"),
    "LCChatOpenAI": (".chats", "LCChatOpenAI"),
    "LlamaCppChat": (".chats", "LlamaCppChat"),
    "LLM": (".completions", "LLM"),
    "OpenAI": (".completions", "OpenAI"),
    "AzureOpenAI": (".completions", "AzureOpenAI"),
    "LlamaCpp": (".completions", "LlamaCpp"),
    "BasePromptComponent": (".prompts", "BasePromptComponent"),
    "PromptTemplate": (".prompts", "PromptTemplate"),
    "SimpleLinearPipeline": (".linear", "SimpleLinearPipeline"),
    "GatedLinearPipeline": (".linear", "GatedLinearPipeline"),
    "SimpleBranchingPipeline": (".branching", "SimpleBranchingPipeline"),
    "GatedBranchingPipeline": (".branching", "GatedBranchingPipeline"),
    "ManualSequentialChainOfThought": (".cot", "ManualSequentialChainOfThought"),
    "Thought": (".cot", "Thought"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
