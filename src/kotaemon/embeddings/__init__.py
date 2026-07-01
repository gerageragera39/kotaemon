"""Embedding providers with optional implementations loaded on demand."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseEmbeddings": (".base", "BaseEmbeddings"),
    "EndpointEmbeddings": (".endpoint_based", "EndpointEmbeddings"),
    "TeiEndpointEmbeddings": (".tei_endpoint_embed", "TeiEndpointEmbeddings"),
    "LCOpenAIEmbeddings": (".langchain_based", "LCOpenAIEmbeddings"),
    "LCAzureOpenAIEmbeddings": (".langchain_based", "LCAzureOpenAIEmbeddings"),
    "LCCohereEmbeddings": (".langchain_based", "LCCohereEmbeddings"),
    "LCHuggingFaceEmbeddings": (".langchain_based", "LCHuggingFaceEmbeddings"),
    "LCGoogleEmbeddings": (".langchain_based", "LCGoogleEmbeddings"),
    "LCMistralEmbeddings": (".langchain_based", "LCMistralEmbeddings"),
    "OpenAIEmbeddings": (".openai", "OpenAIEmbeddings"),
    "AzureOpenAIEmbeddings": (".openai", "AzureOpenAIEmbeddings"),
    "FastEmbedEmbeddings": (".fastembed", "FastEmbedEmbeddings"),
    "VoyageAIEmbeddings": (".voyageai", "VoyageAIEmbeddings"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
