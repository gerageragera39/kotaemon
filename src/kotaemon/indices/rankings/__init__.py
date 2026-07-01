"""Ranking implementations loaded only when selected."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseReranking": (".base", "BaseReranking"),
    "CohereReranking": (".cohere", "CohereReranking"),
    "LLMReranking": (".llm", "LLMReranking"),
    "LLMScoring": (".llm_scoring", "LLMScoring"),
    "LLMTrulensScoring": (".llm_trulens", "LLMTrulensScoring"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
