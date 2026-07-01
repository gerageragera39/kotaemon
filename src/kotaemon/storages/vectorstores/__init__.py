"""Vector-store implementations loaded on demand."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseVectorStore": (".base", "BaseVectorStore"),
    "ChromaVectorStore": (".chroma", "ChromaVectorStore"),
    "InMemoryVectorStore": (".in_memory", "InMemoryVectorStore"),
    "SimpleFileVectorStore": (".simple_file", "SimpleFileVectorStore"),
    "LanceDBVectorStore": (".lancedb", "LanceDBVectorStore"),
    "MilvusVectorStore": (".milvus", "MilvusVectorStore"),
    "QdrantVectorStore": (".qdrant", "QdrantVectorStore"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
