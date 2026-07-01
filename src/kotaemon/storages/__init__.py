"""Storage backends with optional providers loaded on demand."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseDocumentStore": (".docstores.base", "BaseDocumentStore"),
    "InMemoryDocumentStore": (".docstores.in_memory", "InMemoryDocumentStore"),
    "ElasticsearchDocumentStore": (
        ".docstores.elasticsearch",
        "ElasticsearchDocumentStore",
    ),
    "SimpleFileDocumentStore": (".docstores.simple_file", "SimpleFileDocumentStore"),
    "LanceDBDocumentStore": (".docstores.lancedb", "LanceDBDocumentStore"),
    "BaseVectorStore": (".vectorstores.base", "BaseVectorStore"),
    "ChromaVectorStore": (".vectorstores.chroma", "ChromaVectorStore"),
    "InMemoryVectorStore": (".vectorstores.in_memory", "InMemoryVectorStore"),
    "SimpleFileVectorStore": (".vectorstores.simple_file", "SimpleFileVectorStore"),
    "LanceDBVectorStore": (".vectorstores.lancedb", "LanceDBVectorStore"),
    "MilvusVectorStore": (".vectorstores.milvus", "MilvusVectorStore"),
    "QdrantVectorStore": (".vectorstores.qdrant", "QdrantVectorStore"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
